"""Import of the user's own data: a land-cover raster plus optional DEM and vector layers.

Input is a folder (or a ZIP archive with the same content).  File roles are
recognised by name, case-insensitively:

=====================  ======================================================
role                   file name contains
=====================  ======================================================
land cover (required)  ``landcover``, ``land_cover``, ``lulc``, ``lc``, ``cover``
DEM (optional)         ``dem``, ``elev``, ``srtm``, ``height``, ``relief``
protected areas        ``protect``, ``oopt``, ``core``, ``reserve``, ``park``
roads                  ``road``, ``highway``
rivers                 ``river``, ``water``, ``stream``
settlements            ``settle``, ``place``, ``village``, ``town``, ``city``
class table            ``classes.json`` (optional)
=====================  ======================================================

Rasters: GeoTIFF or any GDAL format.  Vectors: GeoJSON, GeoPackage, Shapefile.
If only one raster is supplied it is the land cover.

Land-cover codes are interpreted through a *scheme*:

``ecocorridor``  the codes already are EcoCorridor classes (1 water, 2 wetland,
                 3/4 forest, 5 meadow, 6 cropland, 7 shrub, 8 built-up, 9 road,
                 10 bare ground) -- the default;
``worldcover``   ESA WorldCover codes (10, 20, ... 100);
``custom``       codes are mapped by ``"map"`` in ``classes.json``.

``classes.json`` may also override names, colours and the baseline resistance of
EcoCorridor classes.  Example::

    {"map": {"111": 4, "211": 6, "112": 8},
     "classes": [{"code": 4, "name": "Лес", "color": "#2F6B4C", "resistance": 5}]}

A geographic raster is reprojected to the UTM zone of its centre; a projected
one keeps its CRS when the unit is the metre.  A raster larger than
``MAX_CELLS`` is coarsened (majority resampling) to the nearest sensible
resolution, or to ``resolution_m`` when it is given.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box

from .dataset import CACHE_DIR, LandCoverClass, Project
from .online import (
    MAX_CELLS,
    NICE_RESOLUTIONS,
    ONLINE_CLASSES,
    WORLDCOVER_TO_CLASS,
    Extent,
    OnlineError,
    habitat_cores,
    utm_crs,
)

SCHEMES = ("ecocorridor", "worldcover", "custom")
RASTER_EXT = {".tif", ".tiff", ".img", ".vrt", ".asc"}
VECTOR_EXT = {".geojson", ".json", ".gpkg", ".shp"}
MAX_ZIP_BYTES = 400 * 1024 * 1024

ROLE_WORDS = {
    "landcover": ("landcover", "land_cover", "lulc", "cover", "lc"),
    "dem": ("dem", "elev", "srtm", "height", "relief"),
    "cores": ("protect", "oopt", "core", "reserve", "park"),
    "roads": ("road", "highway"),
    "rivers": ("river", "water", "stream"),
    "settlements": ("settle", "place", "village", "town", "city"),
}
VALID_CODES = set(range(1, 11))


class ImportError_(OnlineError):
    """A problem with the user's files, with a message meant to be shown as is."""


def _role(path: Path, role: str) -> bool:
    stem = path.stem.lower()
    if role == "landcover" and (stem == "lc" or stem.startswith("lc_") or stem.endswith("_lc")):
        return True
    return any(word in stem for word in ROLE_WORDS[role] if word != "lc")


def find_files(directory: Path) -> dict[str, Path]:
    """Assign roles to the files of ``directory`` (searched recursively)."""
    files = [p for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.startswith(("._", "~"))]
    rasters = [p for p in files if p.suffix.lower() in RASTER_EXT]
    vectors = [p for p in files if p.suffix.lower() in VECTOR_EXT and p.name.lower() != "classes.json"]
    found: dict[str, Path] = {}

    dem = [p for p in rasters if _role(p, "dem")]
    if dem:
        found["dem"] = dem[0]
    land = [p for p in rasters if p not in dem and _role(p, "landcover")]
    if not land:
        land = [p for p in rasters if p not in dem]
    if not land:
        raise ImportError_("Не найден растр землепокрова. Назовите файл landcover.tif (или передайте единственный растр).")
    if len(land) > 1:
        raise ImportError_(
            "Найдено несколько растров землепокрова: " + ", ".join(p.name for p in land) + ". Оставьте один или переименуйте лишние."
        )
    found["landcover"] = land[0]

    # Roles are checked in a fixed order so a "protected_areas_park.geojson" does not become a road.
    for role in ("cores", "roads", "rivers", "settlements"):
        for path in vectors:
            if path in found.values():
                continue
            if _role(path, role):
                found[role] = path
                break
    table = [p for p in files if p.name.lower() == "classes.json"]
    if table:
        found["classes"] = table[0]
    return found


def read_class_table(path: Path | None, scheme: str) -> tuple[list[LandCoverClass], dict[int, int] | None]:
    """Class list (defaults + overrides from ``classes.json``) and the code map for ``custom``."""
    classes = {c.code: c for c in ONLINE_CLASSES}
    code_map: dict[int, int] | None = None
    if path is not None:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ImportError_(f"classes.json не читается: {exc}") from exc
        for item in data.get("classes", []):
            try:
                code = int(item["code"])
                base = classes.get(code)
                classes[code] = LandCoverClass(
                    code=code,
                    name=str(item.get("name") or (base.name if base else f"Класс {code}")),
                    color=str(item.get("color") or (base.color if base else "#888888")),
                    resistance=float(item.get("resistance", base.resistance if base else 20.0)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ImportError_(f"classes.json: неверная запись класса {item!r}") from exc
        if data.get("map"):
            try:
                code_map = {int(k): int(v) for k, v in data["map"].items()}
            except (TypeError, ValueError) as exc:
                raise ImportError_("classes.json: \"map\" должен задавать пары «код источника → класс 1…10».") from exc
            bad = sorted({v for v in code_map.values() if v not in VALID_CODES and v != 0})
            if bad:
                raise ImportError_(f"classes.json: целевые классы должны быть 1…10 (или 0 — пропуск), найдено {bad}.")
    if scheme == "custom" and not code_map:
        raise ImportError_("Схема «custom» требует classes.json с полем \"map\".")
    return [classes[c] for c in sorted(classes)], code_map


def _choose_grid(src: rasterio.DatasetReader, resolution_m: float | None) -> tuple[str, float, float, float, float, float, list[str]]:
    """Target CRS, bounds and resolution for the project grid."""
    notes: list[str] = []
    if src.crs is None:
        raise ImportError_("У растра землепокрова не задана система координат — задайте CRS в ГИС и сохраните заново.")
    src_crs = CRS.from_user_input(src.crs)
    west, south, east, north = src.bounds
    if src_crs.is_projected and abs(src_crs.axis_info[0].unit_conversion_factor - 1.0) < 1e-9:
        crs = src_crs.to_string()
        left, bottom, right, top = west, south, east, north
        native = abs(src.res[0])
    else:
        to_ll = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform((west + east) / 2.0, (south + north) / 2.0)
        crs = utm_crs(lon, lat)
        transform, width, height = calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        left, top = transform.c, transform.f
        right, bottom = left + width * transform.a, top + height * transform.e
        native = abs(transform.a)
        notes.append(f"Растр переведён в {crs}.")
    if resolution_m:
        res = float(resolution_m)
    else:
        res = native
        cells = (right - left) * (top - bottom) / res**2
        if cells > MAX_CELLS:
            need = (cells / MAX_CELLS) ** 0.5 * res
            res = next((r for r in NICE_RESOLUTIONS if r >= need), need)
            notes.append(f"Растр велик ({cells / 1e6:.1f} млн ячеек) — разрешение огрублено до {res:g} м (мажоритарная передискретизация).")
    if res < native * 0.999:
        notes.append(f"Заданное разрешение {res:g} м мельче исходного {native:g} м — новой информации оно не добавляет.")
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    if width < 10 or height < 10:
        raise ImportError_("Растр слишком мал: меньше 10 ячеек по стороне. Уменьшите разрешение.")
    if width * height > MAX_CELLS * 1.2:
        raise ImportError_(f"Слишком много ячеек ({width * height:,}); задайте разрешение грубее.".replace(",", " "))
    return crs, left, top - height * res, left + width * res, top, res, notes


def _warp(path: Path, crs: str, left: float, top: float, res: float, width: int, height: int, resampling: Resampling, dtype: str, fill: float, band: int = 1) -> np.ndarray:
    out = np.full((height, width), fill, dtype=dtype)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ImportError_(f"У растра {path.name} не задана система координат.")
        nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=out,
            dst_transform=from_origin(left, top, res, res),
            dst_crs=crs,
            src_nodata=nodata,
            dst_nodata=fill,
            resampling=resampling,
        )
    return out


def _vector(path: Path | None, crs: str, frame) -> gpd.GeoDataFrame:
    if path is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - any driver error is a user-facing message
        raise ImportError_(f"Слой {path.name} не читается: {exc}") from exc
    if gdf.crs is None:
        raise ImportError_(f"У слоя {path.name} не задана система координат.")
    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid().intersection(frame)
    return gdf[~gdf.geometry.is_empty].reset_index(drop=True)


def _text_column(gdf: gpd.GeoDataFrame, column: str, candidates: tuple[str, ...], default: str = "") -> None:
    if column in gdf:
        return
    for name in candidates:
        if name in gdf:
            gdf[column] = gdf[name].astype(str)
            return
    gdf[column] = default


def import_directory(
    directory: str | Path,
    name: str | None = None,
    scheme: str = "ecocorridor",
    resolution_m: float | None = None,
    min_core_ha: float = 100.0,
    progress: Callable[[str, float, str], None] | None = None,
    save: bool = True,
) -> Project:
    """Build a :class:`Project` from the files in ``directory``."""
    say = progress or (lambda *_: None)
    if scheme not in SCHEMES:
        raise ImportError_(f"Схема должна быть одной из: {', '.join(SCHEMES)}.")
    directory = Path(directory)
    say("scan", 0.05, "Ищу файлы…")
    files = find_files(directory)
    classes, code_map = read_class_table(files.get("classes"), scheme)

    say("landcover", 0.15, "Читаю землепокров…")
    with rasterio.open(files["landcover"]) as src:
        crs, left, bottom, right, top, res, notes = _choose_grid(src, resolution_m)
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    mode = Resampling.mode if res > 1.0 else Resampling.nearest
    raw = _warp(files["landcover"], crs, left, top, res, width, height, mode, "int32", -1)

    if scheme == "ecocorridor":
        lut = {c: c for c in VALID_CODES}
    elif scheme == "worldcover":
        lut = dict(WORLDCOVER_TO_CLASS)
    else:
        lut = {k: v for k, v in (code_map or {}).items()}
    if code_map and scheme != "custom":
        lut.update(code_map)
    landcover = np.zeros(raw.shape, dtype=np.uint8)
    for src_code, target in lut.items():
        landcover[raw == src_code] = target
    outside = raw < 0
    unknown = (~outside) & (landcover == 0) & (raw != 0)
    unmapped = sorted({int(v) for v in np.unique(raw[unknown])})[:12]
    valid_share = float((~outside).mean())
    if valid_share < 0.05:
        raise ImportError_("Растр не попал в расчётную сетку: проверьте систему координат файла.")
    if unknown.mean() > 0.2:
        raise ImportError_(
            f"Более 20 % ячеек имеют коды, не входящие в схему «{scheme}»: {unmapped}. "
            "Выберите другую схему или добавьте \"map\" в classes.json."
        )
    if unknown.any():
        notes.append(f"Коды без сопоставления ({unmapped}) считаются «неизвестно» (умеренная проходимость).")
    if outside.any():
        notes.append(f"Вне покрытия растра {outside.mean() * 100:.1f} % сетки — это «неизвестно».")

    say("dem", 0.40, "Читаю рельеф…")
    if "dem" in files:
        dem = _warp(files["dem"], crs, left, top, res, width, height, Resampling.bilinear, "float64", np.nan)
        if not np.isfinite(dem).any():
            notes.append("Рельеф не пересекается с землепокровом — уклоны приняты равными нулю.")
            dem = np.zeros((height, width))
    else:
        dem = np.zeros((height, width))
        notes.append("Рельеф (DEM) не задан — уклоны приняты равными нулю; сопротивление определяют землепокров и дороги.")

    say("vectors", 0.55, "Читаю векторные слои…")
    frame = box(left, bottom, right, top)
    roads = _vector(files.get("roads"), crs, frame)
    rivers = _vector(files.get("rivers"), crs, frame)
    settlements = _vector(files.get("settlements"), crs, frame)
    cores = _vector(files.get("cores"), crs, frame)
    for gdf in (roads, rivers):
        if len(gdf):
            _text_column(gdf, "name", ("NAME", "Name", "name:ru"))
            _text_column(gdf, "kind", ("highway", "type", "fclass", "class"), "—")
    if len(settlements):
        _text_column(settlements, "name", ("NAME", "Name", "name:ru"))
        _text_column(settlements, "kind", ("place", "type", "fclass"), "—")
        if "population" not in settlements:
            settlements["population"] = 0
    if len(roads) and (roads.geom_type.isin(["Polygon", "MultiPolygon"])).any():
        raise ImportError_("Слой дорог должен состоять из линий, а не из полигонов.")

    cores_note = f"ООПТ из файла {files['cores'].name}" if "cores" in files else ""
    if len(cores):
        cores = cores[cores.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
        _text_column(cores, "name", ("NAME", "Name", "name:ru"))
        cores["name"] = [n if n and n != "nan" else f"Ядро {i + 1}" for i, n in enumerate(cores["name"])]
        _text_column(cores, "kind", ("type", "desig", "category", "class"), "ядро")
        cores["area_ha"] = (cores.geometry.area / 10_000.0).round(1)
        small = cores["area_ha"] < min_core_ha
        if small.any():
            notes.append(f"Отброшено ядер меньше {min_core_ha:g} га: {int(small.sum())}.")
            cores = cores[~small].reset_index(drop=True)
    if len(cores) < 2:
        extent = Extent(crs, left, bottom, right, top, res, width, height, _bbox_lonlat(crs, left, bottom, right, top))
        notes.append(
            "В загруженных данных меньше двух ядер — ядрами выбраны крупнейшие лесные и водно-болотные массивы землепокрова."
            if "cores" in files
            else "Слой ООПТ не задан — ядрами выбраны крупнейшие лесные и водно-болотные массивы землепокрова."
        )
        cores = habitat_cores(landcover, extent, max(min_core_ha, 50.0))
        cores_note = "крупнейшие лесные массивы землепокрова"
        if len(cores) < 2:
            notes.append("Найдено меньше двух ядер — коридорная сеть не будет построена. Добавьте слой ООПТ.")

    digest = hashlib.sha1()
    for key in sorted(files):
        digest.update(key.encode())
        digest.update(files[key].read_bytes()[:8_000_000])
    digest.update(f"{scheme}|{resolution_m}|{min_core_ha}|{name}".encode())
    project_id = "u" + digest.hexdigest()[:10]

    title = name or files["landcover"].stem.replace("_", " ")
    meta = {
        "imported": True,
        "scheme": scheme,
        "source_files": {role: path.name for role, path in files.items()},
        "core_source": cores_note or "слой ООПТ пользователя",
        "warnings": notes,
        "area_km2": round(width * height * res**2 / 1e6, 1),
        "sources": ["Данные пользователя: " + ", ".join(sorted({p.name for p in files.values()}))],
        "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    project = Project(
        id=project_id,
        name=f"{title} · {width * res / 1000:.0f} × {height * res / 1000:.0f} км",
        description="Проект собран из файлов пользователя.",
        crs=crs,
        transform=from_origin(left, top, res, res),
        width=width,
        height=height,
        res=res,
        dem=dem,
        landcover=landcover,
        classes=classes,
        cores=cores,
        roads=roads,
        rivers=rivers,
        settlements=settlements,
        region=gpd.GeoDataFrame(geometry=[frame], crs=crs),
        synthetic=False,
        meta=meta,
    )
    if save:
        say("save", 0.92, "Сохраняю в кэш…")
        try:
            project.to_directory(CACHE_DIR / project_id)
        except OSError as exc:
            project.meta["warnings"].append(f"Не удалось сохранить кэш: {exc}")
    return project


def _bbox_lonlat(crs: str, left: float, bottom: float, right: float, top: float) -> tuple[float, float, float, float]:
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys = zip(*[to_ll.transform(x, y) for x, y in ((left, bottom), (right, bottom), (right, top), (left, top))])
    return min(xs), min(ys), max(xs), max(ys)


def import_zip(
    payload: bytes,
    name: str | None = None,
    scheme: str = "ecocorridor",
    resolution_m: float | None = None,
    min_core_ha: float = 100.0,
    progress: Callable[[str, float, str], None] | None = None,
) -> Project:
    """Same as :func:`import_directory`, for an uploaded ZIP archive."""
    import io

    if len(payload) > MAX_ZIP_BYTES:
        raise ImportError_("Архив больше 400 МБ.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ImportError_("Это не ZIP-архив. Упакуйте landcover.tif и слои в один .zip.") from exc
    with tempfile.TemporaryDirectory(prefix="ecoimport_") as tmp:
        root = Path(tmp).resolve()
        total = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            target = (root / info.filename).resolve()
            if root not in target.parents:  # zip-slip
                raise ImportError_("В архиве недопустимые пути файлов.")
            total += info.file_size
            if total > 2 * MAX_ZIP_BYTES:
                raise ImportError_("Содержимое архива слишком велико.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as dest:
                dest.write(source.read())
        return import_directory(root, name, scheme, resolution_m, min_core_ha, progress)
