"""Online loading of an arbitrary territory from open data.

Sources (all free, no API key):

``Nominatim``            place name -> bounding box (OpenStreetMap)
``ESA WorldCover``       10 m land cover, 2021, v200 (Cloud-Optimised GeoTIFF on AWS)
``Copernicus DEM GLO-30``  30 m surface model (COG on AWS; GLO-90 as fall-back)
``Overpass API``         roads, rivers, settlements and protected areas (OpenStreetMap)

The rasters are streamed window-by-window straight from the COGs and warped
into a UTM grid chosen from the territory's centre, so a 60 x 45 km extent
downloads only the pixels it needs.  The result is a regular
:class:`~ecocorridor.dataset.Project`, cached on disk in the same layout as the
demo project.

Known limitations (also recorded in ``meta.json`` of every loaded project):

* WorldCover is a single-date, 11-class product (overall accuracy 76.7 % on a
  global scale); coniferous / broadleaf forest are not separated, so the whole
  tree cover is mapped to one forest class.
* Downsampling to the working resolution takes the majority class, which drops
  thin or small classes (villages, narrow streams).  Roads are therefore burned
  in from OpenStreetMap; small settlements enter only as the ``place`` points.
* Copernicus DEM is a surface model: canopy height is included.
* OpenStreetMap completeness and tagging of protected areas varies by country.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import linemerge, polygonize, unary_union
from shapely.validation import make_valid

from . import __version__
from .dataset import CACHE_DIR, LandCoverClass, Project

SCHEMA_VERSION = 3  # bump when the loader changes what it produces (invalidates the cache)

USER_AGENT = f"EcoCorridor/{__version__} (+https://github.com/DmitryLeLeE/eco_corridor)"
NOMINATIM_URL = os.environ.get("ECOCORRIDOR_NOMINATIM", "https://nominatim.openstreetmap.org/search")
OVERPASS_URLS = [
    u.strip()
    for u in os.environ.get(
        "ECOCORRIDOR_OVERPASS",
        "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter",
    ).split(",")
    if u.strip()
]
WORLDCOVER_URL = os.environ.get(
    "ECOCORRIDOR_WORLDCOVER_URL",
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif",
)
COPDEM30_URL = os.environ.get(
    "ECOCORRIDOR_COPDEM30_URL",
    "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif",
)
COPDEM90_URL = os.environ.get(
    "ECOCORRIDOR_COPDEM90_URL",
    "https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{tile}_DEM/Copernicus_DSM_COG_30_{tile}_DEM.tif",
)

MAX_AREA_KM2 = 15_000.0  # larger requests are cropped to a centred square of this area
MIN_SIDE_KM = 20.0
MAX_CELLS = 700_000
TARGET_CELLS = 300_000  # automatic resolution aims at roughly this many cells
NICE_RESOLUTIONS = (30.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 500.0, 1000.0)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "60",
    "AWS_NO_SIGN_REQUEST": "YES",
}

# ESA WorldCover class -> EcoCorridor class.
WORLDCOVER_TO_CLASS = {
    10: 4,  # tree cover            -> forest
    20: 7,  # shrubland             -> open woodland / shrub
    30: 5,  # grassland             -> meadows and pastures
    40: 6,  # cropland              -> arable land
    50: 8,  # built-up              -> built-up
    60: 10,  # bare / sparse        -> rock and bare ground
    70: 10,  # snow and ice         -> rock and bare ground
    80: 1,  # permanent water       -> water
    90: 2,  # herbaceous wetland    -> wetlands
    95: 2,  # mangroves             -> wetlands
    100: 5,  # moss and lichen      -> meadows and pastures
}

ONLINE_CLASSES = [
    LandCoverClass(1, "Вода", "#2E6E8E", 800),
    LandCoverClass(2, "Водно-болотные угодья", "#5E8A6B", 45),
    LandCoverClass(4, "Лес и древесный покров", "#2F6B4C", 6),
    LandCoverClass(5, "Луга и пастбища", "#9FAF74", 18),
    LandCoverClass(6, "Пашня", "#CDB77F", 40),
    LandCoverClass(7, "Кустарники и редколесье", "#A98B60", 28),
    LandCoverClass(8, "Застройка", "#A9695C", 250),
    LandCoverClass(9, "Дороги", "#E4D3A6", 400),
    LandCoverClass(10, "Скалы, пески, снег", "#8E9188", 90),
]

ROAD_WIDTH_M = {"motorway": 30.0, "trunk": 24.0, "primary": 18.0, "secondary": 14.0, "tertiary": 10.0}
PROTECT_CLASS_KIND = {
    "1a": "строгий природный резерват",
    "1b": "зона дикой природы",
    "2": "национальный парк",
    "3": "памятник природы",
    "4": "заказник",
}
PLACE_KIND = {"city": "город", "town": "город", "village": "село"}

ProgressFn = Callable[[str, float, str], None]


class OnlineError(RuntimeError):
    """Raised for failures the user can act on (no network, nothing found, too large)."""


def _noop(stage: str, fraction: float, message: str) -> None:  # pragma: no cover - trivial
    return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http(url: str, data: bytes | None = None, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


_GEOCODE_CACHE: dict[str, tuple[float, list["Place"]]] = {}


@dataclass
class Place:
    name: str
    display_name: str
    lon: float
    lat: float
    bbox: tuple[float, float, float, float]  # west, south, east, north
    category: str = ""
    osm_type: str = ""

    def as_dict(self) -> dict:
        w, s, e, n = self.bbox
        mid_lat = math.radians((s + n) / 2)
        return {
            "name": self.name,
            "display_name": self.display_name,
            "lon": self.lon,
            "lat": self.lat,
            "bbox": list(self.bbox),
            "width_km": round((e - w) * 111.32 * math.cos(mid_lat), 1),
            "height_km": round((n - s) * 110.57, 1),
            "category": self.category,
            "osm_type": self.osm_type,
        }


def geocode(query: str, limit: int = 6) -> list[Place]:
    """Place-name search through Nominatim (cached for an hour)."""
    query = query.strip()
    if len(query) < 2:
        raise OnlineError("Введите название территории (минимум 2 символа).")
    cached = _GEOCODE_CACHE.get(query.lower())
    if cached and time.time() - cached[0] < 3600:
        return cached[1]
    params = urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": limit, "accept-language": "ru,en", "addressdetails": 0}
    )
    try:
        payload = json.loads(_http(f"{NOMINATIM_URL}?{params}", timeout=30.0))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OnlineError(f"Сервис геокодирования недоступен: {exc}") from exc
    places = []
    for item in payload:
        try:
            south, north, west, east = (float(v) for v in item["boundingbox"])
            places.append(
                Place(
                    name=item.get("name") or item["display_name"].split(",")[0],
                    display_name=item["display_name"],
                    lon=float(item["lon"]),
                    lat=float(item["lat"]),
                    bbox=(west, south, east, north),
                    category=f"{item.get('category', '')}/{item.get('type', '')}".strip("/"),
                    osm_type=item.get("osm_type", ""),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    _GEOCODE_CACHE[query.lower()] = (time.time(), places)
    return places


# ---------------------------------------------------------------------------
# Extent planning
# ---------------------------------------------------------------------------
@dataclass
class TerritoryRequest:
    query: str | None = None
    bbox: tuple[float, float, float, float] | None = None  # west, south, east, north
    center: tuple[float, float] | None = None  # lon, lat
    size_km: float | None = None
    resolution_m: float | None = None
    core_source: str = "auto"  # "protected" | "habitat" | "auto"
    min_core_ha: float = 100.0
    name: str | None = None
    place_index: int = 0

    def fingerprint(self, extent: "Extent") -> str:
        payload = json.dumps(
            {
                "v": SCHEMA_VERSION,
                "bbox": [round(v, 4) for v in extent.bbox_lonlat],
                "res": extent.res,
                "core": self.core_source,
                "min_core_ha": self.min_core_ha,
            },
            sort_keys=True,
        )
        return "t" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


@dataclass
class Extent:
    crs: str
    left: float
    bottom: float
    right: float
    top: float
    res: float
    width: int
    height: int
    bbox_lonlat: tuple[float, float, float, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def transform(self) -> Affine:
        return from_origin(self.left, self.top, self.res, self.res)

    @property
    def area_km2(self) -> float:
        return self.width * self.height * self.res**2 / 1e6

    def as_dict(self) -> dict:
        return {
            "crs": self.crs,
            "width_km": round(self.width * self.res / 1000.0, 1),
            "height_km": round(self.height * self.res / 1000.0, 1),
            "area_km2": round(self.area_km2, 1),
            "resolution_m": self.res,
            "cells": self.width * self.height,
            "bbox": [round(v, 5) for v in self.bbox_lonlat],
            "warnings": list(self.warnings),
        }


def utm_crs(lon: float, lat: float) -> str:
    if abs(lat) > 84:
        raise OnlineError("Полярные территории (выше 84° ш.) не поддерживаются: проекция UTM неприменима.")
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _nice_resolution(value: float) -> float:
    for candidate in NICE_RESOLUTIONS:
        if candidate >= value:
            return candidate
    return NICE_RESOLUTIONS[-1]


def _km_per_degree(lat: float) -> tuple[float, float]:
    return 111.32 * math.cos(math.radians(lat)), 110.57


def plan_extent(request: TerritoryRequest, place: Place | None = None) -> Extent:
    """Turn a request into a projected, grid-aligned extent."""
    warnings: list[str] = []
    if request.bbox is not None:
        west, south, east, north = request.bbox
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise OnlineError("Некорректные границы: ожидается запад < восток и юг < север (градусы WGS84).")
    elif request.center is not None:
        lon, lat = request.center
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise OnlineError("Некорректные координаты центра.")
        side = request.size_km or 50.0
        kx, ky = _km_per_degree(lat)
        west, east = lon - side / 2 / kx, lon + side / 2 / kx
        south, north = lat - side / 2 / ky, lat + side / 2 / ky
    elif place is not None:
        west, south, east, north = place.bbox
    else:
        raise OnlineError("Укажите название территории, границы или центр и размер.")

    lon_c, lat_c = (west + east) / 2, (south + north) / 2
    kx, ky = _km_per_degree(lat_c)
    width_km, height_km = (east - west) * kx, (north - south) * ky

    if request.size_km and request.bbox is None and request.center is None:
        # A named place cropped to a centred square.
        width_km = height_km = float(request.size_km)
    if width_km < MIN_SIDE_KM or height_km < MIN_SIDE_KM:
        warnings.append(
            f"Территория меньше {MIN_SIDE_KM:.0f} км по стороне — расширена до {MIN_SIDE_KM:.0f} км, "
            "чтобы в кадр попали окружающие ООПТ и дороги."
        )
        width_km, height_km = max(width_km, MIN_SIDE_KM), max(height_km, MIN_SIDE_KM)
    if width_km * height_km > MAX_AREA_KM2:
        side = math.sqrt(MAX_AREA_KM2)
        warnings.append(
            f"Территория {width_km:.0f} × {height_km:.0f} км слишком велика — взят центральный квадрат {side:.0f} км."
        )
        width_km = height_km = side

    crs = utm_crs(lon_c, lat_c)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = to_utm.transform(lon_c, lat_c)
    xmin, xmax = cx - width_km * 500.0, cx + width_km * 500.0
    ymin, ymax = cy - height_km * 500.0, cy + height_km * 500.0

    area_m2 = (xmax - xmin) * (ymax - ymin)
    if request.resolution_m:
        res = float(request.resolution_m)
        if res < 10 or res > 2000:
            raise OnlineError("Разрешение должно быть от 10 до 2000 м.")
    else:
        res = _nice_resolution(math.sqrt(area_m2 / TARGET_CELLS))
    min_res = math.sqrt(area_m2 / MAX_CELLS)
    if res < min_res:
        bumped = _nice_resolution(min_res)
        warnings.append(f"Разрешение {res:g} м даёт более {MAX_CELLS:,} ячеек — использовано {bumped:g} м.".replace(",", " "))
        res = bumped

    left, top = round(xmin / res) * res, round(ymax / res) * res
    width = max(1, int(round((xmax - xmin) / res)))
    height = max(1, int(round((ymax - ymin) / res)))
    bottom, right = top - height * res, left + width * res

    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    corners = [to_ll.transform(x, y) for x, y in ((left, bottom), (right, bottom), (right, top), (left, top))]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]
    return Extent(
        crs=crs,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        res=res,
        width=width,
        height=height,
        bbox_lonlat=(min(lons), min(lats), max(lons), max(lats)),
        warnings=warnings,
    )


def resolve_request(request: TerritoryRequest) -> tuple[Extent, Place | None]:
    place = None
    if request.bbox is None and request.center is None:
        if not request.query:
            raise OnlineError("Укажите название территории, границы или центр и размер.")
        found = geocode(request.query)
        if not found:
            raise OnlineError(f"Территория «{request.query}» не найдена.")
        if request.place_index >= len(found):
            raise OnlineError("Выбранный результат поиска устарел — повторите поиск.")
        place = found[request.place_index]
    return plan_extent(request, place), place


# ---------------------------------------------------------------------------
# Rasters
# ---------------------------------------------------------------------------
def _grid_tiles(west: float, south: float, east: float, north: float, step: int) -> list[tuple[int, int]]:
    eps = 1e-9
    lats = range(int(math.floor(south / step)) * step, int(math.floor(north - eps)) // step * step + 1, step)
    lons = range(int(math.floor(west / step)) * step, int(math.floor(east - eps)) // step * step + 1, step)
    return [(la, lo) for la in lats for lo in lons]


def worldcover_tile(lat0: int, lon0: int) -> str:
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"


def copdem_tile(lat0: int, lon0: int) -> str:
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}_00_{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}_00"


def _warp_tiles(urls: list[str], extent: Extent, resampling: Resampling, dtype: str, fill: float) -> tuple[np.ndarray, int]:
    """Warp every reachable tile into the extent grid; returns (array, number of tiles read)."""
    dst = np.full((extent.height, extent.width), fill, dtype=dtype)
    read = 0
    with rasterio.Env(**GDAL_ENV):
        for url in urls:
            try:
                src = rasterio.open(url)
            except RasterioIOError:
                continue  # tile does not exist (open ocean) or host unreachable
            with src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst,
                    dst_transform=extent.transform,
                    dst_crs=extent.crs,
                    resampling=resampling,
                    dst_nodata=fill,
                    init_dest_nodata=False,
                )
            read += 1
    return dst, read


def fetch_landcover(extent: Extent) -> np.ndarray:
    """ESA WorldCover mapped to EcoCorridor classes (uint8)."""
    west, south, east, north = extent.bbox_lonlat
    urls = [WORLDCOVER_URL.format(tile=worldcover_tile(la, lo)) for la, lo in _grid_tiles(west, south, east, north, 3)]
    resampling = Resampling.mode if extent.res > 15 else Resampling.nearest
    raw, read = _warp_tiles(urls, extent, resampling, "uint8", 0)
    if read == 0:
        raise OnlineError(
            "Не удалось получить данные ESA WorldCover (ни один тайл не открылся). "
            "Проверьте подключение к интернету и доступ к esa-worldcover.s3.eu-central-1.amazonaws.com."
        )
    lut = np.zeros(256, dtype=np.uint8)
    for code, cls in WORLDCOVER_TO_CLASS.items():
        lut[code] = cls
    landcover = lut[raw]
    # Pixels no tile covers are open sea (WorldCover has no ocean tiles).
    landcover[raw == 0] = 1
    return landcover


def fetch_dem(extent: Extent) -> tuple[np.ndarray, str | None]:
    """Copernicus DEM (GLO-30, then GLO-90) warped to the grid; returns (dem, warning)."""
    west, south, east, north = extent.bbox_lonlat
    tiles = _grid_tiles(west, south, east, north, 1)
    resampling = Resampling.bilinear if extent.res <= 60 else Resampling.average
    for template, label_ in ((COPDEM30_URL, "GLO-30"), (COPDEM90_URL, "GLO-90")):
        urls = [template.format(tile=copdem_tile(la, lo)) for la, lo in tiles]
        dem, read = _warp_tiles(urls, extent, resampling, "float32", np.nan)
        if read:
            valid = np.isfinite(dem)
            if not valid.all():
                if not valid.any():
                    continue
                # Fill gaps (sea, missing tiles) with the nearest valid elevation: no artificial cliffs.
                idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
                dem = dem[tuple(idx)]
            note = None if label_ == "GLO-30" else "Использована модель Copernicus GLO-90 (90 м): GLO-30 недоступна для этой территории."
            return dem.astype("float32"), note
    return (
        np.zeros((extent.height, extent.width), dtype="float32"),
        "Модель рельефа недоступна — уклон принят нулевым, влияние рельефа на сопротивление не учитывается.",
    )


# ---------------------------------------------------------------------------
# OpenStreetMap
# ---------------------------------------------------------------------------
def _overpass(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                payload = json.loads(_http(url, data=data, timeout=200.0))
                remark = str(payload.get("remark", ""))
                if "runtime error" in remark or "out of memory" in remark.lower():
                    raise OnlineError(f"Overpass: {remark}")
                return payload
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OnlineError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
    raise OnlineError(f"Overpass API недоступен или перегружен: {last}")


def overpass_query(bbox: tuple[float, float, float, float], road_classes: tuple[str, ...]) -> str:
    west, south, east, north = bbox
    b = f"({south:.5f},{west:.5f},{north:.5f},{east:.5f})"
    classes = "|".join(f"{c}|{c}_link" for c in road_classes)
    return (
        "[out:json][timeout:180][maxsize:1073741824];\n(\n"
        f'  way["highway"~"^({classes})$"]{b};\n'
        f'  way["waterway"~"^(river|canal)$"]{b};\n'
        f'  node["place"~"^(city|town|village)$"]{b};\n'
        f'  way["boundary"="protected_area"]{b};\n  relation["boundary"="protected_area"]{b};\n'
        f'  way["boundary"="national_park"]{b};\n  relation["boundary"="national_park"]{b};\n'
        f'  way["leisure"="nature_reserve"]{b};\n  relation["leisure"="nature_reserve"]{b};\n'
        ");\nout geom;"
    )


_AREA_KEYS = {"boundary", "leisure", "natural", "landuse"}


def _coords(geometry: list[dict]) -> list[tuple[float, float]]:
    return [(float(p["lon"]), float(p["lat"])) for p in geometry or [] if "lon" in p and "lat" in p]


def _relation_polygon(element: dict):
    outers, inners = [], []
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        pts = _coords(member.get("geometry", []))
        if len(pts) < 2:
            continue
        (inners if member.get("role") == "inner" else outers).append(LineString(pts))
    if not outers:
        return None
    shells = list(polygonize(linemerge(outers)))
    if not shells:
        return None
    polygon = unary_union(shells)
    if inners:
        holes = list(polygonize(linemerge(inners)))
        if holes:
            polygon = polygon.difference(unary_union(holes))
    return polygon if not polygon.is_empty else None


def osm_features(elements: list[dict]) -> list[tuple[dict, object]]:
    """Overpass ``out geom`` elements -> (tags, shapely geometry) in WGS84."""
    result: list[tuple[dict, object]] = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        tags = {**tags, "_osm": f"{el.get('type', '?')}/{el.get('id', '?')}"}
        kind = el.get("type")
        geom = None
        if kind == "node" and "lon" in el:
            geom = Point(float(el["lon"]), float(el["lat"]))
        elif kind == "way":
            pts = _coords(el.get("geometry", []))
            if len(pts) >= 2:
                closed = pts[0] == pts[-1] and len(pts) >= 4
                if closed and any(k in tags for k in _AREA_KEYS):
                    geom = Polygon(pts)
                else:
                    geom = LineString(pts)
        elif kind == "relation":
            geom = _relation_polygon(el)
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        result.append((tags, geom))
    return result


def _protect_class(tags: dict) -> str:
    raw = str(tags.get("protect_class", "")).split(";")[0].strip().lower()
    return raw


def _is_core_candidate(tags: dict) -> bool:
    if tags.get("leisure") == "nature_reserve" or tags.get("boundary") == "national_park":
        return _protect_class(tags) not in {"5", "6", "7", "8", "97", "98", "99"}
    return tags.get("boundary") == "protected_area" and _protect_class(tags) in PROTECT_CLASS_KIND


def _core_kind(tags: dict) -> str:
    if tags.get("protection_title"):
        return str(tags["protection_title"])
    if tags.get("boundary") == "national_park":
        return "национальный парк"
    return PROTECT_CLASS_KIND.get(_protect_class(tags), "природный резерват" if tags.get("leisure") == "nature_reserve" else "ООПТ")


def _to_utm(geom, transformer: Transformer):
    from shapely.ops import transform

    return transform(transformer.transform, geom)


def _dedupe_cores(rows: list[dict], min_ha: float) -> list[dict]:
    """Drop protected areas that mostly repeat larger ones; trim partial overlaps."""
    rows = sorted(rows, key=lambda r: r["geometry"].area, reverse=True)
    kept: list[dict] = []
    union = None
    for row in rows:
        geom = row["geometry"]
        if union is not None:
            geom = geom.difference(union)
            if geom.is_empty or geom.area < 0.3 * row["geometry"].area:
                continue
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        if geom.geom_type != "Polygon" or geom.area / 10_000.0 < min_ha:
            continue
        kept.append({**row, "geometry": geom})
        union = geom if union is None else union.union(geom)
    return kept


def build_vectors(payload: dict, extent: Extent, min_core_ha: float) -> dict[str, gpd.GeoDataFrame]:
    to_utm = Transformer.from_crs("EPSG:4326", extent.crs, always_xy=True)
    frame = box(extent.left, extent.bottom, extent.right, extent.top)
    roads, rivers, settlements, cores = [], [], [], []
    for tags, geom in osm_features(payload.get("elements", [])):
        highway = tags.get("highway")
        if highway and geom.geom_type == "LineString":
            if str(tags.get("tunnel", "no")) not in ("no", ""):
                continue  # animals pass over tunnels
            base = highway.replace("_link", "")
            roads.append(
                {
                    "name": tags.get("name", ""),
                    "kind": base,
                    "width_m": ROAD_WIDTH_M.get(base, 10.0),
                    "bridge": str(tags.get("bridge", "no")) not in ("no", ""),
                    "geometry": _to_utm(geom, to_utm),
                }
            )
        elif tags.get("waterway") in ("river", "canal") and geom.geom_type == "LineString":
            rivers.append(
                {
                    "name": tags.get("name", ""),
                    "order": 2 if tags["waterway"] == "river" else 1,
                    "width_m": 40.0 if tags["waterway"] == "river" else 12.0,
                    "geometry": _to_utm(geom, to_utm),
                }
            )
        elif tags.get("place") in PLACE_KIND and geom.geom_type == "Point":
            try:
                population = int(str(tags.get("population", "0")).replace(" ", "").split(";")[0])
            except ValueError:
                population = 0
            settlements.append(
                {
                    "name": tags.get("name:ru") or tags.get("name", ""),
                    "population": population,
                    "kind": PLACE_KIND[tags["place"]],
                    "geometry": _to_utm(geom, to_utm),
                }
            )
        elif _is_core_candidate(tags) and geom.geom_type in ("Polygon", "MultiPolygon"):
            utm = _to_utm(geom, to_utm).buffer(0).intersection(frame)
            if utm.is_empty or utm.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            cores.append(
                {
                    "name": tags.get("name:ru") or tags.get("name") or f"ООПТ {tags['_osm']}",
                    "kind": _core_kind(tags),
                    "geometry": utm,
                }
            )

    def frame_gdf(rows: list[dict], clip: bool = True, simplify: float = 0.0) -> gpd.GeoDataFrame:
        if not rows:
            return gpd.GeoDataFrame(geometry=[], crs=extent.crs)
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=extent.crs)
        if clip:
            gdf["geometry"] = gdf.geometry.intersection(frame)
            gdf = gdf[~gdf.geometry.is_empty]
        if simplify:
            gdf["geometry"] = gdf.geometry.simplify(simplify)
        return gdf.reset_index(drop=True)

    settlements = sorted(settlements, key=lambda r: r["population"], reverse=True)[:400]
    core_rows = _dedupe_cores(cores, min_core_ha)
    for row in core_rows:
        row["area_ha"] = round(row["geometry"].area / 10_000.0, 1)
    settlement_gdf = frame_gdf(settlements)
    return {
        "roads": frame_gdf(roads, simplify=extent.res * 0.4),
        "rivers": frame_gdf(rivers, simplify=extent.res * 0.4),
        "settlements": settlement_gdf,
        "cores": frame_gdf(core_rows, clip=False),
    }


def burn_roads(landcover: np.ndarray, roads: gpd.GeoDataFrame, extent: Extent) -> np.ndarray:
    """Burn OSM roads into the land cover (class 9); bridges also cross water."""
    if roads.empty:
        return landcover
    out = landcover.copy()
    for bridge in (False, True):
        subset = roads[roads["bridge"] == bridge] if "bridge" in roads else (roads if not bridge else roads.iloc[0:0])
        if subset.empty:
            continue
        shapes_ = [
            (geom.buffer(max(width, extent.res) / 2.0), 1)
            for geom, width in zip(subset.geometry, subset["width_m"])
            if geom is not None and not geom.is_empty
        ]
        mask = rasterize(shapes_, out_shape=out.shape, transform=extent.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
        out[mask & (bridge | (out != 1))] = 9
    return out


def habitat_cores(landcover: np.ndarray, extent: Extent, min_ha: float, limit: int = 10) -> gpd.GeoDataFrame:
    """Fall-back cores: the largest forest massifs (classes forest, wetland) of the territory."""
    mask = np.isin(landcover, (4, 2))
    labels, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return gpd.GeoDataFrame(geometry=[], crs=extent.crs)
    cell_ha = extent.res**2 / 10_000.0
    areas = np.bincount(labels.ravel(), minlength=n + 1) * cell_ha
    areas[0] = 0
    ranked = [i for i in np.argsort(areas)[::-1] if areas[i] >= min_ha][:limit]
    rows = []
    for rank, patch in enumerate(ranked, start=1):
        parts = [
            shape(geom)
            for geom, value in shapes((labels == patch).astype("uint8"), mask=labels == patch, transform=extent.transform)
            if value == 1
        ]
        if not parts:
            continue
        geom = unary_union(parts).simplify(extent.res * 0.5)
        rows.append(
            {
                "name": f"Лесной массив {rank}",
                "kind": "лесной массив (по ESA WorldCover)",
                "area_ha": round(float(areas[patch]), 1),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=extent.crs) if rows else gpd.GeoDataFrame(geometry=[], crs=extent.crs)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def cache_path(project_id: str) -> Path:
    return CACHE_DIR / project_id


def list_cached() -> list[dict]:
    items = []
    if CACHE_DIR.exists():
        for meta_path in sorted(CACHE_DIR.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items.append(
                {
                    "id": meta.get("id", meta_path.parent.name),
                    "name": meta.get("name", meta_path.parent.name),
                    "resolution_m": meta.get("resolution_m"),
                    "area_km2": meta.get("area_km2"),
                    "created": meta.get("created"),
                }
            )
    return items


def load_territory(request: TerritoryRequest, progress: ProgressFn = _noop, force: bool = False) -> Project:
    """Download (or reuse from cache) everything for a territory and return a Project."""
    progress("plan", 0.02, "Определяю границы и разрешение…")
    extent, place = resolve_request(request)
    project_id = request.fingerprint(extent)
    target = cache_path(project_id)
    if not force and (target / "meta.json").exists():
        progress("cache", 0.95, "Найдено в кэше — загружаю с диска…")
        return Project.from_directory(target, project_id=project_id)

    progress("landcover", 0.08, "Загружаю землепокров ESA WorldCover (10 м)…")
    landcover = fetch_landcover(extent)

    progress("dem", 0.40, "Загружаю рельеф Copernicus DEM…")
    dem, dem_note = fetch_dem(extent)

    progress("osm", 0.62, "Запрашиваю OpenStreetMap: дороги, реки, ООПТ, населённые пункты…")
    area = extent.area_km2
    road_classes = ("motorway", "trunk", "primary", "secondary", "tertiary") if area <= 3000 else ("motorway", "trunk", "primary", "secondary")
    payload = _overpass(overpass_query(extent.bbox_lonlat, road_classes))

    progress("assemble", 0.86, "Собираю проект: растр сопротивления, ядра, слои…")
    vectors = build_vectors(payload, extent, request.min_core_ha)
    landcover = burn_roads(landcover, vectors["roads"], extent)

    warnings = list(extent.warnings)
    if dem_note:
        warnings.append(dem_note)
    protected = vectors["cores"]
    source = request.core_source
    if source not in ("auto", "protected", "habitat"):
        raise OnlineError("core_source должен быть auto, protected или habitat.")
    cores_origin = "OpenStreetMap: охраняемые природные территории"
    if source == "habitat" or (source == "auto" and len(protected) < 2):
        if source == "auto":
            warnings.append(
                f"В OpenStreetMap найдено ООПТ: {len(protected)} (нужно минимум 2 от {request.min_core_ha:g} га) — "
                "ядрами выбраны крупнейшие лесные массивы."
            )
        protected = habitat_cores(landcover, extent, max(request.min_core_ha, 50.0))
        cores_origin = "крупнейшие лесные массивы (ESA WorldCover)"
    if len(protected) < 2:
        warnings.append("Найдено меньше двух ядер — коридорная сеть не будет построена. Увеличьте территорию или уменьшите порог площади ядра.")

    name = request.name or (place.name if place else None) or f"{abs(extent.bbox_lonlat[1]):.2f}°, {extent.bbox_lonlat[0]:.2f}°"
    region = gpd.GeoDataFrame(geometry=[box(extent.left, extent.bottom, extent.right, extent.top)], crs=extent.crs)
    meta = {
        "online": True,
        "schema": SCHEMA_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": request.query,
        "bbox_lonlat": [round(v, 5) for v in extent.bbox_lonlat],
        "area_km2": round(area, 1),
        "core_source": cores_origin,
        "road_classes": list(road_classes),
        "warnings": warnings,
        "sources": [
            "© участники OpenStreetMap (ODbL): дороги, реки, ООПТ, населённые пункты",
            "ESA WorldCover 10 m 2021 v200 © ESA WorldCover project / Contains modified Copernicus Sentinel data (2021) "
            "processed by ESA WorldCover consortium (CC BY 4.0); общая точность 76,7 %",
            "Copernicus DEM GLO-30/90 © DLR e.V., Airbus Defence and Space GmbH, ESA, Copernicus (модель поверхности)",
        ],
    }
    project = Project(
        id=project_id,
        name=f"{name} · {extent.width * extent.res / 1000:.0f} × {extent.height * extent.res / 1000:.0f} км",
        description=f"Данные загружены онлайн: {place.display_name if place else 'заданные границы'}.",
        crs=extent.crs,
        transform=extent.transform,
        width=extent.width,
        height=extent.height,
        res=extent.res,
        dem=dem.astype("float64"),
        landcover=landcover,
        classes=ONLINE_CLASSES,
        cores=protected,
        roads=vectors["roads"],
        rivers=vectors["rivers"],
        settlements=vectors["settlements"],
        region=region,
        synthetic=False,
        meta=meta,
    )
    progress("save", 0.96, "Сохраняю в кэш…")
    try:
        project.to_directory(target)
    except OSError as exc:  # read-only deployment: the in-memory project is still usable
        project.meta["warnings"].append(f"Не удалось сохранить кэш: {exc}")
    return project
