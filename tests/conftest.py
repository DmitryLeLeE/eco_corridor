"""Shared fixtures: tiny synthetic projects built from numpy arrays."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecocorridor.dataset import LandCoverClass, Project  # noqa: E402

CRS = "EPSG:32637"

CLASSES = [
    LandCoverClass(1, "Вода", "#2E6E8E", 800),
    LandCoverClass(2, "Заболоченные земли", "#5E8A6B", 45),
    LandCoverClass(3, "Хвойный лес", "#1B4A38", 3),
    LandCoverClass(4, "Лес", "#2F6B4C", 6),
    LandCoverClass(5, "Луга", "#9FAF74", 18),
    LandCoverClass(6, "Пашня", "#CDB77F", 40),
    LandCoverClass(7, "Редколесье", "#A98B60", 28),
    LandCoverClass(8, "Застройка", "#A9695C", 250),
    LandCoverClass(9, "Дороги", "#E4D3A6", 400),
    LandCoverClass(10, "Скалы", "#8E9188", 90),
]


def empty_gdf(crs: str = CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def make_project(landcover: np.ndarray, res: float = 100.0, dem: np.ndarray | None = None, **vectors) -> Project:
    h, w = landcover.shape
    return Project(
        id="toy",
        name="toy",
        description="",
        crs=CRS,
        transform=from_origin(500_000.0, 5_000_000.0 + h * res, res, res),
        width=w,
        height=h,
        res=res,
        dem=np.zeros((h, w)) if dem is None else dem,
        landcover=landcover.astype(np.uint8),
        classes=CLASSES,
        cores=vectors.get("cores", empty_gdf()),
        roads=vectors.get("roads", empty_gdf()),
        rivers=vectors.get("rivers", empty_gdf()),
        settlements=vectors.get("settlements", empty_gdf()),
        region=vectors.get("region", empty_gdf()),
    )


@pytest.fixture
def toy():
    return make_project


# ---------------------------------------------------------------------------
# Synthetic "world": WorldCover / Copernicus DEM tiles on disk + a fake Overpass
# ---------------------------------------------------------------------------
LON0, LAT0 = 45.0, 51.5  # centre of the synthetic territory (straddles two WorldCover and two DEM tiles)


def _wc_class(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    out = np.full(lon.shape, 10, dtype=np.uint8)  # tree cover
    out[lat < LAT0 - 0.10] = 40  # cropland in the south
    out[lat > LAT0 + 0.12] = 30  # grassland in the north
    out[(abs(lon - (LON0 + 0.18)) < 0.015) & (abs(lat - (LAT0 - 0.12)) < 0.015)] = 50  # a town
    out[abs(lon - LON0) < 0.01] = 80  # a river band running north-south
    return out


def _write_tile(path: Path, west: float, north: float, size_deg: float, px: float, data: np.ndarray, dtype: str, nodata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import rasterio

    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=dtype,
        crs="EPSG:4326", transform=from_origin(west, north, px, px), nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)


def write_synthetic_world(root: Path) -> dict[str, str]:
    """Write WorldCover (3 deg) and DEM (1 deg) tiles around the test territory; return URL templates."""
    px_wc, px_dem = 0.003, 0.001
    for lat_t in (51,):
        for lon_t in (42, 45):
            n = int(3 / px_wc)
            lon = lon_t + (np.arange(n) + 0.5) * px_wc
            lat = lat_t + 3 - (np.arange(n) + 0.5) * px_wc
            grid_lon, grid_lat = np.meshgrid(lon, lat)
            name = f"N{lat_t:02d}E{lon_t:03d}"
            _write_tile(root / "wc" / f"wc_{name}.tif", lon_t, lat_t + 3, 3, px_wc, _wc_class(grid_lon, grid_lat), "uint8", 0)
    for lon_t in (44, 45):
        n = int(1 / px_dem)
        lon = lon_t + (np.arange(n) + 0.5) * px_dem
        lat = 51 + 1 - (np.arange(n) + 0.5) * px_dem
        grid_lon, grid_lat = np.meshgrid(lon, lat)
        z = 100.0 + 3000.0 * (grid_lat - 51.0)  # a uniform northward ramp: ~1.55 deg slope
        for template in ("dem30", "dem90"):
            _write_tile(root / template / f"{template}_N51_00_E{lon_t:03d}_00.tif", lon_t, 52, 1, px_dem, z, "float32", None)
    return {
        "wc": str(root / "wc" / "wc_{tile}.tif"),
        "dem30": str(root / "dem30" / "dem30_{tile}.tif"),
        "dem90": str(root / "dem90" / "dem90_{tile}.tif"),
    }


def _ring(w, s, e, n):
    return [{"lon": w, "lat": s}, {"lon": e, "lat": s}, {"lon": e, "lat": n}, {"lon": w, "lat": n}, {"lon": w, "lat": s}]


def _line(points):
    return [{"lon": x, "lat": y} for x, y in points]


def fake_overpass_payload(with_protected: bool = True) -> dict:
    els: list[dict] = [
        {"type": "way", "id": 1, "tags": {"highway": "primary", "name": "Р-1"}, "geometry": _line([(LON0 - 0.25, LAT0 - 0.1), (LON0 + 0.25, LAT0 + 0.1)])},
        {"type": "way", "id": 2, "tags": {"highway": "secondary_link"}, "geometry": _line([(LON0 + 0.1, LAT0 - 0.3), (LON0 + 0.1, LAT0 + 0.3)])},
        {"type": "way", "id": 3, "tags": {"highway": "tertiary", "bridge": "yes"}, "geometry": _line([(LON0 - 0.1, LAT0 + 0.05), (LON0 + 0.1, LAT0 + 0.05)])},
        {"type": "way", "id": 4, "tags": {"highway": "primary", "tunnel": "yes"}, "geometry": _line([(LON0 - 0.1, LAT0 - 0.05), (LON0 + 0.1, LAT0 - 0.05)])},
        {"type": "way", "id": 16, "tags": {"highway": "trunk"}, "geometry": _line([(LON0 - 0.02, LAT0 - 0.3), (LON0 - 0.02, LAT0 + 0.3)])},
        {"type": "way", "id": 5, "tags": {"waterway": "river", "name": "Тестовая"}, "geometry": _line([(LON0, LAT0 - 0.3), (LON0, LAT0 + 0.3)])},
        {"type": "node", "id": 6, "lat": LAT0 - 0.12, "lon": LON0 + 0.18, "tags": {"place": "town", "name": "Тестоград", "population": "15000"}},
        {"type": "node", "id": 7, "lat": LAT0 + 0.1, "lon": LON0 - 0.2, "tags": {"place": "village", "name": "Малое"}},
        {"type": "node", "id": 8, "lat": LAT0, "lon": LON0, "tags": {"place": "hamlet", "name": "Хутор"}},  # not requested
    ]
    if with_protected:
        els += [
            {
                "type": "relation", "id": 10,
                "tags": {"type": "boundary", "boundary": "protected_area", "protect_class": "1a", "name": "Заповедник «Тест»"},
                "members": [
                    {"type": "way", "role": "outer", "geometry": _ring(LON0 - 0.15, LAT0 - 0.08, LON0 - 0.05, LAT0 + 0.02)},
                    {"type": "way", "role": "inner", "geometry": _ring(LON0 - 0.11, LAT0 - 0.05, LON0 - 0.09, LAT0 - 0.03)},
                ],
            },
            {"type": "way", "id": 11, "tags": {"leisure": "nature_reserve", "name": "Заказник «Север»"}, "geometry": _ring(LON0 + 0.05, LAT0 - 0.02, LON0 + 0.15, LAT0 + 0.08)},
            # a duplicate that lies inside the first reserve -> dropped
            {"type": "way", "id": 12, "tags": {"boundary": "protected_area", "protect_class": "4", "name": "Дубль"}, "geometry": _ring(LON0 - 0.14, LAT0 - 0.07, LON0 - 0.06, LAT0 + 0.01)},
            # about 3 ha -> below the minimum core area
            {"type": "way", "id": 13, "tags": {"boundary": "protected_area", "protect_class": "2", "name": "Крошка"}, "geometry": _ring(LON0 + 0.2, LAT0 + 0.2, LON0 + 0.202, LAT0 + 0.202)},
            # IUCN class 5 (protected landscape) is not a core
            {"type": "way", "id": 14, "tags": {"boundary": "protected_area", "protect_class": "5", "name": "Ландшафт"}, "geometry": _ring(LON0 - 0.2, LAT0 + 0.1, LON0 - 0.15, LAT0 + 0.18)},
            # outside the territory
            {"type": "way", "id": 15, "tags": {"boundary": "protected_area", "protect_class": "2", "name": "Далеко"}, "geometry": _ring(LON0 + 0.5, LAT0 + 0.5, LON0 + 0.6, LAT0 + 0.6)},
        ]
    return {"version": 0.6, "elements": els}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Point every data source at local synthetic files; returns the centre and a call counter."""
    from ecocorridor import api, dataset, online

    urls = write_synthetic_world(tmp_path / "tiles")
    monkeypatch.setattr(online, "WORLDCOVER_URL", urls["wc"])
    monkeypatch.setattr(online, "COPDEM30_URL", urls["dem30"])
    monkeypatch.setattr(online, "COPDEM90_URL", urls["dem90"])
    cache = tmp_path / "cache"
    monkeypatch.setattr(online, "CACHE_DIR", cache)
    monkeypatch.setattr(api, "CACHE_DIR", cache)
    monkeypatch.setattr(dataset, "CACHE_DIR", cache)
    state = {"overpass_calls": 0, "protected": True}

    def fake(query: str) -> dict:
        state["overpass_calls"] += 1
        state["last_query"] = query
        return fake_overpass_payload(state["protected"])

    monkeypatch.setattr(online, "_overpass", fake)
    state["center"] = (LON0, LAT0)
    return state
