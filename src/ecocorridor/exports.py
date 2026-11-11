"""GIS exports of an analysis: GeoTIFF rasters and a GeoPackage with the vector results."""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from .analysis import AnalysisResult

NODATA = -9999.0

RASTER_LAYERS = {
    "resistance": ("Поверхность сопротивления (относительная стоимость движения)", "float32"),
    "cost": ("Накопленная стоимость пути от ближайшего ядра", "float32"),
    "flow": ("Интенсивность использования коридоров (число проходящих путей)", "float32"),
    "current": ("Накопленный ток (теория цепей) между ядрами", "float32"),
    "landcover": ("Землепокров сценария (классы EcoCorridor)", "uint8"),
}


def raster_array(result: AnalysisResult, layer: str) -> np.ndarray:
    if layer == "resistance":
        return np.asarray(result.resistance, dtype="float32")
    if layer == "cost":
        return np.asarray(result.cost, dtype="float32")
    if layer == "flow":
        return np.asarray(result.network.flow, dtype="float32")
    if layer == "current":
        from .analysis import circuit_flow

        return np.asarray(circuit_flow(result)[0].current, dtype="float32")
    if layer == "landcover":
        return np.asarray(result.landcover, dtype="uint8")
    raise KeyError(layer)


def raster_bytes(result: AnalysisResult, layer: str) -> bytes:
    """A single-band GeoTIFF in the project CRS (deflate, nodata = -9999 for floats, 0 for classes)."""
    if layer not in RASTER_LAYERS:
        raise KeyError(layer)
    description, dtype = RASTER_LAYERS[layer]
    data = raster_array(result, layer)
    project = result.project
    profile = {
        "driver": "GTiff",
        "height": project.height,
        "width": project.width,
        "count": 1,
        "dtype": dtype,
        "crs": project.crs,
        "transform": project.transform,
        "compress": "deflate",
    }
    if dtype == "uint8":
        profile["nodata"] = 0
    else:
        profile["nodata"] = NODATA
        data = np.where(np.isfinite(data), data, NODATA).astype("float32")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{layer}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, description)
            dst.update_tags(
                SOFTWARE="EcoCorridor",
                SCENARIO=result.scenario.name,
                PROJECT=project.name,
            )
        return path.read_bytes()


def _plain(gdf: gpd.GeoDataFrame | None, crs: str) -> gpd.GeoDataFrame:
    """GeoPackage-safe copy: scalar attributes only, CRS set."""
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    out = gdf.copy()
    if out.crs is None:
        out = out.set_crs(crs)
    for column in list(out.columns):
        if column == out.geometry.name:
            continue
        if out[column].map(lambda v: isinstance(v, (list, dict, tuple, set))).any():
            out[column] = out[column].map(lambda v: str(v) if isinstance(v, (list, dict, tuple, set)) else v)
    return out


def geopackage_bytes(result: AnalysisResult) -> bytes:
    """Cores, corridor lines and polygons, pinch points, roads, rivers and settlements in one .gpkg."""
    project = result.project
    layers = {
        "cores": result.cores,
        "corridors": result.network.lines,
        "corridor_zones": result.network.polygons,
        "pinchpoints": result.network.pinchpoints,
        "roads": project.roads,
        "rivers": project.rivers,
        "settlements": project.settlements,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ecocorridor.gpkg"
        written = 0
        for name, gdf in layers.items():
            frame = _plain(gdf, project.crs)
            if len(frame) == 0:
                continue
            frame.to_file(path, layer=name, driver="GPKG")
            written += 1
        if written == 0:
            gpd.GeoDataFrame({"note": ["no results"]}, geometry=[None], crs=project.crs).to_file(path, layer="empty", driver="GPKG")
        return path.read_bytes()
