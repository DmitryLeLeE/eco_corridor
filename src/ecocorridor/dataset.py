"""Project loading: rasters, vectors and the land-cover class table."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "data" / "demo"
CACHE_DIR = Path(os.environ.get("ECOCORRIDOR_CACHE", REPO_ROOT / "data" / "cache"))


@dataclass(frozen=True)
class LandCoverClass:
    """One land-cover class and its baseline resistance (relative cost)."""

    code: int
    name: str
    color: str
    resistance: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        value = self.color.lstrip("#")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class Project:
    """A modelling project: one raster grid plus optional vector context."""

    id: str
    name: str
    description: str
    crs: str
    transform: Affine
    width: int
    height: int
    res: float
    dem: np.ndarray
    landcover: np.ndarray
    classes: list[LandCoverClass]
    cores: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    rivers: gpd.GeoDataFrame
    settlements: gpd.GeoDataFrame
    region: gpd.GeoDataFrame
    synthetic: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # -- geometry helpers ---------------------------------------------------
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        return left, top - self.height * self.res, left + self.width * self.res, top

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        x, y = self.transform * (col + 0.5, row + 0.5)
        return float(x), float(y)

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return float(lon), float(lat)

    def extent_lonlat(self) -> list[list[float]]:
        """Corners (SW, SE, NE, NW) in WGS84, used by the client for the graticule."""
        left, bottom, right, top = self.bounds
        return [
            list(self.to_lonlat(left, bottom)),
            list(self.to_lonlat(right, bottom)),
            list(self.to_lonlat(right, top)),
            list(self.to_lonlat(left, top)),
        ]

    # -- masks --------------------------------------------------------------
    def rasterize(self, gdf: gpd.GeoDataFrame, all_touched: bool = False) -> np.ndarray:
        shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
        out = rasterize(
            shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
        )
        return out.astype(bool)

    def core_mask(self) -> np.ndarray:
        return self.rasterize(self.cores)

    def habitat_mask(self, natural_codes: list[int] | None = None) -> np.ndarray:
        codes = natural_codes or [2, 3, 4, 5, 7, 10]
        return np.isin(self.landcover, codes)

    def class_resistance(self) -> dict[int, float]:
        return {c.code: c.resistance for c in self.classes}

    # -- persistence --------------------------------------------------------
    def to_directory(self, path: str | Path) -> Path:
        """Write the project in the same layout ``from_directory`` reads."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": 1,
            "crs": self.crs,
            "transform": self.transform,
            "compress": "deflate",
        }
        with rasterio.open(path / "landcover.tif", "w", dtype="uint8", nodata=0, **profile) as dst:
            dst.write(self.landcover.astype("uint8"), 1)
        with rasterio.open(path / "dem.tif", "w", dtype="float32", nodata=-9999.0, **profile) as dst:
            dst.write(np.where(np.isfinite(self.dem), self.dem, -9999.0).astype("float32"), 1)
        (path / "landcover_classes.json").write_text(
            json.dumps(
                {
                    "field": "class",
                    "classes": [
                        {"code": c.code, "name": c.name, "color": c.color, "resistance": c.resistance}
                        for c in self.classes
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        files = {"dem": "dem.tif", "landcover": "landcover.tif", "classes": "landcover_classes.json"}
        for key, gdf in (
            ("protected_areas", self.cores),
            ("roads", self.roads),
            ("rivers", self.rivers),
            ("settlements", self.settlements),
            ("region", self.region),
        ):
            if gdf is None or gdf.empty:
                continue
            fname = f"{key}.geojson"
            out = gdf.copy()
            # GeoJSON cannot store list / dict columns.
            for column in out.columns:
                if column != "geometry" and out[column].map(lambda v: isinstance(v, (list, dict, tuple))).any():
                    out[column] = out[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict, tuple)) else v)
            out.to_file(path / fname, driver="GeoJSON")
            files[key] = fname
        meta = dict(self.meta)
        meta.update(
            {
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "crs": self.crs,
                "resolution_m": self.res,
                "width": self.width,
                "height": self.height,
                "bounds": list(self.bounds),
                "synthetic": self.synthetic,
                "files": files,
            }
        )
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_directory(cls, path: str | Path, project_id: str | None = None) -> "Project":
        path = Path(path)
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        files = meta.get("files", {})

        with rasterio.open(path / files["landcover"]) as src:
            landcover = src.read(1)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else meta["crs"]
            height, width = src.height, src.width
            res = float(src.res[0])
            nodata = src.nodata

        if nodata is not None:
            landcover[landcover == nodata] = 0

        with rasterio.open(path / files["dem"]) as src:
            dem = src.read(1).astype(np.float64)
            dem_nodata = src.nodata
        if dem_nodata is not None:
            dem[dem == dem_nodata] = np.nan

        classes = [
            LandCoverClass(code=int(c["code"]), name=c["name"], color=c["color"], resistance=float(c["resistance"]))
            for c in json.loads((path / files["classes"]).read_text(encoding="utf-8"))["classes"]
        ]

        def read_vector(key: str) -> gpd.GeoDataFrame:
            fname = files.get(key)
            if not fname or not (path / fname).exists():
                return gpd.GeoDataFrame(geometry=[], crs=crs)
            gdf = gpd.read_file(path / fname)
            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            return gdf.to_crs(crs)

        return cls(
            id=project_id or meta.get("id", "demo"),
            name=meta.get("name", "Проект"),
            description=meta.get("description", ""),
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            res=res,
            dem=dem,
            landcover=landcover,
            classes=classes,
            cores=read_vector("protected_areas"),
            roads=read_vector("roads"),
            rivers=read_vector("rivers"),
            settlements=read_vector("settlements"),
            region=read_vector("region"),
            synthetic=bool(meta.get("synthetic", False)),
            meta=meta,
        )


def load_demo(project_id: str = "demo") -> Project:
    return Project.from_directory(DEMO_DIR, project_id=project_id)
