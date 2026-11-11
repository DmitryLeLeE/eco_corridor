"""Urbanisation / restoration scenarios.

A scenario is a small, explicit transformation of the resistance surface:

* built-up pressure grows by ``urban_growth_m`` metres around existing
  settlements and the new cells become built-up (high resistance);
* a new road (polyline in map coordinates) is cut into the surface;
* riparian restoration lowers resistance within ``restoration_width_m`` of
  streams, modelling the recovery of floodplain corridors.

The transformations are applied to the land-cover raster as well, so that the
habitat mask used by the fragmentation metrics changes consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely.geometry import LineString

from .dataset import Project
from .resistance import ResistanceSurface


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    urban_growth_m: float = 0.0
    urban_resistance: float = 250.0
    restoration: float = 0.0
    restoration_width_m: float = 400.0
    new_road: list[list[float]] = field(default_factory=list)
    new_road_width_m: float = 60.0
    accent: str = "#58E0A8"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "urban_growth_m": self.urban_growth_m,
            "restoration": self.restoration,
            "restoration_width_m": self.restoration_width_m,
            "new_road": self.new_road,
            "accent": self.accent,
        }


def default_scenarios(project: Project) -> list[Scenario]:
    """Four illustrative scenarios expressed in relative map coordinates."""
    left, bottom, right, top = project.bounds
    width, height = right - left, top - bottom

    def xy(fx: float, fy: float) -> list[float]:
        return [left + fx * width, bottom + fy * height]

    return [
        Scenario(
            id="base",
            name="Базовый (2025)",
            description="Текущая поверхность сопротивления без изменений.",
            accent="#58E0A8",
        ),
        Scenario(
            id="urban",
            name="Урбанизация +600 м",
            description="Застройка разрастается на 600 м вокруг существующих населённых пунктов.",
            urban_growth_m=600.0,
            accent="#E4A05A",
        ),
        Scenario(
            id="road",
            name="Новая автодорога",
            description="Трасса 60 м пересекает лесной массив между двумя ООПТ.",
            new_road=[xy(0.12, 0.30), xy(0.22, 0.44), xy(0.30, 0.62), xy(0.36, 0.78)],
            new_road_width_m=60.0,
            accent="#D9736B",
        ),
        Scenario(
            id="restore",
            name="Восстановление пойм",
            description="Пойменные биотопы в полосе 500 м восстанавливаются, сопротивление снижается вдвое.",
            restoration=0.55,
            restoration_width_m=500.0,
            accent="#6FB8E8",
        ),
    ]


def apply_scenario(
    project: Project,
    surface: ResistanceSurface,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (resistance, landcover, habitat-mask, summary)."""
    resistance = surface.values.astype(np.float64).copy()
    landcover = project.landcover.copy()
    habitat = project.habitat_mask()
    summary: dict = {"converted_ha": 0.0, "road_ha": 0.0, "restored_ha": 0.0}

    if scenario.urban_growth_m > 0:
        urban = landcover == 8
        growth = distance_transform_edt(~urban, sampling=project.res) <= scenario.urban_growth_m
        new_urban = growth & ~urban & (landcover != 1)
        resistance[new_urban] = np.maximum(resistance[new_urban], scenario.urban_resistance)
        landcover[new_urban] = 8
        habitat[new_urban] = False
        summary["converted_ha"] = round(float(new_urban.sum() * project.res**2 / 10_000.0), 1)

    if scenario.new_road:
        line = LineString(scenario.new_road)
        road = project.rasterize(
            gpd.GeoDataFrame(geometry=[line.buffer(scenario.new_road_width_m / 2.0)], crs=project.crs),
            all_touched=True,
        )
        road &= landcover != 1
        resistance[road] = np.maximum(resistance[road], 400.0)
        landcover[road] = 9
        habitat[road] = False
        summary["road_ha"] = round(float(road.sum() * project.res**2 / 10_000.0), 1)

    if scenario.restoration > 0:
        rivers = project.landcover == 1
        riparian = distance_transform_edt(~rivers, sampling=project.res) <= scenario.restoration_width_m
        riparian &= ~binary_dilation(landcover == 8, iterations=2)
        updated = riparian & (resistance > 2.0)
        resistance[updated] *= 1.0 - float(np.clip(scenario.restoration, 0.0, 0.9))
        habitat |= riparian & np.isin(landcover, [2, 5, 7])
        summary["restored_ha"] = round(float(updated.sum() * project.res**2 / 10_000.0), 1)

    resistance = np.clip(resistance, 1.0, surface.params.max_resistance).astype(np.float32)
    summary["mean_resistance"] = round(float(resistance.mean()), 2)
    return resistance, landcover, habitat, summary
