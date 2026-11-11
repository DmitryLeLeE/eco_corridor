"""Urbanisation / restoration scenarios.

A scenario is a small, explicit transformation of the resistance surface:

* built-up pressure grows by ``urban_growth_m`` metres around existing
  settlements and the new cells become built-up (high resistance);
* a new road (polyline in map coordinates) is cut into the surface;
* riparian restoration lowers resistance within ``restoration_width_m`` of
  streams, modelling the recovery of floodplain corridors;
* user edits (``edits``), drawn on the map in project coordinates:

  ``road``      polyline, ``width_m`` wide: a new road (resistance >= 400, class 9);
  ``builtup``   polygon: new built-up area (resistance >= urban resistance, class 8);
  ``habitat``   polygon: new forest (class 4, resistance of the forest class);
  ``crossing``  polyline, ``width_m`` wide: a wildlife crossing, resistance capped at 3
                inside the band (the land cover, and therefore the road, stays).

The transformations are applied to the land-cover raster as well, so that the
habitat mask used by the fragmentation metrics changes consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely.geometry import LineString, Polygon

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
    accent: str = "#2F5D8C"
    edits: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "urban_growth_m": self.urban_growth_m,
            "restoration": self.restoration,
            "restoration_width_m": self.restoration_width_m,
            "new_road": self.new_road,
            "edits": self.edits,
            "accent": self.accent,
        }


def _road_between_cores(project: Project) -> list[list[float]] | None:
    """Straight link between the two largest cores (nearest boundary points)."""
    cores = project.cores
    if cores is None or len(cores) < 2:
        return None
    from shapely.ops import nearest_points

    ranked = cores.assign(_area=cores.geometry.area).sort_values("_area", ascending=False).geometry.tolist()[:2]
    a, b = nearest_points(ranked[0], ranked[1])
    if a.distance(b) < project.res * 3:
        return None
    return [[a.x, a.y], [b.x, b.y]]


def default_scenarios(project: Project) -> list[Scenario]:
    """Four illustrative scenarios.

    The demo project uses hand-placed coordinates; for real territories the new
    road is laid along the shortest line between the two largest cores, which
    is where a road does the most harm to the connection between them.
    """
    left, bottom, right, top = project.bounds
    width, height = right - left, top - bottom

    def xy(fx: float, fy: float) -> list[float]:
        return [left + fx * width, bottom + fy * height]

    if project.synthetic:
        road_line = [xy(0.12, 0.30), xy(0.22, 0.44), xy(0.30, 0.62), xy(0.36, 0.78)]
        road_text = "Трасса 60 м пересекает лесной массив между двумя ООПТ."
    else:
        road_line = _road_between_cores(project) or [xy(0.15, 0.25), xy(0.5, 0.5), xy(0.85, 0.75)]
        road_text = "Трасса 60 м прокладывается по кратчайшей линии между двумя крупнейшими ядрами."

    return [
        Scenario(
            id="base",
            name="Базовый (2025)",
            description="Текущая поверхность сопротивления без изменений.",
            accent="#2F5D8C",
        ),
        Scenario(
            id="urban",
            name="Урбанизация +600 м",
            description="Застройка разрастается на 600 м вокруг существующих населённых пунктов.",
            urban_growth_m=600.0,
            accent="#9A5B00",
        ),
        Scenario(
            id="road",
            name="Новая автодорога",
            description=road_text,
            new_road=road_line,
            new_road_width_m=60.0,
            accent="#A32A2A",
        ),
        Scenario(
            id="restore",
            name="Восстановление пойм",
            description="Пойменные биотопы в полосе 500 м восстанавливаются, сопротивление снижается вдвое.",
            restoration=0.55,
            restoration_width_m=500.0,
            accent="#2E7D5B",
        ),
    ]


EDIT_TYPES = ("road", "builtup", "habitat", "crossing")


def edit_geometry(edit: dict):
    """Shapely geometry of one edit (buffered line for ``road`` / ``crossing``)."""
    coords = [(float(x), float(y)) for x, y in edit["coords"]]
    kind = edit["type"]
    if kind in ("road", "crossing"):
        if len(coords) < 2:
            raise ValueError("Линии нужны минимум две точки.")
        return LineString(coords).buffer(float(edit.get("width_m", 60.0)) / 2.0, cap_style="flat")
    if kind in ("builtup", "habitat"):
        if len(coords) < 3:
            raise ValueError("Полигону нужны минимум три точки.")
        polygon = Polygon(coords)
        return polygon if polygon.is_valid else polygon.buffer(0)
    raise ValueError(f"Неизвестный тип правки: {kind}")


def _apply_edits(project, edits, surface, resistance, landcover, habitat, codes, summary) -> float:
    forest = next((c for c in project.classes if c.code == 4), None)
    forest_resistance = surface.params.class_resistance.get(4, forest.resistance if forest else 6.0)
    cells = 0
    for item in edits:
        mask = project.rasterize(gpd.GeoDataFrame(geometry=[edit_geometry(item)], crs=project.crs), all_touched=True)
        kind = item["type"]
        if kind == "road":
            mask &= landcover != 1
            resistance[mask] = np.maximum(resistance[mask], 400.0)
            landcover[mask] = 9
            habitat[mask] = False
        elif kind == "builtup":
            mask &= landcover != 1
            resistance[mask] = np.maximum(resistance[mask], 250.0)
            landcover[mask] = 8
            habitat[mask] = False
        elif kind == "habitat":
            mask &= ~np.isin(landcover, (1, 8, 9))
            resistance[mask] = np.minimum(resistance[mask], forest_resistance)
            landcover[mask] = 4
            habitat[mask] = codes is None or 4 in codes
        elif kind == "crossing":
            resistance[mask] = np.minimum(resistance[mask], 3.0)
        cells += int(mask.sum())
    return round(cells * project.res**2 / 10_000.0, 1)


def apply_scenario(
    project: Project,
    surface: ResistanceSurface,
    scenario: Scenario,
    natural_codes: tuple[int, ...] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (resistance, landcover, habitat-mask, summary)."""
    resistance = surface.values.astype(np.float64).copy()
    landcover = project.landcover.copy()
    codes = list(natural_codes) if natural_codes else None
    habitat = project.habitat_mask(codes)
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
        habitat |= riparian & np.isin(landcover, [c for c in (2, 5, 7) if codes is None or c in codes])
        summary["restored_ha"] = round(float(updated.sum() * project.res**2 / 10_000.0), 1)

    if scenario.edits:
        edited = _apply_edits(project, scenario.edits, surface, resistance, landcover, habitat, codes, summary)
        summary["edited_ha"] = edited

    resistance = np.clip(resistance, 1.0, surface.params.max_resistance).astype(np.float32)
    summary["mean_resistance"] = round(float(resistance.mean()), 2)
    return resistance, landcover, habitat, summary
