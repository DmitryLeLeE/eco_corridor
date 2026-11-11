"""Resistance ("cost of movement") surface.

The surface combines a land-cover baseline with distance-decayed pressure from
roads and built-up areas and a slope penalty.  Two combination modes are
available:

``multiplicative`` (default)
    R = base * (1 + w_s * s) * (1 + w_r * e^(-d_r/D_r)) * (1 + w_u * e^(-d_u/D_u))

    Multiplicative composition is the common choice in landscape connectivity
    studies: a barrier stays a barrier even where the land cover is otherwise
    favourable, and weights are interpretable as relative multipliers.

``weighted_sum``
    R = 1 + (R_max - 1) * (b + w_s*s + w_r*e^(-d_r/D_r) + w_u*e^(-d_u/D_u)) / (1 + w_s + w_r + w_u)

    A classic weighted overlay, useful when all criteria should be expressed in
    the same unit (0..1).  ``b`` is the normalised land-cover resistance.

where ``s`` is slope in degrees clipped to ``slope_ref_deg`` (0..1),
``d_r``/``d_u`` are distances to the nearest road / built-up cell in metres and
``D_r``/``D_u`` their influence radii.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .dataset import Project


@dataclass
class ResistanceParams:
    mode: str = "multiplicative"
    slope_weight: float = 0.9
    road_weight: float = 0.8
    urban_weight: float = 0.6
    road_decay_m: float = 350.0
    urban_decay_m: float = 1200.0
    slope_ref_deg: float = 22.0
    max_resistance: float = 5000.0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "slope_weight": self.slope_weight,
            "road_weight": self.road_weight,
            "urban_weight": self.urban_weight,
            "road_decay_m": self.road_decay_m,
            "urban_decay_m": self.urban_decay_m,
            "slope_ref_deg": self.slope_ref_deg,
            "max_resistance": self.max_resistance,
        }


@dataclass
class ResistanceSurface:
    values: np.ndarray
    base: np.ndarray
    slope_deg: np.ndarray
    dist_road: np.ndarray
    dist_urban: np.ndarray
    params: ResistanceParams
    stats: dict = field(default_factory=dict)
    class_summary: list[dict] = field(default_factory=list)


def slope_degrees(dem: np.ndarray, res: float) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float64)
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    smooth = gaussian_filter(filled, 0.6)
    gy, gx = np.gradient(smooth, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def _distance_to(mask: np.ndarray, res: float, smooth: float = 0.0) -> np.ndarray:
    distance = distance_transform_edt(~mask, sampling=res)
    if smooth:
        distance = gaussian_filter(distance, smooth)
    return distance


def pressure_masks(project: Project) -> tuple[np.ndarray, np.ndarray]:
    """Road and built-up masks, combining the land-cover raster with vectors."""
    road = project.landcover == 9
    urban = project.landcover == 8
    if not project.roads.empty:
        road |= project.rasterize(project.roads.buffer(project.res * 0.5), all_touched=True)
    return road, urban


def build_resistance(project: Project, params: ResistanceParams | None = None) -> ResistanceSurface:
    params = params or ResistanceParams()
    res = project.res

    base_map = np.ones(project.shape, dtype=np.float64)
    for cls in project.classes:
        base_map[project.landcover == cls.code] = cls.resistance
    # Unknown classes (0 / missing) are treated as moderately permeable.
    base_map[project.landcover == 0] = 20.0

    slope = slope_degrees(project.dem, res)
    road_mask, urban_mask = pressure_masks(project)
    dist_road = _distance_to(road_mask, res, smooth=0.6)
    dist_urban = _distance_to(urban_mask, res, smooth=0.6)

    slope_norm = np.clip(slope / max(params.slope_ref_deg, 1e-6), 0.0, 1.0)
    road_infl = np.exp(-dist_road / max(params.road_decay_m, 1.0))
    urban_infl = np.exp(-dist_urban / max(params.urban_decay_m, 1.0))

    if params.mode == "weighted_sum":
        base_norm = np.clip(base_map / max(float(np.nanmax(base_map)), 1.0), 0.0, 1.0)
        total_w = 1.0 + params.slope_weight + params.road_weight + params.urban_weight
        combined = (
            base_norm
            + params.slope_weight * slope_norm
            + params.road_weight * road_infl
            + params.urban_weight * urban_infl
        ) / total_w
        values = 1.0 + (params.max_resistance - 1.0) * combined
    else:
        values = (
            base_map
            * (1.0 + params.slope_weight * slope_norm)
            * (1.0 + params.road_weight * road_infl)
            * (1.0 + params.urban_weight * urban_infl)
        )

    values = np.clip(values, 1.0, params.max_resistance)
    values = np.round(values, 3).astype(np.float32)
    water = project.landcover == 1
    values[water] = np.maximum(values[water], 600.0)

    stats = {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "share_above_100": float((values > 100).mean()),
    }
    class_summary = []
    for cls in project.classes:
        mask = project.landcover == cls.code
        if not mask.any():
            continue
        class_summary.append(
            {
                "code": cls.code,
                "name": cls.name,
                "color": cls.color,
                "base_resistance": cls.resistance,
                "mean_resistance": round(float(values[mask].mean()), 2),
                "share": round(float(mask.mean()), 4),
                "area_ha": round(float(mask.sum() * res * res / 10_000), 1),
            }
        )

    return ResistanceSurface(
        values=values,
        base=base_map.astype(np.float32),
        slope_deg=slope.astype(np.float32),
        dist_road=dist_road.astype(np.float32),
        dist_urban=dist_urban.astype(np.float32),
        params=params,
        stats=stats,
        class_summary=class_summary,
    )
