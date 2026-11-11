"""Server-side map rendering (PNG layers for the browser client).

Rasters are rendered at their native resolution and stylised with a hillshade
multiplier, so that relief remains readable under the thematic colours.  All
layers are RGBA; the client draws them as images positioned by their extent.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .dataset import Project
from .resistance import ResistanceSurface

RESISTANCE_STOPS: list[tuple[float, tuple[int, int, int]]] = [  # ColorBrewer YlOrRd (sequential)
    (0.00, (255, 255, 204)),
    (0.25, (254, 217, 118)),
    (0.50, (253, 141, 60)),
    (0.75, (227, 26, 28)),
    (1.00, (128, 0, 38)),
]

COST_STOPS: list[tuple[float, tuple[int, int, int]]] = [  # ColorBrewer Blues: accessible (dark) -> remote (pale)
    (0.00, (8, 48, 107)),
    (0.25, (33, 113, 181)),
    (0.50, (107, 174, 214)),
    (0.75, (198, 219, 239)),
    (1.00, (247, 251, 255)),
]


def _stops_to_lut(stops: list[tuple[float, tuple[int, int, int]]], size: int = 256) -> np.ndarray:
    positions = np.array([s[0] for s in stops], dtype=np.float64)
    colors = np.array([s[1] for s in stops], dtype=np.float64)
    xs = np.linspace(0.0, 1.0, size)
    lut = np.empty((size, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.interp(xs, positions, colors[:, channel]).astype(np.uint8)
    return lut


RESISTANCE_LUT = _stops_to_lut(RESISTANCE_STOPS)
COST_LUT = _stops_to_lut(COST_STOPS)

CURRENT_STOPS: list[tuple[float, tuple[int, int, int]]] = [  # dark violet -> orange -> pale yellow (sequential)
    (0.00, (45, 16, 96)),
    (0.30, (124, 29, 111)),
    (0.55, (212, 72, 66)),
    (0.80, (250, 160, 58)),
    (1.00, (252, 255, 164)),
]
CURRENT_LUT = _stops_to_lut(CURRENT_STOPS)


def shade_field(project: Project, exaggeration: float = 2.2) -> np.ndarray:
    """Hillshade multiplier in roughly 0.55..1.35 (1.0 = flat lit surface)."""
    dem = np.asarray(project.dem, dtype=np.float64)
    if not np.isfinite(dem).all():
        dem = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    dem = gaussian_filter(dem, 0.8)
    gy, gx = np.gradient(dem, project.res)
    gx, gy = gx * exaggeration, gy * exaggeration
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = math.radians(315.0), math.radians(42.0)
    shade = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    lo, hi = np.percentile(shade, 2), np.percentile(shade, 98)
    norm = np.clip((shade - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return 0.55 + 0.80 * norm


def _tone(rgb: np.ndarray, shade: np.ndarray, strength: float = 1.0) -> np.ndarray:
    factor = 1.0 - strength + strength * shade
    return np.clip(rgb * factor[..., None], 0, 255)


def hillshade_rgba(project: Project) -> np.ndarray:
    shade = shade_field(project)
    grey = np.clip((shade - 0.55) / 0.80 * 255.0, 0, 255).astype(np.uint8)
    rgba = np.empty((*grey.shape, 4), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = grey
    rgba[..., 3] = 255
    return rgba


def landcover_rgba(project: Project, landcover: np.ndarray | None = None) -> np.ndarray:
    data = project.landcover if landcover is None else landcover
    rgb = np.zeros((*project.shape, 3), dtype=np.float64)
    for cls in project.classes:
        rgb[data == cls.code] = cls.rgb
    rgb[data == 0] = (90, 90, 90)
    shade = shade_field(project, exaggeration=1.6)
    toned = _tone(rgb, shade, strength=0.85)
    rgba = np.empty((*project.shape, 4), dtype=np.uint8)
    rgba[..., :3] = toned.astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _apply_lut(values: np.ndarray, lut: np.ndarray, vmin: float, vmax: float, log: bool) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if log:
        data = np.log(np.maximum(data, 1e-3))
        vmin, vmax = math.log(max(vmin, 1e-3)), math.log(max(vmax, 1e-3))
    t = np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    idx = (t * (len(lut) - 1)).astype(np.uint8)
    return lut[idx]


def resistance_rgba(project: Project, values: np.ndarray, alpha: int = 235) -> np.ndarray:
    vmin = 1.0
    vmax = float(max(np.percentile(values, 99.5), 10.0))
    rgb = _apply_lut(values, RESISTANCE_LUT, vmin, vmax, log=True).astype(np.float64)
    shade = shade_field(project)
    rgb = _tone(rgb, shade, strength=0.55)
    rgba = np.empty((*project.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def cost_rgba(project: Project, cost: np.ndarray) -> np.ndarray:
    finite = cost[np.isfinite(cost)]
    vmax = float(np.percentile(finite, 92)) if finite.size else 1.0
    vmax = max(vmax, 1.0)
    rgb = _apply_lut(np.nan_to_num(cost, nan=vmax, posinf=vmax), COST_LUT, 0.0, vmax, log=False).astype(np.float64)
    shade = shade_field(project, exaggeration=1.4)
    rgb = _tone(rgb, shade, strength=0.45)
    t = np.clip(np.nan_to_num(cost, nan=vmax, posinf=vmax) / vmax, 0.0, 1.0)
    rgba = np.empty((*project.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = np.clip(230 * (1.0 - t**0.6), 0, 230).astype(np.uint8)
    return rgba


def current_rgba(project: Project, current: np.ndarray) -> np.ndarray:
    """Cumulative circuit current: log scale between the median and the 99.7th percentile of non-zero cells."""
    values = np.asarray(current, dtype=np.float64)
    positive = values[values > 0]
    rgba = np.zeros((*project.shape, 4), dtype=np.uint8)
    if positive.size == 0:
        return rgba
    vmin = float(np.percentile(positive, 40))
    vmax = float(max(np.percentile(positive, 99.7), vmin * 1.5))
    rgb = _apply_lut(values, CURRENT_LUT, vmin, vmax, log=True).astype(np.float64)
    shade = shade_field(project, exaggeration=1.4)
    rgb = _tone(rgb, shade, strength=0.35)
    t = np.clip((np.log(np.maximum(values, 1e-9)) - math.log(vmin)) / (math.log(vmax) - math.log(vmin)), 0.0, 1.0)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = np.where(values > 0, 60 + 190 * t, 0).astype(np.uint8)
    return rgba


def png_bytes(rgba: np.ndarray) -> bytes:
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def legend(ramp: str = "resistance") -> list[dict]:
    stops = RESISTANCE_STOPS if ramp == "resistance" else COST_STOPS
    return [{"position": p, "color": "#%02x%02x%02x" % c} for p, c in stops]
