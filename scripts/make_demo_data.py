"""Generate the reproducible synthetic demo landscape used by EcoCorridor.

The demo region ("Синий Бор", 60 x 45 km, UTM zone 37N, EPSG:32637) is fully
synthetic: terrain, river network, land cover, roads, settlements and four
protected areas are produced by this script with a fixed seed.  It is
intentionally similar to a fragmented forested low-mountain region so that the
whole modelling chain (resistance surface -> least-cost paths -> connectivity
metrics) has something meaningful to show.

Run:
    python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter, label
from shapely.geometry import LineString, Point, box, mapping, shape

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "ecocorridor" / "demo_data"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Region definition
# ----------------------------------------------------------------------------
CRS = "EPSG:32637"
X0, Y0 = 380_000.0, 6_060_000.0  # lower-left corner
WIDTH, HEIGHT = 60_000.0, 45_000.0  # metres
RES = 100.0
NX, NY = int(WIDTH / RES), int(HEIGHT / RES)
TRANSFORM = from_origin(X0, Y0 + HEIGHT, RES, RES)
SEED = 20_240_517

LN, LT = np.meshgrid((np.arange(NX) + 0.5) * RES, (np.arange(NY) + 0.5) * RES)
E, N = X0 + LN, Y0 + LT


def fractal(shape: tuple[int, int], beta: float, seed: int, smooth: float = 0.0) -> np.ndarray:
    """Spectral synthesis of a Gaussian random field with power ~ k^-beta."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = 1e-6
    field = np.fft.ifft2((k ** (-beta / 2.0)) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).real
    if smooth:
        field = gaussian_filter(field, smooth)
    field -= field.mean()
    return field / (field.std() + 1e-9)


# ----------------------------------------------------------------------------
# 1. Terrain: forested ridge in the north, agricultural plain in the south
# ----------------------------------------------------------------------------
ridge_axis = Y0 + 30_500.0 + 2_400.0 * np.sin(2 * np.pi * (E - X0) / 72_000.0) + 0.045 * (E - X0)
ridge = 360.0 * np.exp(-(((N - ridge_axis) / 5_200.0) ** 2))
# A wind gap / mountain pass that localises future corridors.
pass_x = X0 + 37_000.0
pass_dip = 1.0 - 0.58 * np.exp(-(((E - pass_x) / 4_200.0) ** 2)) * np.exp(-(((N - ridge_axis) / 8_500.0) ** 2))
relief = 42.0 * fractal((NY, NX), beta=3.0, seed=SEED, smooth=2.4)
relief += 6.0 * fractal((NY, NX), beta=2.2, seed=SEED + 1, smooth=1.6)
dem = 118.0 + ridge * pass_dip + relief
dem = gaussian_filter(dem, 1.6)
dem = np.clip(dem, 90.0, 600.0)

# ----------------------------------------------------------------------------
# 2. Rivers: a dendritic network of smooth polylines, cut into the terrain
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED + 31)
rivers: list[tuple[int, float, float, LineString]] = []  # order, width_m, depth_m, line


def grow(x: float, y: float, angle: float, length: float, width: float, depth: float, order: int) -> None:
    pts = [(x, y)]
    steps = max(4, int(length / 450.0))
    phase = rng.uniform(0, 2 * np.pi)
    for i in range(steps):
        angle += rng.normal(0.0, 0.10) + 0.055 * np.sin(i * 0.33 + phase)
        nx = x + 450.0 * np.cos(angle)
        ny = y + 450.0 * np.sin(angle)
        if not (X0 - 2_000 < nx < X0 + WIDTH + 2_000 and Y0 - 2_000 < ny < Y0 + HEIGHT + 2_000):
            break
        x, y = nx, ny
        pts.append((x, y))
        if order < 2 and i > 4 and rng.random() < 0.035:
            side = 1.0 if rng.random() < 0.5 else -1.0
            grow(
                x,
                y,
                angle + side * rng.uniform(0.55, 1.15),
                length * rng.uniform(0.35, 0.55),
                width * 0.55,
                depth * 0.62,
                order + 1,
            )
    if len(pts) >= 3:
        line = LineString(pts)
        # Continue the channel to the region edge so that every stream has an outlet.
        end = pts[-1]
        if X0 < end[0] < X0 + WIDTH and Y0 < end[1] < Y0 + HEIGHT:
            dx, dy = end[0] - pts[-2][0], end[1] - pts[-2][1]
            norm = np.hypot(dx, dy) or 1.0
            dx, dy = dx / norm, dy / norm
            extra = [(end[0] + dx * 600.0 * k, end[1] + dy * 600.0 * k) for k in range(1, 7)]
            extra = [(px, py) for px, py in extra if X0 - 2_000 < px < X0 + WIDTH + 2_000 and Y0 - 2_000 < py < Y0 + HEIGHT + 2_000]
            if len(extra) > 1:
                line = LineString(pts + extra)
        rivers.append((order, width, depth, line))


grow(X0 + 17_500.0, Y0 + 43_000.0, -1.42, 44_000.0, 95.0, 32.0, 0)
grow(X0 + 46_000.0, Y0 + 43_000.0, -1.85, 34_000.0, 75.0, 26.0, 0)
grow(X0 + 58_500.0, Y0 + 33_000.0, -2.45, 24_000.0, 55.0, 18.0, 0)

# Rasterise the water network: wide order-0 stems, narrow headwaters.
water = np.zeros((NY, NX), dtype=bool)
depth_field = np.zeros((NY, NX), dtype=float)
for order, width, depth, line in sorted(rivers, key=lambda r: -r[0]):
    band = rasterize([(line.buffer(width / 2.0), 1)], out_shape=(NY, NX), transform=TRANSFORM, all_touched=True) > 0
    water |= band
    depth_field[band] = np.maximum(depth_field[band], depth)

# Lakes: irregular lowland water bodies on the main stem.
lake_noise = gaussian_filter(fractal((NY, NX), 3.2, SEED + 7, 4.0), 1.0)
lake = np.zeros((NY, NX), dtype=bool)
lake_centres: list[tuple[float, float]] = []
main_stem = max((r for r in rivers if r[0] == 0), key=lambda r: r[3].length)[3]
for frac in (0.42, 0.72):
    pt = main_stem.interpolate(frac, normalized=True).coords[0]
    cx, cy = pt
    if any(np.hypot(cx - px, cy - py) < 10_000.0 for px, py in lake_centres):
        continue
    lake_centres.append((cx, cy))
    lake |= ((E - cx) / 2_300.0) ** 2 + ((N - cy) / 1_100.0) ** 2 + 0.55 * lake_noise < 1.0
water |= lake

# Carve valleys into the DEM so that the terrain and the hydrography agree.
dist_river = distance_transform_edt(~water, sampling=RES)
dem -= 40.0 * np.exp(-((dist_river / 950.0) ** 2))
dem -= 12.0 * np.exp(-((dist_river / 320.0) ** 2))
dem = gaussian_filter(dem, 1.0)

gy, gx = np.gradient(dem, RES)
slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
slope_pct = np.hypot(gx, gy) * 100.0

# ----------------------------------------------------------------------------
# 3. Settlements and roads
# ----------------------------------------------------------------------------
towns = [
    ("Северный Лог", X0 + 12_400.0, Y0 + 12_600.0, 12_800, 2.4),
    ("Синий Бор", X0 + 31_200.0, Y0 + 10_400.0, 8_400, 1.9),
    ("Заречье", X0 + 47_300.0, Y0 + 14_200.0, 5_100, 1.3),
]
villages = [
    ("Крутой Яр", X0 + 8_200.0, Y0 + 7_100.0, 420, 0.55),
    ("Пихтовка", X0 + 19_800.0, Y0 + 15_600.0, 260, 0.5),
    ("Малиновка", X0 + 27_600.0, Y0 + 5_900.0, 180, 0.45),
    ("Ольховка", X0 + 38_900.0, Y0 + 8_200.0, 310, 0.55),
    ("Сосновый Гай", X0 + 44_100.0, Y0 + 20_100.0, 150, 0.4),
    ("Верхний Плёс", X0 + 54_900.0, Y0 + 9_600.0, 240, 0.5),
]
settlements = towns + villages

urban = np.zeros((NY, NX), dtype=bool)
urban_noise = gaussian_filter(fractal((NY, NX), 3.6, SEED + 3, 3.0), 1.0)
for _name, cx, cy, pop, r_km in settlements:
    r = r_km * 1000.0
    # A settlement is a union of a few irregular built-up lobes, not one disc.
    lobes = 4 if pop >= 5_000 else 1
    for k in range(lobes):
        off = r * 0.42 if k else 0.0
        ang = rng.uniform(0, np.pi) if k else 0.0
        ox = cx + off * np.cos(ang) * (0.5 + 0.5 * urban_noise[np.clip(int((cy - Y0) / RES), 0, NY - 1), np.clip(int((cx - X0) / RES), 0, NX - 1)])
        oy = cy + off * np.sin(ang)
        a = r * (0.55 if k else 1.0) * (1.0 + 0.10 * np.sin(k * 2.1))
        b = a * rng.uniform(0.55, 0.9)
        rot = rng.uniform(0, np.pi)
        u = (E - ox) * np.cos(rot) + (N - oy) * np.sin(rot)
        v = -(E - ox) * np.sin(rot) + (N - oy) * np.cos(rot)
        urban |= (u / a) ** 2 + (v / b) ** 2 + 0.18 * urban_noise < 1.0
urban = gaussian_filter(urban.astype(float), 1.1) > 0.45


def catmull_rom(points: list[tuple[float, float]], samples_per_segment: int = 40) -> LineString:
    """Smooth interpolating spline through the given control points."""
    pts = np.asarray(points, dtype=float)
    padded_pts = np.vstack([pts[0] + (pts[0] - pts[1]), pts, pts[-1] + (pts[-1] - pts[-2])])
    out: list[tuple[float, float]] = []
    for i in range(1, len(padded_pts) - 2):
        p0, p1, p2, p3 = padded_pts[i - 1 : i + 3]
        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append((float(pt[0]), float(pt[1])))
    return LineString(out)


highway = catmull_rom(
    [
        (X0 + 500.0, Y0 + 9_000.0),
        (X0 + 12_400.0, Y0 + 12_600.0),
        (X0 + 22_000.0, Y0 + 10_000.0),
        (X0 + 31_200.0, Y0 + 10_400.0),
        (X0 + 39_000.0, Y0 + 12_000.0),
        (X0 + 47_300.0, Y0 + 14_200.0),
        (X0 + 59_500.0, Y0 + 13_000.0),
    ]
)
north_road = catmull_rom(
    [
        (X0 + 31_200.0, Y0 + 10_400.0),
        (X0 + 33_500.0, Y0 + 21_000.0),
        (X0 + 37_000.0, Y0 + 28_500.0),
        (X0 + 36_400.0, Y0 + 36_500.0),
        (X0 + 39_500.0, Y0 + 44_800.0),
    ]
)
east_road = catmull_rom(
    [
        (X0 + 47_300.0, Y0 + 14_200.0),
        (X0 + 44_400.0, Y0 + 22_000.0),
        (X0 + 45_600.0, Y0 + 31_000.0),
        (X0 + 52_000.0, Y0 + 44_800.0),
    ]
)
spurs = [
    ("Подъезд к Пихтовке", [(X0 + 12_400.0, Y0 + 12_600.0), (X0 + 16_000.0, Y0 + 14_000.0), (X0 + 19_800.0, Y0 + 15_600.0)], 30.0),
    ("Подъезд к Ольховке", [(X0 + 38_000.0, Y0 + 11_800.0), (X0 + 38_900.0, Y0 + 8_200.0)], 30.0),
    ("Подъезд к Малиновке", [(X0 + 28_500.0, Y0 + 8_900.0), (X0 + 27_600.0, Y0 + 5_900.0)], 30.0),
    ("Подъезд к Верхнему Плёсу", [(X0 + 52_500.0, Y0 + 13_100.0), (X0 + 54_900.0, Y0 + 9_600.0)], 30.0),
    ("Подъезд к Крутому Яру", [(X0 + 10_500.0, Y0 + 11_400.0), (X0 + 8_200.0, Y0 + 7_100.0)], 30.0),
]
roads = [
    ("Автодорога Р-21", highway, 70.0),
    ("Автодорога 37К-4", north_road, 50.0),
    ("Автодорога 37К-7", east_road, 45.0),
] + [(name, catmull_rom(pts), 30.0) for name, pts, width in spurs]

road_mask = np.zeros((NY, NX), dtype=bool)
road_core = np.zeros((NY, NX), dtype=bool)
for _name, geom, width in roads:
    road_mask |= rasterize([(geom.buffer(width), 1)], out_shape=(NY, NX), transform=TRANSFORM, all_touched=True) > 0
    road_core |= rasterize([(geom.buffer(width * 0.35), 1)], out_shape=(NY, NX), transform=TRANSFORM, all_touched=True) > 0

# ----------------------------------------------------------------------------
# 4. Land cover
# ----------------------------------------------------------------------------
LC = {
    1: ("Вода", "#2E6E8E", 800),
    2: ("Пойменные водно-болотные угодья", "#5E8A6B", 45),
    3: ("Тёмнохвойный лес", "#1B4A38", 3),
    4: ("Смешанный лес", "#2F6B4C", 6),
    5: ("Луга и пастбища", "#9FAF74", 18),
    6: ("Пашня", "#CDB77F", 40),
    7: ("Редколесье и гари", "#A98B60", 28),
    8: ("Селитебная застройка", "#A9695C", 250),
    9: ("Дороги", "#E4D3A6", 400),
    10: ("Скалы и обнажения", "#8E9188", 90),
}

dist_settle = distance_transform_edt(~urban, sampling=RES)
dist_road = distance_transform_edt(~road_mask, sampling=RES)

# Large-scale ecological gradients, smoothed so that patch boundaries are coherent.
gradient = gaussian_filter(fractal((NY, NX), 3.1, SEED + 11, 7.0), 1.2)
patchiness = gaussian_filter(fractal((NY, NX), 3.4, SEED + 12, 4.0), 0.8)
conifer_field = gaussian_filter(fractal((NY, NX), 3.0, SEED + 13, 5.0), 0.8)

landcover = np.full((NY, NX), 4, dtype=np.uint8)

# Forest cover probability: high on the ridge, lower on the plain around towns.
forest_index = 0.55 * (dem - 180.0) / 220.0 - 0.35 * np.exp(-dist_settle / 6_000.0) + 0.65 * gradient
forest = forest_index > 0.02
landcover[~forest] = 5
# Arable land: flat, low, near settlements, away from streams.
arable = (~forest) & (dem < 235) & (slope_deg < 4.0) & (dist_settle < 8_500) & (dist_river > 350) & (patchiness < 0.9)
landcover[arable] = 6
# Sparse / burnt stands on steep or high ground.
sparse = ((slope_deg > 23.0) & (dem > 280)) | ((dem > 430) & (patchiness > -0.2))
landcover[sparse] = 7
# Rock outcrops on the highest, steepest ridge crests.
landcover[(dem > 450) & (slope_deg > 26)] = 10
# Conifer forest at higher, cooler elevations, mixed forest low down.
landcover[(landcover == 4) & ((dem > 300) | (conifer_field > 0.55))] = 3
# Logging clearings: compact blobs inside the forest.
clearing = (gaussian_filter(fractal((NY, NX), 3.8, SEED + 21, 8.0), 1.0) > 2.1) & np.isin(landcover, [3, 4])
landcover[clearing] = 7
# Broad forest gaps (old burns, abandoned farmland) fragment the massifs.
gap = gaussian_filter(fractal((NY, NX), 3.5, SEED + 22, 5.0), 1.0) > 1.35
landcover[gap & np.isin(landcover, [3, 4])] = 5
# Riparian wetlands and meadows along streams.
landcover[(dist_river < 300) & (slope_deg < 3.0) & (dem < 320)] = 2
# Water, settlements and roads always win.
landcover[urban] = 8
landcover[road_mask] = 9
landcover[road_core] = 9
landcover[water] = 1

# ----------------------------------------------------------------------------
# 5. Protected areas: forest cores clipped by a circular design envelope
# ----------------------------------------------------------------------------
# A real protected-area network is not a random sample of forest patches: it is
# a designed system spread over the region.  We place five envelopes and keep
# the forest (or, failing that, natural habitat) inside them, so the geometry
# stays natural while the spatial layout is controlled.
targets = [
    ("Заказник «Кедровый Лог»", "заказник", X0 + 13_000.0, Y0 + 26_500.0, 2_700.0),
    ("Заповедник «Синий Бор»", "заповедник", X0 + 30_500.0, Y0 + 33_000.0, 2_500.0),
    ("Заказник «Верхний Плёс»", "заказник", X0 + 50_500.0, Y0 + 31_000.0, 2_400.0),
    ("Памятник природы «Сосновый Гай»", "памятник природы", X0 + 19_000.0, Y0 + 10_000.0, 2_200.0),
    ("Заказник «Тихая Пойма»", "заказник", X0 + 41_000.0, Y0 + 11_500.0, 2_100.0),
]
forest_here = np.isin(landcover, [3, 4])
natural_here = np.isin(landcover, [2, 3, 4, 5, 7])
envelope_noise = gaussian_filter(fractal((NY, NX), 3.3, SEED + 41, 4.0), 1.0)
protected = []
for name, kind, cx, cy, radius in targets:
    best_geom = None
    best_score = -1.0
    for scale in (1.0, 1.3, 1.6, 2.0, 2.4):
        envelope = (E - cx) ** 2 + (N - cy) ** 2 < (radius * scale * (1.0 + 0.26 * envelope_noise)) ** 2
        for pool in (forest_here, natural_here):
            mask = envelope & pool
            mask = gaussian_filter(mask.astype(float), 1.2) > 0.45
            if mask.sum() * RES * RES / 10_000.0 < 250.0:
                continue
            sub, sub_n = label(mask, structure=np.ones((3, 3)))
            if sub_n == 0:
                continue
            sizes = np.bincount(sub.ravel())
            sizes[0] = 0
            comp = sub == int(np.argmax(sizes))
            polys = [shape(g) for g, v in rasterio.features.shapes(comp.astype(np.uint8), mask=comp, transform=TRANSFORM) if v == 1]
            if not polys:
                continue
            geom = max(polys, key=lambda p: p.area)
            compactness = 4.0 * np.pi * geom.area / max(geom.length**2, 1e-9)
            score = compactness + 0.02 * min(geom.area / 10_000.0, 2_000.0) / 1_000.0
            if score > best_score:
                best_geom, best_score = geom, score
        if best_geom is not None and best_score > 0.35:
            break
    if best_geom is None:
        continue
    protected.append({"name": name, "kind": kind, "geometry": best_geom.simplify(120.0)})

pa_gdf = gpd.GeoDataFrame(protected, crs=CRS)
pa_gdf["area_ha"] = (pa_gdf.area / 10_000).round(1)
pa_gdf = pa_gdf[["name", "kind", "area_ha", "geometry"]]

# ----------------------------------------------------------------------------
# 6. Write the dataset
# ----------------------------------------------------------------------------
def write_raster(path: Path, array: np.ndarray, dtype: str, nodata: float | int) -> None:
    profile = {
        "driver": "GTiff",
        "height": NY,
        "width": NX,
        "count": 1,
        "dtype": dtype,
        "crs": CRS,
        "transform": TRANSFORM,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


write_raster(OUT / "dem.tif", dem.astype(np.float32), "float32", -9999.0)
write_raster(OUT / "landcover.tif", landcover.astype(np.uint8), "uint8", 0)

with open(OUT / "landcover_classes.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "field": "class",
            "classes": [
                {"code": code, "name": name, "color": color, "resistance": resistance}
                for code, (name, color, resistance) in sorted(LC.items())
            ],
        },
        fh,
        ensure_ascii=False,
        indent=2,
    )

settlement_features = []
for name, cx, cy, pop, _r in settlements:
    kind = "город" if pop >= 5_000 else "село"
    settlement_features.append(
        {"type": "Feature", "properties": {"name": name, "population": pop, "kind": kind}, "geometry": mapping(Point(cx, cy))}
    )
(OUT / "settlements.geojson").write_text(
    json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32637"}},
            "features": settlement_features,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

road_features = [
    {"type": "Feature", "properties": {"name": name, "width_m": width}, "geometry": mapping(geom)}
    for name, geom, width in roads
]
(OUT / "roads.geojson").write_text(
    json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32637"}},
            "features": road_features,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

river_features = []
for i, (order, width, depth, line) in enumerate(sorted(rivers, key=lambda r: r[0])):
    river_features.append(
        {
            "type": "Feature",
            "properties": {"name": f"Водоток {i + 1}", "order": order, "width_m": width},
            "geometry": mapping(line.simplify(30.0)),
        }
    )
(OUT / "rivers.geojson").write_text(
    json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32637"}},
            "features": river_features,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

pa_gdf.to_file(OUT / "protected_areas.geojson", driver="GeoJSON")
gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + WIDTH, Y0 + HEIGHT)], crs=CRS).to_file(OUT / "region.geojson", driver="GeoJSON")

meta = {
    "name": "Демонстрационный полигон «Синий Бор»",
    "description": "Синтетический ландшафт 60 × 45 км: низкогорный хребет на севере, освоенная равнина на юге, четыре ООПТ.",
    "crs": CRS,
    "resolution_m": RES,
    "width": NX,
    "height": NY,
    "bounds": [X0, Y0, X0 + WIDTH, Y0 + HEIGHT],
    "synthetic": True,
    "seed": SEED,
    "files": {
        "dem": "dem.tif",
        "landcover": "landcover.tif",
        "classes": "landcover_classes.json",
        "protected_areas": "protected_areas.geojson",
        "roads": "roads.geojson",
        "rivers": "rivers.geojson",
        "settlements": "settlements.geojson",
        "region": "region.geojson",
    },
}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"written {OUT}")
print(f"  grid {NX}x{NY} @ {RES:.0f} m, DEM {dem.min():.0f}-{dem.max():.0f} m")
print(f"  land cover classes: {sorted(np.unique(landcover).tolist())}")
print(f"  rivers: {len(rivers)}, protected areas: {len(pa_gdf)}")
for rec in pa_gdf.itertuples():
    print(f"    {rec.area_ha:.0f} ha")
