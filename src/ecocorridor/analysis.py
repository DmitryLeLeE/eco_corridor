"""Full analysis run: resistance -> corridors -> fragmentation, with caching."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np

from .corridors import CorridorNetwork, CorridorParams, build_network
from .costpath import RasterGraph, path_resistance_stats
from .dataset import Project
from .fragmentation import FragmentationParams, analyse as analyse_fragmentation
from .resistance import ResistanceParams, ResistanceSurface, build_resistance
from .scenarios import Scenario, apply_scenario


@dataclass
class ModelParams:
    resistance: ResistanceParams = field(default_factory=ResistanceParams)
    fragmentation: FragmentationParams = field(default_factory=FragmentationParams)
    corridors: CorridorParams = field(default_factory=CorridorParams)

    def as_dict(self) -> dict:
        return {
            "resistance": self.resistance.as_dict(),
            "fragmentation": self.fragmentation.as_dict(),
            "corridors": self.corridors.as_dict(),
        }


@dataclass
class AnalysisResult:
    token: str
    project: Project
    scenario: Scenario
    params: ModelParams
    surface: ResistanceSurface
    resistance: np.ndarray
    landcover: np.ndarray
    habitat: np.ndarray
    graph: RasterGraph
    network: CorridorNetwork
    fragmentation: dict
    cost: np.ndarray
    cores: gpd.GeoDataFrame
    scenario_summary: dict
    elapsed: float

    def cache_key(self) -> str:
        return self.token


_CACHE: "OrderedDict[str, AnalysisResult]" = OrderedDict()
_CACHE_LIMIT = 6


def make_token(project: Project, scenario: Scenario, params: ModelParams) -> str:
    payload = json.dumps(
        {"project": project.id, "scenario": scenario.as_dict(), "params": params.as_dict()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def run(project: Project, scenario: Scenario, params: ModelParams | None = None) -> AnalysisResult:
    params = params or ModelParams()
    token = make_token(project, scenario, params)
    cached = _CACHE.get(token)
    if cached is not None:
        _CACHE.move_to_end(token)
        return cached

    started = time.perf_counter()
    surface = build_resistance(project, params.resistance)
    resistance, landcover, habitat, scenario_summary = apply_scenario(
        project, surface, scenario, params.fragmentation.natural_codes
    )
    graph = RasterGraph(resistance, project.res, neighbours=params.corridors.neighbours)
    network = build_network(project, graph, resistance, params.corridors)

    view = project
    if scenario.id != "base":
        # Fragmentation must be evaluated on the scenario land cover.
        view = _with_landcover(project, landcover)
    fragmentation = analyse_fragmentation(view, habitat, resistance, params.fragmentation, graph=graph)

    core_cells = graph.cells_of(project.core_mask())
    cost = graph.cost_surface(core_cells) if core_cells.size else np.full(project.shape, np.nan)

    cores = _core_features(project, network, resistance)
    elapsed = round(time.perf_counter() - started, 2)
    result = AnalysisResult(
        token=token,
        project=project,
        scenario=scenario,
        params=params,
        surface=surface,
        resistance=resistance,
        landcover=landcover,
        habitat=habitat,
        graph=graph,
        network=network,
        fragmentation=fragmentation,
        cost=cost.astype(np.float32),
        cores=cores,
        scenario_summary=scenario_summary,
        elapsed=elapsed,
    )
    _CACHE[token] = result
    _CACHE.move_to_end(token)
    while len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return result


def get(token: str) -> AnalysisResult | None:
    return _CACHE.get(token)


def forget_project(project_id: str) -> None:
    """Drop cached results of a project whose data has been replaced."""
    for token in [t for t, r in _CACHE.items() if r.project.id == project_id]:
        del _CACHE[token]


def _with_landcover(project: Project, landcover: np.ndarray) -> Project:
    clone = Project(
        id=project.id,
        name=project.name,
        description=project.description,
        crs=project.crs,
        transform=project.transform,
        width=project.width,
        height=project.height,
        res=project.res,
        dem=project.dem,
        landcover=landcover,
        classes=project.classes,
        cores=project.cores,
        roads=project.roads,
        rivers=project.rivers,
        settlements=project.settlements,
        region=project.region,
        synthetic=project.synthetic,
        meta=project.meta,
    )
    return clone


def _core_features(project: Project, network: CorridorNetwork, resistance: np.ndarray) -> gpd.GeoDataFrame:
    base = project.cores.reset_index(drop=True).copy()
    if base.empty:
        return base
    base["core_id"] = base.index.astype(int)
    by_id = {int(node["id"]): node for node in network.nodes}
    for column, fallback in (
        ("betweenness", 0.0),
        ("degree", 0),
        ("mean_resistance", 0.0),
    ):
        base[column] = [by_id.get(int(i), {}).get(column, fallback) for i in base["core_id"]]
    if "area_ha" not in base.columns:
        base["area_ha"] = (base.geometry.area / 10_000.0).round(1)
    if "kind" not in base.columns:
        base["kind"] = "ООПТ"
    return base


def route(project: Project, result: AnalysisResult, start: tuple[float, float], end: tuple[float, float]) -> dict | None:
    path = result.graph.least_cost_points(project, start, end)
    if path is None:
        return None
    stats = path_resistance_stats(project, result.resistance, path.cells)
    return {
        "coords": [[round(x, 2), round(y, 2)] for x, y in path.coords],
        "cost": round(path.cost, 1),
        "length_km": round(path.length_m / 1000.0, 2),
        "mean_resistance": stats["mean"],
        "max_resistance": stats["max"],
        "straight_km": round(math.dist(start, end) / 1000.0, 2),
    }


def serialize(project: Project, result: AnalysisResult, base_result: AnalysisResult | None = None) -> dict:
    """JSON-ready payload for the browser client."""
    network = result.network
    frag = result.fragmentation
    payload = {
        "token": result.token,
        "scenario": result.scenario.as_dict(),
        "scenario_summary": result.scenario_summary,
        "elapsed": result.elapsed,
        "params": result.params.as_dict(),
        "layers": {
            "hillshade": _layer(result, "hillshade"),
            "landcover": _layer(result, "landcover"),
            "resistance": _layer(result, "resistance"),
            "cost": _layer(result, "cost"),
        },
        "resistance": {
            "stats": result.surface.stats,
            "classes": result.surface.class_summary,
            "legend": _legend(),
        },
        "cores": _feature_collection(result.cores, keep=["core_id", "name", "kind", "area_ha", "degree", "betweenness", "mean_resistance"]),
        "corridors": _feature_collection(network.lines, keep=None),
        "corridor_area": _feature_collection(network.polygons, keep=None),
        "pinchpoints": _feature_collection(network.pinchpoints, keep=None),
        "network": {"stats": network.stats, "nodes": network.nodes, "links": network.links},
        "fragmentation": frag,
        "meta": {
            "bounds": project.bounds,
            "crs": project.crs,
            "resolution_m": project.res,
            "width": project.width,
            "height": project.height,
        },
    }
    if base_result is not None and base_result.token != result.token:
        payload["delta"] = _delta(base_result, result)
    return payload


def _delta(base: AnalysisResult, current: AnalysisResult) -> dict:
    base_frag, cur_frag = base.fragmentation, current.fragmentation
    curve = {
        "mesh_ha_pct": _pct(cur_frag.get("mesh_ha"), base_frag.get("mesh_ha")),
        "pc_pct": _pct(cur_frag.get("pc"), base_frag.get("pc")),
        "pc_cost_pct": _pct(cur_frag.get("pc_cost"), base_frag.get("pc_cost")),
        "eca_cost_pct": _pct(cur_frag.get("eca_cost_ha"), base_frag.get("eca_cost_ha")),
        "iic_pct": _pct(cur_frag.get("iic"), base_frag.get("iic")),
        "link_length_pct": _pct(current.network.stats.get("total_length_km"), base.network.stats.get("total_length_km")),
        "corridor_area_pct": _pct(current.network.stats.get("corridor_area_ha"), base.network.stats.get("corridor_area_ha")),
    }
    return curve


def _pct(value: float | None, base: float | None) -> float | None:
    if not value or not base:
        return None
    return round((float(value) / float(base) - 1.0) * 100.0, 1)


def _layer(result: AnalysisResult, name: str) -> dict:
    return {
        "url": f"/api/layers/{result.token}/{name}.png",
        "bounds": None,
        "name": name,
    }


def _legend() -> list[dict]:
    from .render import RESISTANCE_STOPS

    return [{"position": p, "color": "#%02x%02x%02x" % c} for p, c in RESISTANCE_STOPS]


def _feature_collection(gdf: gpd.GeoDataFrame, keep: list[str] | None) -> dict:
    features = []
    if gdf is None or len(gdf) == 0:
        return {"type": "FeatureCollection", "features": []}
    for row in gdf.itertuples(index=False):
        geom = getattr(row, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        props = {}
        for column in gdf.columns:
            if column == "geometry":
                continue
            if keep is not None and column not in keep:
                continue
            value = getattr(row, column, None)
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            if isinstance(value, (int, float, str, bool)) or value is None:
                props[column] = value
        features.append({"type": "Feature", "properties": props, "geometry": _geometry_to_dict(geom)})
    return {"type": "FeatureCollection", "features": features}


def _geometry_to_dict(geom) -> dict:
    from shapely.geometry import mapping

    return mapping(geom)


def _decimals(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)
