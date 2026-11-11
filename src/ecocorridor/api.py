"""FastAPI application: JSON analysis endpoints, PNG layers and exports."""

from __future__ import annotations

import csv
import io
import numpy as np
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, analysis as analysis_mod, render
from .analysis import ModelParams, get, route, run, serialize
from .corridors import CorridorParams
from .dataset import Project, load_demo
from .fragmentation import FragmentationParams
from .resistance import ResistanceParams
from .scenarios import Scenario, default_scenarios

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="EcoCorridor", version=__version__, description="Моделирование экологических коридоров")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROJECT: Project = load_demo()
SCENARIOS: dict[str, Scenario] = {s.id: s for s in default_scenarios(PROJECT)}


class ParamsModel(BaseModel):
    mode: str = "multiplicative"
    slope_weight: float = Field(0.9, ge=0.0, le=3.0)
    road_weight: float = Field(0.8, ge=0.0, le=3.0)
    urban_weight: float = Field(0.6, ge=0.0, le=3.0)
    road_decay_m: float = Field(350.0, ge=50.0, le=3000.0)
    urban_decay_m: float = Field(1200.0, ge=100.0, le=8000.0)
    slope_ref_deg: float = Field(22.0, ge=5.0, le=60.0)
    max_link_km: float = Field(34.0, ge=1.0, le=120.0)
    corridor_width_m: float = Field(250.0, ge=50.0, le=2000.0)
    detour_ratio: float = Field(1.25, ge=1.0, le=3.0)
    dispersal_m: float = Field(5000.0, ge=500.0, le=50000.0)
    min_patch_ha: float = Field(5.0, ge=0.5, le=200.0)
    urban_growth_m: float | None = Field(None, ge=0.0, le=5000.0)
    restoration: float | None = Field(None, ge=0.0, le=0.9)


class AnalyzeRequest(BaseModel):
    scenario: str = "base"
    params: ParamsModel = Field(default_factory=ParamsModel)


class CompareRequest(BaseModel):
    scenarios: list[str] = Field(default_factory=lambda: ["base", "urban", "road", "restore"])
    params: ParamsModel = Field(default_factory=ParamsModel)


class RouteRequest(BaseModel):
    token: str
    start: tuple[float, float]
    end: tuple[float, float]


def _scenario(scenario_id: str, params: ParamsModel) -> Scenario:
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {scenario_id}")
    overrides = {}
    if params.urban_growth_m is not None:
        overrides["urban_growth_m"] = params.urban_growth_m
    if params.restoration is not None:
        overrides["restoration"] = params.restoration
    return replace(scenario, **overrides) if overrides else scenario


def _params(params: ParamsModel) -> ModelParams:
    return ModelParams(
        resistance=ResistanceParams(
            mode=params.mode,
            slope_weight=params.slope_weight,
            road_weight=params.road_weight,
            urban_weight=params.urban_weight,
            road_decay_m=params.road_decay_m,
            urban_decay_m=params.urban_decay_m,
            slope_ref_deg=params.slope_ref_deg,
        ),
        fragmentation=FragmentationParams(dispersal_m=params.dispersal_m, min_patch_ha=params.min_patch_ha),
        corridors=CorridorParams(
            max_link_km=params.max_link_km,
            corridor_width_m=params.corridor_width_m,
            detour_ratio=params.detour_ratio,
        ),
    )


def _fc(gdf, keep: list[str] | None = None) -> dict:
    from .analysis import _feature_collection

    return _feature_collection(gdf, keep=keep)


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "project": PROJECT.id}


@app.get("/api/project")
def project_info() -> dict:
    defaults = ParamsModel()
    return {
        "id": PROJECT.id,
        "name": PROJECT.name,
        "description": PROJECT.description,
        "crs": PROJECT.crs,
        "bounds": PROJECT.bounds,
        "resolution_m": PROJECT.res,
        "width": PROJECT.width,
        "height": PROJECT.height,
        "extent_lonlat": PROJECT.extent_lonlat(),
        "synthetic": PROJECT.synthetic,
        "classes": [asdict(c) for c in PROJECT.classes],
        "scenarios": [s.as_dict() for s in SCENARIOS.values()],
        "layers": {
            "cores": _fc(PROJECT.cores, keep=["name", "kind", "area_ha"]),
            "roads": _fc(PROJECT.roads, keep=["name", "width_m"]),
            "rivers": _fc(PROJECT.rivers, keep=["name", "order", "width_m"]),
            "settlements": _fc(PROJECT.settlements, keep=["name", "population", "kind"]),
            "region": _fc(PROJECT.region, keep=[]),
        },
        "defaults": defaults.model_dump(),
        "version": __version__,
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    params = _params(request.params)
    scenario = _scenario(request.scenario, request.params)
    result = run(PROJECT, scenario, params)
    base = run(PROJECT, SCENARIOS["base"], params) if scenario.id != "base" else None
    return serialize(PROJECT, result, base)


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    params = _params(request.params)
    base = run(PROJECT, SCENARIOS["base"], params)
    rows = []
    for scenario_id in request.scenarios:
        scenario = _scenario(scenario_id, request.params)
        result = run(PROJECT, scenario, params)
        rows.append(
            {
                "id": scenario.id,
                "name": scenario.name,
                "accent": scenario.accent,
                "token": result.token,
                "elapsed": result.elapsed,
                "scenario_summary": result.scenario_summary,
                "fragmentation": {
                    key: result.fragmentation.get(key)
                    for key in (
                        "mesh_ha",
                        "pc",
                        "iic",
                        "patch_count",
                        "largest_patch_ha",
                        "lpi",
                        "edge_density_m_per_ha",
                        "largest_component_share",
                        "total_area_ha",
                    )
                },
                "network": result.network.stats,
                "resistance_mean": round(float(result.resistance.mean()), 2),
                "delta": serialize(PROJECT, result, base).get("delta", {}),
            }
        )
    return {"base_token": base.token, "rows": rows}


_LAYER_CACHE: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
_LAYER_LIMIT = 24


@app.get("/api/layers/{token}/{layer}.png")
def layer_png(token: str, layer: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired, please re-run")
    key = (token, layer)
    cached = _LAYER_CACHE.get(key)
    if cached is None:
        if layer == "hillshade":
            rgba = render.hillshade_rgba(PROJECT)
        elif layer == "landcover":
            rgba = render.landcover_rgba(PROJECT, result.landcover)
        elif layer == "resistance":
            rgba = render.resistance_rgba(PROJECT, result.resistance)
        elif layer == "cost":
            rgba = render.cost_rgba(PROJECT, result.cost)
        else:
            raise HTTPException(status_code=404, detail="Unknown layer")
        cached = render.png_bytes(rgba)
        _LAYER_CACHE[key] = cached
        while len(_LAYER_CACHE) > _LAYER_LIMIT:
            _LAYER_CACHE.popitem(last=False)
    return Response(content=cached, media_type="image/png", headers={"Cache-Control": "public, max-age=7200"})


@app.post("/api/route")
def build_route(request: RouteRequest) -> dict:
    result = get(request.token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired, please re-run")
    path = route(PROJECT, result, tuple(request.start), tuple(request.end))
    if path is None:
        raise HTTPException(status_code=422, detail="No path between the selected points")
    return path


@app.get("/api/sample/{token}")
def sample(token: str, x: float, y: float) -> dict:
    """Resistance / land-cover / cost readout for the cursor."""
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    col, row = ~PROJECT.transform * (x, y)
    if not (0 <= row < PROJECT.height and 0 <= col < PROJECT.width):
        raise HTTPException(status_code=404, detail="Outside the raster")
    row, col = int(row), int(col)
    code = int(result.landcover[row, col])
    name = next((c.name for c in PROJECT.classes if c.code == code), "—")
    return {
        "resistance": round(float(result.resistance[row, col]), 2),
        "cost": round(float(result.cost[row, col]), 1) if np.isfinite(result.cost[row, col]) else None,
        "class_code": code,
        "class_name": name,
        "x": round(x, 1),
        "y": round(y, 1),
        "lonlat": list(PROJECT.to_lonlat(x, y)),
    }


@app.get("/api/export/{token}/corridors.geojson")
def export_corridors(token: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    features = _fc(result.network.lines, keep=None)
    features["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{PROJECT.crs.split(':')[-1]}"}}
    body = json.dumps(features, ensure_ascii=False)
    return Response(body, media_type="application/geo+json", headers={"Content-Disposition": 'attachment; filename="corridors.geojson"'})


@app.get("/api/export/{token}/metrics.csv")
def export_metrics(token: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    rows = [
        ("Сценарий", result.scenario.name),
        ("Площадь ООПТ, га", sum(result.cores.get("area_ha", []))),
        ("Патчей местообитаний", result.fragmentation.get("patch_count")),
        ("Общая площадь местообитаний, га", result.fragmentation.get("total_area_ha")),
        ("MESH, га", result.fragmentation.get("mesh_ha")),
        ("PC", result.fragmentation.get("pc")),
        ("IIC", result.fragmentation.get("iic")),
        ("Доля крупнейшего патча, %", result.fragmentation.get("lpi")),
        ("Плотность границ, м/га", result.fragmentation.get("edge_density_m_per_ha")),
        ("Коридоров", result.network.stats.get("link_count")),
        ("Длина сети, км", result.network.stats.get("total_length_km")),
        ("Площадь коридоров, га", result.network.stats.get("corridor_area_ha")),
        ("Пересечений с дорогами", result.network.stats.get("road_crossings")),
        ("Узких мест", result.network.stats.get("pinchpoint_count")),
        ("Среднее сопротивление коридоров", result.network.stats.get("mean_resistance")),
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(["Показатель", "Значение"])
    for name, value in rows:
        writer.writerow([name, value])
    return Response(
        "\ufeff" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ecocorridor_metrics.csv"'},
    )


@app.get("/api/export/{token}/report.html", response_class=HTMLResponse)
def export_report(token: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    from .report import build_report

    html = build_report(PROJECT, result)
    return Response(
        html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ecocorridor_report.html"'},
    )
