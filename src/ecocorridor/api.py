"""FastAPI application: JSON analysis endpoints, PNG layers and exports."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import __version__, circuit, exports, importer, online, render
from .analysis import ModelParams, circuit_flow, forget_project, get, route, run, serialize
from .corridors import CorridorParams
from .dataset import CACHE_DIR, Project, load_demo
from .fragmentation import HABITAT_PRESETS, FragmentationParams
from .resistance import ResistanceParams
from .scenarios import Scenario, default_scenarios

log = logging.getLogger("ecocorridor")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="EcoCorridor", version=__version__, description="Моделирование экологических коридоров")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DEFAULT_PROJECT_ID = "demo"
PROJECTS: dict[str, Project] = {}
SCENARIOS: dict[str, dict[str, Scenario]] = {}
_PROJECT_LOCK = threading.Lock()


def register_project(project: Project) -> Project:
    with _PROJECT_LOCK:
        if project.id in PROJECTS and PROJECTS[project.id] is not project:
            forget_project(project.id)
        PROJECTS[project.id] = project
        SCENARIOS[project.id] = {s.id: s for s in default_scenarios(project)}
    return project


def get_project(project_id: str | None) -> Project:
    """Registered project, or one restored from the on-disk cache."""
    project_id = project_id or DEFAULT_PROJECT_ID
    project = PROJECTS.get(project_id)
    if project is not None:
        return project
    cached = CACHE_DIR / project_id
    if project_id.isalnum() and (cached / "meta.json").exists():
        return register_project(Project.from_directory(cached, project_id=project_id))
    raise HTTPException(status_code=404, detail=f"Unknown project: {project_id}")


register_project(load_demo(DEFAULT_PROJECT_ID))


class EditModel(BaseModel):
    type: Literal["road", "builtup", "habitat", "crossing"]
    coords: list[tuple[float, float]] = Field(min_length=2, max_length=2000, description="vertices in project coordinates (metres)")
    width_m: float = Field(60.0, ge=10.0, le=2000.0)

    @field_validator("coords")
    @classmethod
    def _finite(cls, value):
        if any(not (abs(x) < 1e9 and abs(y) < 1e9) for x, y in value):
            raise ValueError("Координаты вне допустимого диапазона")
        return value


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
    neighbours: Literal[8, 16] = 8
    habitat: Literal["natural", "forest", "forest_wetland"] = "natural"
    urban_growth_m: float | None = Field(None, ge=0.0, le=5000.0)
    restoration: float | None = Field(None, ge=0.0, le=0.9)
    edits: list[EditModel] | None = Field(None, max_length=60, description="scenario edits drawn on the map")
    class_resistance: dict[int, float] | None = Field(
        None, description="Baseline resistance per land-cover class code (1…10), replaces the project's table"
    )

    @field_validator("class_resistance")
    @classmethod
    def _check_class_resistance(cls, value):
        if value is None:
            return value
        for code, resistance in value.items():
            if not 1 <= code <= 10:
                raise ValueError(f"Код класса должен быть 1…10, получено {code}")
            if not 1.0 <= resistance <= 5000.0:
                raise ValueError(f"Сопротивление класса {code} должно быть от 1 до 5000")
        return value


class AnalyzeRequest(BaseModel):
    project_id: str | None = None
    scenario: str = "base"
    params: ParamsModel = Field(default_factory=ParamsModel)


class CompareRequest(BaseModel):
    project_id: str | None = None
    scenarios: list[str] = Field(default_factory=lambda: ["base", "urban", "road", "restore"])
    params: ParamsModel = Field(default_factory=ParamsModel)


class TerritoryModel(BaseModel):
    query: str | None = Field(None, max_length=200)
    place_index: int = Field(0, ge=0, le=20)
    bbox: list[float] | None = Field(None, min_length=4, max_length=4, description="west, south, east, north (WGS84)")
    center: list[float] | None = Field(None, min_length=2, max_length=2, description="lon, lat (WGS84)")
    size_km: float | None = Field(None, ge=5.0, le=400.0)
    resolution_m: float | None = Field(None, ge=10.0, le=2000.0)
    core_source: Literal["auto", "protected", "habitat"] = "auto"
    min_core_ha: float = Field(100.0, ge=10.0, le=100_000.0)
    name: str | None = Field(None, max_length=120)
    force: bool = False

    def to_request(self) -> online.TerritoryRequest:
        return online.TerritoryRequest(
            query=self.query,
            bbox=tuple(self.bbox) if self.bbox else None,  # type: ignore[arg-type]
            center=tuple(self.center) if self.center else None,  # type: ignore[arg-type]
            size_km=self.size_km,
            resolution_m=self.resolution_m,
            core_source=self.core_source,
            min_core_ha=self.min_core_ha,
            name=self.name,
            place_index=self.place_index,
        )


class RouteRequest(BaseModel):
    token: str
    start: tuple[float, float]
    end: tuple[float, float]


def _scenario(project: Project, scenario_id: str, params: ParamsModel) -> Scenario:
    scenario = SCENARIOS[project.id].get(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {scenario_id}")
    overrides = {}
    if params.urban_growth_m is not None:
        overrides["urban_growth_m"] = params.urban_growth_m
    if params.restoration is not None:
        overrides["restoration"] = params.restoration
    if params.edits:
        for item in params.edits:
            minimum = 3 if item.type in ("builtup", "habitat") else 2
            if len(item.coords) < minimum:
                raise HTTPException(status_code=422, detail=f"Правка «{item.type}» требует минимум {minimum} точки.")
        overrides["edits"] = [{"type": e.type, "coords": [list(c) for c in e.coords], "width_m": e.width_m} for e in params.edits]
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
            class_resistance=dict(params.class_resistance or {}),
        ),
        fragmentation=FragmentationParams(
            dispersal_m=params.dispersal_m,
            min_patch_ha=params.min_patch_ha,
            natural_codes=HABITAT_PRESETS[params.habitat],
        ),
        corridors=CorridorParams(
            max_link_km=params.max_link_km,
            corridor_width_m=params.corridor_width_m,
            detour_ratio=params.detour_ratio,
            neighbours=params.neighbours,
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
    return {"status": "ok", "version": __version__, "project": DEFAULT_PROJECT_ID}


def _project_payload(project: Project) -> dict:
    defaults = ParamsModel()
    meta = project.meta or {}
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "crs": project.crs,
        "bounds": project.bounds,
        "resolution_m": project.res,
        "width": project.width,
        "height": project.height,
        "extent_lonlat": project.extent_lonlat(),
        "synthetic": project.synthetic,
        "online": bool(meta.get("online")),
        "imported": bool(meta.get("imported")),
        "sources": meta.get("sources", []),
        "warnings": meta.get("warnings", []),
        "core_source": meta.get("core_source"),
        "classes": [asdict(c) for c in project.classes],
        "scenarios": [s.as_dict() for s in SCENARIOS[project.id].values()],
        "layers": {
            "cores": _fc(project.cores, keep=["name", "kind", "area_ha"]),
            "roads": _fc(project.roads, keep=["name", "width_m", "kind"]),
            "rivers": _fc(project.rivers, keep=["name", "order", "width_m"]),
            "settlements": _fc(project.settlements, keep=["name", "population", "kind"]),
            "region": _fc(project.region, keep=[]),
        },
        "defaults": defaults.model_dump(),
        "version": __version__,
    }


@app.get("/api/project")
def project_info(project_id: str | None = None) -> dict:
    return _project_payload(get_project(project_id))


@app.get("/api/projects")
def projects_list() -> dict:
    items = [{"id": DEFAULT_PROJECT_ID, "name": PROJECTS[DEFAULT_PROJECT_ID].name, "synthetic": True}]
    seen = {DEFAULT_PROJECT_ID}
    for item in online.list_cached():
        if item["id"] not in seen:
            items.append({**item, "synthetic": False})
            seen.add(item["id"])
    return {"projects": items}


# ---------------------------------------------------------------------------
# Online territories
# ---------------------------------------------------------------------------
_JOBS: "OrderedDict[str, dict]" = OrderedDict()
_JOBS_LOCK = threading.Lock()
_JOB_LIMIT = 20
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="territory")  # one at a time: be polite to public services


def _set_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id].update(fields)


def _run_job(job_id: str, request: online.TerritoryRequest, force: bool) -> None:
    def progress(stage: str, fraction: float, message: str) -> None:
        _set_job(job_id, status="running", stage=stage, fraction=round(fraction, 3), message=message)

    try:
        project = online.load_territory(request, progress=progress, force=force)
        register_project(project)
        _set_job(job_id, status="done", stage="done", fraction=1.0, message="Готово", project_id=project.id, finished=time.time())
    except online.OnlineError as exc:
        _set_job(job_id, status="error", error=str(exc), finished=time.time())
    except Exception as exc:  # noqa: BLE001 - reported to the client, full trace goes to the log
        log.exception("territory load failed")
        _set_job(job_id, status="error", error=f"Внутренняя ошибка загрузки: {exc}", finished=time.time())


@app.get("/api/territory/search")
def territory_search(q: str = Query(..., min_length=2, max_length=200)) -> dict:
    try:
        places = online.geocode(q)
    except online.OnlineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"places": [p.as_dict() for p in places]}


@app.post("/api/territory/plan")
def territory_plan(request: TerritoryModel) -> dict:
    try:
        extent, place = online.resolve_request(request.to_request())
    except online.OnlineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"extent": extent.as_dict(), "place": place.as_dict() if place else None, "project_id": request.to_request().fingerprint(extent)}


@app.post("/api/territory/load")
def territory_load(request: TerritoryModel) -> dict:
    with _JOBS_LOCK:
        if any(job["status"] in ("queued", "running") for job in _JOBS.values()):
            raise HTTPException(status_code=409, detail="Уже идёт загрузка другой территории — дождитесь её завершения.")
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "fraction": 0.0,
            "message": "В очереди…",
            "error": None,
            "project_id": None,
            "started": time.time(),
        }
        while len(_JOBS) > _JOB_LIMIT:
            _JOBS.popitem(last=False)
    _EXECUTOR.submit(_run_job, job_id, request.to_request(), request.force)
    return {"job_id": job_id}


@app.get("/api/territory/jobs/{job_id}")
def territory_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(job)


@app.post("/api/import")
async def import_data(
    request: Request,
    name: str | None = Query(None, max_length=120),
    scheme: Literal["ecocorridor", "worldcover", "custom"] = "ecocorridor",
    resolution_m: float | None = Query(None, ge=1.0, le=5000.0),
    min_core_ha: float = Query(100.0, ge=1.0, le=100_000.0),
) -> dict:
    """Own data: the request body is a ZIP archive (land cover raster + optional DEM and vector layers)."""
    if os.environ.get("ECOCORRIDOR_DISABLE_IMPORT"):
        raise HTTPException(status_code=403, detail="Загрузка своих данных отключена на этом сервере.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > importer.MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail=f"Архив больше {importer.MAX_ZIP_BYTES // (1024 * 1024)} МБ.")
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=422, detail="Пустой запрос: приложите ZIP-архив с данными.")
    with _JOBS_LOCK:
        if any(job["status"] in ("queued", "running") for job in _JOBS.values()):
            raise HTTPException(status_code=409, detail="Уже идёт загрузка — дождитесь её завершения.")
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "id": job_id, "status": "queued", "stage": "queued", "fraction": 0.0, "message": "В очереди…",
            "error": None, "project_id": None, "started": time.time(),
        }
        while len(_JOBS) > _JOB_LIMIT:
            _JOBS.popitem(last=False)

    def work() -> None:
        def progress(stage: str, fraction: float, message: str) -> None:
            _set_job(job_id, status="running", stage=stage, fraction=round(fraction, 3), message=message)

        try:
            project = importer.import_zip(payload, name, scheme, resolution_m, min_core_ha, progress)
            register_project(project)
            _set_job(job_id, status="done", stage="done", fraction=1.0, message="Готово", project_id=project.id, finished=time.time())
        except online.OnlineError as exc:
            _set_job(job_id, status="error", error=str(exc), finished=time.time())
        except Exception as exc:  # noqa: BLE001
            log.exception("import failed")
            _set_job(job_id, status="error", error=f"Внутренняя ошибка импорта: {exc}", finished=time.time())

    _EXECUTOR.submit(work)
    return {"job_id": job_id}


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    project = get_project(request.project_id)
    params = _params(request.params)
    scenario = _scenario(project, request.scenario, request.params)
    result = run(project, scenario, params)
    base = run(project, SCENARIOS[project.id]["base"], params) if scenario.id != "base" else None
    return serialize(project, result, base)


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    project = get_project(request.project_id)
    params = _params(request.params)
    base = run(project, SCENARIOS[project.id]["base"], params)
    rows = []
    for scenario_id in request.scenarios:
        scenario = _scenario(project, scenario_id, request.params)
        result = run(project, scenario, params)
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
                        "eca_ha",
                        "patch_count",
                        "largest_patch_ha",
                        "lpi",
                        "edge_density_m_per_ha",
                        "largest_component_share",
                        "total_area_ha",
                        "landscape_area_ha",
                    )
                },
                "network": result.network.stats,
                "resistance_mean": round(float(result.resistance.mean()), 2),
                "delta": serialize(project, result, base).get("delta", {}),
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
            rgba = render.hillshade_rgba(result.project)
        elif layer == "landcover":
            rgba = render.landcover_rgba(result.project, result.landcover)
        elif layer == "resistance":
            rgba = render.resistance_rgba(result.project, result.resistance)
        elif layer == "cost":
            rgba = render.cost_rgba(result.project, result.cost)
        elif layer == "current":
            try:
                solved = circuit_flow(result)[0]
            except circuit.CircuitError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            rgba = render.current_rgba(result.project, solved.current)
        else:
            raise HTTPException(status_code=404, detail="Unknown layer")
        cached = render.png_bytes(rgba)
        _LAYER_CACHE[key] = cached
        while len(_LAYER_CACHE) > _LAYER_LIMIT:
            _LAYER_CACHE.popitem(last=False)
    return Response(content=cached, media_type="image/png", headers={"Cache-Control": "public, max-age=7200"})


@app.get("/api/circuit/{token}")
def circuit_summary(token: str) -> dict:
    """Circuit-theory summary: effective resistance between cores and cumulative-current statistics."""
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired, please re-run")
    try:
        solved, cores, seconds = circuit_flow(result)
    except circuit.CircuitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "cores": cores,
        "effective_resistance": np.round(solved.effective_resistance, 3).tolist(),
        "stats": solved.stats(),
        "seconds": seconds,
        "method": "Circuit theory (McRae et al. 2008): 8 neighbours, mean resistance of the two cells, all core pairs",
    }


@app.post("/api/route")
def build_route(request: RouteRequest) -> dict:
    result = get(request.token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired, please re-run")
    path = route(result.project, result, tuple(request.start), tuple(request.end))
    if path is None:
        raise HTTPException(status_code=422, detail="No path between the selected points")
    return path


@app.get("/api/sample/{token}")
def sample(token: str, x: float, y: float) -> dict:
    """Resistance / land-cover / cost readout for the cursor."""
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    project = result.project
    col, row = ~project.transform * (x, y)
    if not (0 <= row < project.height and 0 <= col < project.width):
        raise HTTPException(status_code=404, detail="Outside the raster")
    row, col = int(row), int(col)
    code = int(result.landcover[row, col])
    name = next((c.name for c in project.classes if c.code == code), "—")
    return {
        "resistance": round(float(result.resistance[row, col]), 2),
        "cost": round(float(result.cost[row, col]), 1) if np.isfinite(result.cost[row, col]) else None,
        "class_code": code,
        "class_name": name,
        "x": round(x, 1),
        "y": round(y, 1),
        "lonlat": list(project.to_lonlat(x, y)),
    }


@app.get("/api/export/{token}/corridors.geojson")
def export_corridors(token: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    features = _fc(result.network.lines, keep=None)
    features["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{result.project.crs.split(':')[-1]}"}}
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
        ("Площадь ландшафта, га", result.fragmentation.get("landscape_area_ha")),
        ("Общая площадь местообитаний, га", result.fragmentation.get("total_area_ha")),
        ("Доля местообитаний", result.fragmentation.get("habitat_share")),
        ("MESH, га", result.fragmentation.get("mesh_ha")),
        ("PC", result.fragmentation.get("pc")),
        ("ECA (эквивалентная связная площадь), га", result.fragmentation.get("eca_ha")),
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

    html = build_report(result.project, result)
    return Response(
        html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ecocorridor_report.html"'},
    )


@app.get("/api/export/{token}/{layer}.tif")
def export_raster(token: str, layer: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    if layer not in exports.RASTER_LAYERS:
        raise HTTPException(status_code=404, detail=f"Unknown raster: {layer}")
    return Response(
        exports.raster_bytes(result, layer),
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="ecocorridor_{layer}.tif"'},
    )


@app.get("/api/export/{token}/result.gpkg")
def export_geopackage(token: str) -> Response:
    result = get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis expired")
    return Response(
        exports.geopackage_bytes(result),
        media_type="application/geopackage+sqlite3",
        headers={"Content-Disposition": 'attachment; filename="ecocorridor.gpkg"'},
    )
