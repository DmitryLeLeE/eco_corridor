"""Online territory loader, exercised end-to-end on synthetic tiles and a fake Overpass."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ecocorridor import analysis, api, online
from ecocorridor.online import OnlineError, TerritoryRequest
from ecocorridor.resistance import slope_degrees
from ecocorridor.scenarios import default_scenarios


# --------------------------------------------------------------------- naming
def test_tile_names_follow_the_published_schemes():
    assert online.worldcover_tile(51, 45) == "N51E045"
    assert online.worldcover_tile(-3, -3) == "S03W003"
    assert online.copdem_tile(51, 46) == "N51_00_E046_00"
    assert online.copdem_tile(-14, -172) == "S14_00_W172_00"


def test_grid_tiles_cover_the_bbox_without_touching_neighbours():
    assert online._grid_tiles(44.78, 51.36, 45.22, 51.64, 3) == [(51, 42), (51, 45)]
    assert online._grid_tiles(44.78, 51.36, 45.22, 51.64, 1) == [(51, 44), (51, 45)]
    assert online._grid_tiles(45.0, 51.0, 46.0, 52.0, 1) == [(51, 45)]  # exact 1-degree cell
    assert online._grid_tiles(-3.5, -1.5, -2.5, -0.5, 3) == [(-3, -6), (-3, -3)]


# ------------------------------------------------------------------- planning
def test_plan_picks_utm_zone_and_resolution():
    ext = online.plan_extent(TerritoryRequest(center=(45.0, 51.5), size_km=30))
    assert ext.crs == "EPSG:32638"
    assert ext.res == 100.0 and ext.width == ext.height == 300
    south = online.plan_extent(TerritoryRequest(center=(-70.0, -33.0), size_km=30))
    assert south.crs == "EPSG:32719"


def test_plan_expands_small_and_crops_huge_territories():
    small = online.plan_extent(TerritoryRequest(center=(45.0, 51.5), size_km=5))
    assert small.width * small.res / 1000 == pytest.approx(online.MIN_SIDE_KM, abs=0.2)
    assert any("расширена" in w for w in small.warnings)
    huge = online.plan_extent(TerritoryRequest(bbox=(40.0, 48.0, 50.0, 56.0)))
    assert huge.area_km2 <= online.MAX_AREA_KM2 * 1.01
    assert any("слишком велика" in w for w in huge.warnings)


def test_plan_respects_resolution_but_caps_the_cell_count():
    ext = online.plan_extent(TerritoryRequest(center=(45.0, 51.5), size_km=100, resolution_m=30))
    assert ext.width * ext.height <= online.MAX_CELLS
    assert ext.res > 30 and any("Разрешение" in w for w in ext.warnings)


@pytest.mark.parametrize(
    "request_",
    [TerritoryRequest(), TerritoryRequest(bbox=(10, 5, 5, 6)), TerritoryRequest(center=(200, 0)), TerritoryRequest(center=(10, 89))],
)
def test_plan_rejects_nonsense(request_):
    with pytest.raises(OnlineError):
        online.plan_extent(request_)


# ------------------------------------------------------------------- rasters
def test_landcover_is_mosaicked_across_tiles_and_mapped(world):
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    lc = online.fetch_landcover(ext)
    assert lc.shape == (ext.height, ext.width) and lc.dtype == np.uint8
    assert set(np.unique(lc)) <= {1, 4, 5, 6, 8}
    assert {1, 4, 5, 6, 8} <= set(np.unique(lc))  # river, forest, grass, crops, town all present
    # The river runs through the middle (the tile seam at 45 E is at x = centre): column of water.
    mid = ext.width // 2
    assert (lc[:, mid - 1 : mid + 1] == 1).mean() > 0.9
    # No gap along the 45 E seam between the two WorldCover tiles (would show up as class 1 = missing).
    assert (lc[ext.height // 2, : mid - 6] == 4).all()
    assert (lc[ext.height // 2, mid + 6 :] == 4).sum() > 0.5 * (ext.width - mid - 6)


def test_landcover_without_any_tile_is_an_error_not_a_sea(world, monkeypatch):
    monkeypatch.setattr(online, "WORLDCOVER_URL", "/nonexistent/{tile}.tif")
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    with pytest.raises(OnlineError, match="WorldCover"):
        online.fetch_landcover(ext)


def test_dem_slope_matches_the_synthetic_ramp(world):
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    dem, note = online.fetch_dem(ext)
    assert note is None
    slope = slope_degrees(dem, ext.res)
    expected = math.degrees(math.atan(3000.0 / 110_570.0))  # ~1.55 degrees
    assert np.median(slope[10:-10, 10:-10]) == pytest.approx(expected, abs=0.1)


def test_dem_falls_back_to_glo90_then_to_flat(world, monkeypatch):
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    monkeypatch.setattr(online, "COPDEM30_URL", "/nonexistent/{tile}.tif")
    dem, note = online.fetch_dem(ext)
    assert "GLO-90" in note and np.isfinite(dem).all() and dem.std() > 0
    monkeypatch.setattr(online, "COPDEM90_URL", "/nonexistent/{tile}.tif")
    dem, note = online.fetch_dem(ext)
    assert "недоступна" in note and float(dem.max()) == 0.0


# ----------------------------------------------------------------------- OSM
def test_overpass_query_is_well_formed(world):
    q = online.overpass_query((45.0, 51.0, 46.0, 52.0), ("motorway", "primary"))
    assert q.startswith("[out:json]") and q.rstrip().endswith("out geom;")
    assert '"^(motorway|motorway_link|primary|primary_link)$"' in q
    assert "(51.00000,45.00000,52.00000,46.00000)" in q


def test_vectors_are_filtered_clipped_and_deduplicated(world):
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    vectors = online.build_vectors(online._overpass("q"), ext, min_core_ha=100.0)
    roads = vectors["roads"]
    assert sorted(roads["kind"]) == ["primary", "secondary", "tertiary", "trunk"]  # tunnel dropped, _link merged
    assert roads.loc[roads["kind"] == "tertiary", "bridge"].all() and not roads.loc[roads["kind"] == "primary", "bridge"].any()
    cores = vectors["cores"]
    assert sorted(cores["name"]) == ["Заказник «Север»", "Заповедник «Тест»"]
    reserve = cores[cores["name"].str.startswith("Заповедник")].iloc[0]
    assert reserve["kind"] == "строгий природный резерват"
    # outer ring is about 7.0 x 11.1 km = 7 700 ha, minus a 1.4 x 2.2 km hole = ~300 ha
    assert 6_500 < reserve["area_ha"] < 8_200
    assert reserve.geometry.interiors and len(list(reserve.geometry.interiors)) == 1
    assert list(vectors["settlements"]["name"]) == ["Тестоград", "Малое"]  # sorted by population
    assert vectors["settlements"]["population"].tolist() == [15000, 0]
    assert len(vectors["rivers"]) == 1
    # everything is inside the frame
    frame = online.box(ext.left, ext.bottom, ext.right, ext.top).buffer(1)
    assert all(frame.contains(g) for g in roads.geometry)


def test_roads_are_burned_into_landcover_and_bridges_cross_water(world):
    ext = online.plan_extent(TerritoryRequest(center=world["center"], size_km=30))
    vectors = online.build_vectors(online._overpass("q"), ext, 100.0)
    lc = np.full((ext.height, ext.width), 4, dtype=np.uint8)
    lc[:, ext.width // 2 - 1 : ext.width // 2 + 1] = 1  # a river
    burned = online.burn_roads(lc, vectors["roads"], ext)
    assert (burned == 9).sum() > 300
    # The tertiary road is a bridge: its crossing of the river is road, not water.
    row = int((ext.top - online.Transformer.from_crs("EPSG:4326", ext.crs, always_xy=True).transform(world["center"][0], world["center"][1] + 0.05)[1]) / ext.res)
    assert burned[row, ext.width // 2] == 9
    # The primary road crosses the river without a bridge: the water cells stay water.
    assert (burned[:, ext.width // 2 - 1 : ext.width // 2 + 1] == 1).any()


# ------------------------------------------------------------- orchestration
def _load(world, **kw):
    req = TerritoryRequest(center=world["center"], size_km=30, name="Тест", **kw)
    stages = []
    project = online.load_territory(req, progress=lambda s, f, m: stages.append((s, f)))
    return project, stages


def test_load_territory_builds_a_working_project_and_caches_it(world):
    project, stages = _load(world)
    assert [s for s, _ in stages][:4] == ["plan", "landcover", "dem", "osm"]
    assert all(a[1] <= b[1] for a, b in zip(stages, stages[1:]))  # progress never goes backwards
    assert project.id.startswith("t") and not project.synthetic
    assert project.shape == (300, 300) and project.crs == "EPSG:32638"
    assert len(project.cores) == 2 and project.meta["core_source"].startswith("OpenStreetMap")
    assert set(np.unique(project.landcover)) <= {c.code for c in project.classes} | {0}
    assert (project.landcover == 9).any()
    assert any("OpenStreetMap" in s for s in project.meta["sources"]) and any("WorldCover" in s for s in project.meta["sources"])
    assert world["overpass_calls"] == 1

    again, stages2 = _load(world)  # served from the on-disk cache
    assert world["overpass_calls"] == 1 and stages2[-1][0] == "cache"
    assert again.id == project.id and again.shape == project.shape
    assert np.array_equal(again.landcover, project.landcover)
    assert len(again.cores) == 2 and len(again.roads) == len(project.roads)
    assert [i["id"] for i in online.list_cached()] == [project.id]


def test_loaded_project_runs_the_whole_pipeline(world):
    project, _ = _load(world)
    scenarios = {s.id: s for s in default_scenarios(project)}
    assert len(scenarios["road"].new_road) == 2  # laid between the two largest cores
    base = analysis.run(project, scenarios["base"])
    assert base.network.stats["link_count"] == 1
    assert base.network.stats["road_crossings"] >= 1
    frag = base.fragmentation
    assert frag["landscape_area_ha"] == pytest.approx(90_000.0) and 0 < frag["pc"] < 1
    road = analysis.run(project, scenarios["road"])
    assert road.fragmentation["pc"] <= frag["pc"] + 1e-9
    assert road.scenario_summary["road_ha"] > 0


def test_habitat_cores_are_the_fallback_when_osm_has_no_protected_areas(world):
    world["protected"] = False
    project, _ = _load(world)
    assert len(project.cores) >= 2
    assert project.cores["kind"].str.contains("лесной массив").all()
    assert any("ядрами выбраны" in w for w in project.meta["warnings"])
    assert analysis.run(project, default_scenarios(project)[0]).network.stats["link_count"] >= 1


def test_forced_protected_source_with_too_few_areas_warns_and_builds_no_network(world):
    world["protected"] = False
    project, _ = _load(world, core_source="protected")
    assert len(project.cores) == 0
    assert any("меньше двух ядер" in w for w in project.meta["warnings"])
    result = analysis.run(project, default_scenarios(project)[0])
    assert result.network.stats["link_count"] == 0


# ----------------------------------------------------------------------- API
def test_territory_api_flow(world, monkeypatch):
    client = TestClient(api.app)
    monkeypatch.setattr(online, "geocode", lambda q, limit=6: [online.Place("Тестоград", "Тестоград, Россия", 45.0, 51.5, (44.9, 51.4, 45.1, 51.6), "place/city", "relation")])
    found = client.get("/api/territory/search", params={"q": "Тестоград"}).json()["places"]
    assert found[0]["name"] == "Тестоград" and found[0]["width_km"] > 10

    plan = client.post("/api/territory/plan", json={"query": "Тестоград", "size_km": 30}).json()
    assert plan["extent"]["resolution_m"] == 100.0 and plan["place"]["name"] == "Тестоград"

    job = client.post("/api/territory/load", json={"query": "Тестоград", "size_km": 30, "name": "Тестоград"}).json()["job_id"]
    for _ in range(300):
        state = client.get(f"/api/territory/jobs/{job}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    pid = state["project_id"]
    assert pid == plan["project_id"]

    info = client.get("/api/project", params={"project_id": pid}).json()
    assert info["online"] is True and info["warnings"] == [] and len(info["layers"]["cores"]["features"]) == 2
    assert {c["code"] for c in info["classes"]} == {1, 2, 4, 5, 6, 7, 8, 9, 10}
    assert any(p["id"] == pid for p in client.get("/api/projects").json()["projects"])

    result = client.post("/api/analyze", json={"project_id": pid, "scenario": "road"}).json()
    assert result["network"]["stats"]["link_count"] == 1 and result["delta"]["pc_pct"] <= 0
    token = result["token"]
    assert client.get(f"/api/layers/{token}/landcover.png").status_code == 200
    assert client.get(f"/api/export/{token}/report.html").status_code == 200
    # the demo project is untouched
    assert client.post("/api/analyze", json={"scenario": "base"}).json()["fragmentation"]["landscape_area_ha"] == pytest.approx(270_000.0)


def test_territory_api_errors_are_readable(world, monkeypatch):
    client = TestClient(api.app)
    assert client.post("/api/territory/plan", json={}).status_code == 422
    assert client.get("/api/territory/search", params={"q": "x"}).status_code == 422
    monkeypatch.setattr(online, "geocode", lambda q, limit=6: [])
    r = client.post("/api/territory/plan", json={"query": "Нигдеград"})
    assert r.status_code == 422 and "не найдена" in r.json()["detail"]
    monkeypatch.setattr(online, "WORLDCOVER_URL", "/nonexistent/{tile}.tif")
    job = client.post("/api/territory/load", json={"center": [45.0, 51.5], "size_km": 30}).json()["job_id"]
    for _ in range(100):
        state = client.get(f"/api/territory/jobs/{job}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["status"] == "error" and "WorldCover" in state["error"]
    assert client.get("/api/territory/jobs/nope").status_code == 404
