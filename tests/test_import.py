"""Import of user data (ZIP / folder) and the GIS exports."""

from __future__ import annotations

import io
import json
import time
import zipfile

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from ecocorridor import api, importer
from ecocorridor.analysis import run
from ecocorridor.dataset import load_demo


@pytest.fixture()
def demo_dir(tmp_path):
    out = tmp_path / "own"
    load_demo().to_directory(out)
    return out


def zip_dir(path, rename=None) -> bytes:
    rename = rename or {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for file in sorted(path.iterdir()):
            archive.write(file, rename.get(file.name, file.name))
    return buffer.getvalue()


def test_import_roundtrip_matches_demo(demo_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "CACHE_DIR", tmp_path / "cache")
    project = importer.import_directory(demo_dir, name="Своё")
    demo = load_demo()
    assert project.crs == demo.crs
    assert project.shape == demo.shape
    assert project.res == demo.res
    assert (project.landcover == demo.landcover).all()
    assert len(project.cores) == len(demo.cores)
    assert project.id.startswith("u") and project.id.isalnum()
    assert (tmp_path / "cache" / project.id / "meta.json").exists()
    # Same data -> same analysis.
    a = run(demo, api.SCENARIOS["demo"]["base"])
    b = run(project, api.default_scenarios(project)[0])
    assert b.fragmentation["pc"] == pytest.approx(a.fragmentation["pc"], rel=1e-9)


def test_roles_recognised_by_name(demo_dir, tmp_path):
    (demo_dir / "dem.tif").rename(demo_dir / "SRTM_elevation.tif")
    (demo_dir / "protected_areas.geojson").rename(demo_dir / "OOPT.geojson")
    found = importer.find_files(demo_dir)
    assert found["landcover"].name == "landcover.tif"
    assert found["dem"].name == "SRTM_elevation.tif"
    assert found["cores"].name == "OOPT.geojson"
    assert found["roads"].name == "roads.geojson"


def test_missing_landcover_is_a_clear_error(tmp_path):
    (tmp_path / "roads.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    with pytest.raises(importer.ImportError_, match="землепокров"):
        importer.import_directory(tmp_path, save=False)


def test_two_landcover_rasters_are_rejected(demo_dir):
    (demo_dir / "lulc_2.tif").write_bytes((demo_dir / "landcover.tif").read_bytes())
    with pytest.raises(importer.ImportError_, match="несколько"):
        importer.import_directory(demo_dir, save=False)


def _geographic_raster(path, codes):
    """A small WGS84 raster near Saratov filled with ``codes`` in stripes."""
    height, width = 120, 160
    data = np.zeros((height, width), dtype="uint8")
    for i, code in enumerate(codes):
        data[:, i::len(codes)] = code
    transform = from_origin(46.0, 52.0, 0.003, 0.003)
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=transform) as dst:
        dst.write(data, 1)


def test_geographic_raster_is_reprojected_to_utm(tmp_path):
    _geographic_raster(tmp_path / "landcover.tif", [4, 4, 5, 6])
    project = importer.import_directory(tmp_path, save=False, min_core_ha=50)
    assert project.crs.startswith("EPSG:326")
    assert 25 < project.res < 400
    assert set(np.unique(project.landcover)) <= {0, 4, 5, 6}
    assert any("переведён" in w for w in project.meta["warnings"])
    assert any("DEM" in w for w in project.meta["warnings"])
    assert np.allclose(project.dem, 0.0)


def test_worldcover_scheme(tmp_path):
    _geographic_raster(tmp_path / "landcover.tif", [10, 30, 40, 80])
    project = importer.import_directory(tmp_path, scheme="worldcover", save=False, min_core_ha=50)
    present = set(np.unique(project.landcover)) - {0}
    assert present <= {4, 5, 6, 1}
    assert {4, 5, 6, 1} <= present


def test_custom_map_and_class_override(tmp_path):
    _geographic_raster(tmp_path / "landcover.tif", [111, 211, 112])
    (tmp_path / "classes.json").write_text(
        json.dumps({"map": {"111": 4, "211": 6, "112": 8},
                    "classes": [{"code": 4, "name": "Хвойный лес", "color": "#112233", "resistance": 2}]}),
        encoding="utf-8",
    )
    project = importer.import_directory(tmp_path, scheme="custom", save=False, min_core_ha=50)
    assert set(np.unique(project.landcover)) - {0} <= {4, 6, 8}
    forest = next(c for c in project.classes if c.code == 4)
    assert (forest.name, forest.resistance) == ("Хвойный лес", 2.0)


def test_wrong_scheme_for_codes_is_rejected(tmp_path):
    _geographic_raster(tmp_path / "landcover.tif", [111, 211])
    with pytest.raises(importer.ImportError_, match="не входящие"):
        importer.import_directory(tmp_path, scheme="ecocorridor", save=False)


def test_custom_scheme_requires_map(tmp_path):
    _geographic_raster(tmp_path / "landcover.tif", [4])
    with pytest.raises(importer.ImportError_, match="map"):
        importer.import_directory(tmp_path, scheme="custom", save=False)


def test_raster_without_crs_is_rejected(tmp_path):
    with rasterio.open(tmp_path / "landcover.tif", "w", driver="GTiff", height=20, width=20, count=1, dtype="uint8",
                       transform=from_origin(0, 20, 1, 1)) as dst:
        dst.write(np.full((20, 20), 4, dtype="uint8"), 1)
    with pytest.raises(importer.ImportError_, match="систем"):
        importer.import_directory(tmp_path, save=False)


def test_without_core_layer_forest_massifs_are_used(demo_dir):
    (demo_dir / "protected_areas.geojson").unlink()
    project = importer.import_directory(demo_dir, save=False, min_core_ha=100)
    assert len(project.cores) >= 2
    assert any("крупнейшие" in w for w in project.meta["warnings"])


def test_zip_slip_is_blocked():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.tif", b"x")
    with pytest.raises(importer.ImportError_, match="недопустимые"):
        importer.import_zip(buffer.getvalue())


def test_not_a_zip():
    with pytest.raises(importer.ImportError_, match="ZIP"):
        importer.import_zip(b"hello")


# ---------------------------------------------------------------- API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "CACHE_DIR", tmp_path / "cache")
    return TestClient(api.app)


def _wait(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/territory/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("import job did not finish")


def test_api_import_analyze_and_exports(client, demo_dir, tmp_path):
    body = zip_dir(demo_dir)
    response = client.post("/api/import?name=Мои%20данные&min_core_ha=100", content=body, headers={"Content-Type": "application/zip"})
    assert response.status_code == 200, response.text
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "done", job
    pid = job["project_id"]

    analysis = client.post("/api/analyze", json={"project_id": pid, "scenario": "base"})
    assert analysis.status_code == 200
    token = analysis.json()["token"]
    assert 0.48 < analysis.json()["fragmentation"]["pc"] < 0.49

    for layer in ("resistance", "cost", "flow", "landcover"):
        tif = client.get(f"/api/export/{token}/{layer}.tif")
        assert tif.status_code == 200, layer
        with rasterio.MemoryFile(tif.content) as memory, memory.open() as src:
            assert (src.height, src.width) == load_demo().shape
            assert src.crs is not None
            data = src.read(1)
            if layer == "resistance":
                assert data.min() >= 1 and np.isfinite(data).all()
            if layer == "landcover":
                assert src.dtypes[0] == "uint8"
    assert client.get(f"/api/export/{token}/nothing.tif").status_code == 404

    gpkg = client.get(f"/api/export/{token}/result.gpkg")
    assert gpkg.status_code == 200
    from pyogrio import list_layers

    path = tmp_path / "out.gpkg"
    path.write_bytes(gpkg.content)
    layers = {row[0] for row in list_layers(path)}
    assert {"cores", "corridors"} <= layers


# ------------------------------------------------- per-class resistance
def test_class_resistance_override_changes_surface_and_token():
    from ecocorridor.resistance import ResistanceParams, build_resistance

    project = load_demo()
    default = build_resistance(project, ResistanceParams())
    forest = next(c for c in project.classes if c.code == 4)
    raised = build_resistance(project, ResistanceParams(class_resistance={4: forest.resistance * 5}))
    mask = project.landcover == 4
    assert mask.any()
    assert raised.values[mask].mean() > default.values[mask].mean() * 2
    # Other classes are untouched.
    other = project.landcover == 5
    assert np.allclose(raised.base[other], default.base[other])
    assert next(c for c in raised.class_summary if c["code"] == 4)["base_resistance"] == forest.resistance * 5


def test_api_class_resistance(client):
    base = client.post("/api/analyze", json={"scenario": "base"}).json()
    changed = client.post("/api/analyze", json={"scenario": "base", "params": {"class_resistance": {"6": 400, "5": 90}}}).json()
    assert changed["token"] != base["token"]
    assert changed["fragmentation"]["pc_cost"] != base["fragmentation"]["pc_cost"] or changed["network"]["stats"] != base["network"]["stats"]
    bad = client.post("/api/analyze", json={"scenario": "base", "params": {"class_resistance": {"42": 5}}})
    assert bad.status_code == 422
    bad = client.post("/api/analyze", json={"scenario": "base", "params": {"class_resistance": {"4": 0.2}}})
    assert bad.status_code == 422


# ---------------------------------------------------- drawn scenario edits
def test_edits_change_surface_and_landcover():
    from ecocorridor.scenarios import Scenario

    project = load_demo()
    left, bottom, right, top = project.bounds
    cx, cy = (left + right) / 2, (bottom + top) / 2
    base = run(project, api.SCENARIOS["demo"]["base"])
    road = [[cx - 8000, cy], [cx + 8000, cy]]
    scenario = Scenario("custom", "c", "", edits=[{"type": "road", "coords": road, "width_m": 100}])
    result = run(project, scenario)
    import geopandas as gpd

    from ecocorridor.scenarios import edit_geometry

    mask = project.rasterize(gpd.GeoDataFrame(geometry=[edit_geometry(scenario.edits[0])], crs=project.crs), all_touched=True)
    assert (result.landcover[mask & (project.landcover != 1)] == 9).all()
    assert (result.resistance[mask & (project.landcover != 1)] >= 400).all()
    assert result.scenario_summary["edited_ha"] > 0
    assert result.token != base.token

    # A crossing over the same road restores passage: resistance capped at 3 but land cover stays road.
    both = Scenario("custom2", "c", "", edits=[
        {"type": "road", "coords": road, "width_m": 100},
        {"type": "crossing", "coords": [[cx, cy - 400], [cx, cy + 400]], "width_m": 200},
    ])
    crossed = run(project, both)
    row, col = (lambda rc: (int(rc[1]), int(rc[0])))(~project.transform * (cx, cy))
    assert crossed.resistance[row, col] <= 3.0 and crossed.landcover[row, col] == 9
    assert crossed.network.stats.get("mean_resistance") is not None


def test_builtup_and_habitat_polygons():
    from ecocorridor.scenarios import Scenario

    project = load_demo()
    left, bottom, right, top = project.bounds
    cx, cy = (left + right) / 2, (bottom + top) / 2
    square = [[cx - 1500, cy - 1500], [cx + 1500, cy - 1500], [cx + 1500, cy + 1500], [cx - 1500, cy + 1500]]
    built = run(project, Scenario("b", "b", "", edits=[{"type": "builtup", "coords": square}]))
    assert (built.landcover == 8).sum() > (project.landcover == 8).sum()
    assert built.habitat.sum() < run(project, api.SCENARIOS["demo"]["base"]).habitat.sum()
    forest = run(project, Scenario("h", "h", "", edits=[{"type": "habitat", "coords": square}]))
    assert forest.habitat.sum() >= run(project, api.SCENARIOS["demo"]["base"]).habitat.sum()


def test_api_edits_validation_and_effect(client):
    left, bottom, right, top = (lambda b: b)(load_demo().bounds)
    cx, cy = (left + right) / 2, (bottom + top) / 2
    ok = client.post("/api/analyze", json={"scenario": "base", "params": {"edits": [
        {"type": "road", "coords": [[cx - 9000, cy], [cx + 9000, cy]], "width_m": 80}]}})
    assert ok.status_code == 200, ok.text
    plain = client.post("/api/analyze", json={"scenario": "base"}).json()
    assert ok.json()["token"] != plain["token"]
    assert ok.json()["scenario_summary"]["edited_ha"] > 0
    too_short = client.post("/api/analyze", json={"scenario": "base", "params": {"edits": [
        {"type": "builtup", "coords": [[0, 0], [1, 1]]}]}})
    assert too_short.status_code == 422
    bad_type = client.post("/api/analyze", json={"scenario": "base", "params": {"edits": [
        {"type": "tunnel", "coords": [[0, 0], [1, 1]]}]}})
    assert bad_type.status_code == 422
