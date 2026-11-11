"""HTTP API smoke tests on the bundled demo project."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ecocorridor import api

client = TestClient(api.app)


def test_project_and_projects():
    info = client.get("/api/project").json()
    assert info["id"] == "demo" and info["synthetic"] is True
    assert {s["id"] for s in info["scenarios"]} == {"base", "urban", "road", "restore"}
    assert info["defaults"]["neighbours"] == 8 and info["defaults"]["habitat"] == "natural"
    assert client.get("/api/projects").json()["projects"][0]["id"] == "demo"


def test_unknown_project_is_404():
    assert client.get("/api/project", params={"project_id": "nope"}).status_code == 404
    assert client.get("/api/project", params={"project_id": "../x"}).status_code == 404


def test_analyze_fragmentation_is_normalised_by_landscape_area():
    result = client.post("/api/analyze", json={"scenario": "base"}).json()
    frag = result["fragmentation"]
    assert frag["landscape_area_ha"] == pytest.approx(270_000.0)
    assert 0 < frag["pc"] <= 1 and 0 < frag["iic"] <= 1
    assert frag["eca_ha"] == pytest.approx(frag["landscape_area_ha"] * frag["pc"] ** 0.5, rel=0.01)
    assert result["network"]["stats"]["link_count"] >= 1


def test_urban_scenario_never_raises_connectivity():
    base = client.post("/api/analyze", json={"scenario": "base"}).json()["fragmentation"]
    urban = client.post("/api/analyze", json={"scenario": "urban"}).json()["fragmentation"]
    assert urban["pc"] <= base["pc"] + 1e-9
    assert urban["mesh_ha"] <= base["mesh_ha"] + 1e-9
    assert urban["total_area_ha"] < base["total_area_ha"]


def test_forest_preset_reduces_habitat_and_16_neighbours_work():
    nat = client.post("/api/analyze", json={"scenario": "base"}).json()["fragmentation"]
    forest = client.post("/api/analyze", json={"scenario": "base", "params": {"habitat": "forest", "neighbours": 16}}).json()
    assert forest["fragmentation"]["total_area_ha"] < nat["total_area_ha"]
    assert forest["params"]["corridors"]["neighbours"] == 16


def test_layers_route_sample_exports_use_the_result_project():
    result = client.post("/api/analyze", json={"scenario": "base"}).json()
    token = result["token"]
    assert client.get(f"/api/layers/{token}/resistance.png").headers["content-type"] == "image/png"
    left, bottom, right, top = result["meta"]["bounds"]
    sample = client.get(f"/api/sample/{token}", params={"x": (left + right) / 2, "y": (bottom + top) / 2}).json()
    assert sample["resistance"] > 0
    assert client.get(f"/api/export/{token}/corridors.geojson").status_code == 200
    assert "PC" in client.get(f"/api/export/{token}/metrics.csv").text
    assert client.get(f"/api/export/{token}/report.html").status_code == 200


def test_compare_returns_all_scenarios():
    rows = client.post("/api/compare", json={}).json()["rows"]
    assert [r["id"] for r in rows] == ["base", "urban", "road", "restore"]
