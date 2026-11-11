"""Cross-checks of the production indices against the naive reference implementations."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ecocorridor import interop, reference
from ecocorridor.costpath import RasterGraph
from ecocorridor.fragmentation import FragmentationParams, _cost_distances, analyse, patch_graph

RES = 100.0
D0 = 800.0


def _landscape(seed: int, n: int = 36, share: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=(n, n)), 1.6)
    return np.where(field >= np.quantile(field, 1 - share), 4, 6)


def _distances(pg) -> dict[tuple[int, int], float]:
    return {(int(i), int(j)): float(d) for i, j, d in zip(pg.pair_i, pg.pair_j, pg.pair_d)}


# -- the reference itself, against hand arithmetic ---------------------------------------------


def test_reference_two_nodes_by_hand():
    areas = np.array([3.0, 5.0])
    d = {(0, 1): 400.0}
    pc, eca = reference.pc_reference(areas, d, 1_000.0, 100.0)
    p = math.exp(-0.4)
    assert math.isclose(pc, (9 + 25 + 2 * 15 * p) / 100.0**2, rel_tol=1e-12)
    assert math.isclose(eca, math.sqrt(9 + 25 + 30 * p), rel_tol=1e-12)
    # distance 400 <= d0 -> linked, nl = 1
    assert math.isclose(reference.iic_reference(areas, d, 1_000.0, 100.0), (9 + 25 + 2 * 15 / 2) / 100.0**2, rel_tol=1e-12)


def test_reference_stepping_stone_beats_a_poor_direct_link():
    areas = np.array([1.0, 1.0, 1.0])
    edges = {(0, 1): 0.9, (1, 2): 0.9, (0, 2): 0.5}
    prob = reference.max_product_probabilities(3, edges)
    assert math.isclose(prob[0, 2], 0.81)  # 0.9 * 0.9 > 0.5
    assert math.isclose(prob[0, 0], 1.0)


# -- patch graph and distances -------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_edge_distances_equal_exact_cell_gaps(toy, seed):
    lc = _landscape(seed)
    pg = patch_graph(toy(lc, RES), lc == 4, FragmentationParams(min_patch_ha=0.5, dispersal_m=D0))
    exact = reference.edge_distances_exact(pg.labels, pg.ids, RES, pg.r_pair)
    got = _distances(pg)
    assert set(got) == set(exact)
    assert max(abs(got[k] - exact[k]) for k in got) < 1e-6


def test_diagonal_gap_is_not_overestimated(toy):
    # Two single-cell patches whose centres are 3 cells apart in both directions:
    # the cells are hypot(2, 2) = 2.83 cells apart, not 3 * sqrt(2) - 1 = 3.24.
    lc = np.full((10, 10), 6)
    lc[2, 2] = lc[5, 5] = 4
    pg = patch_graph(toy(lc, RES), lc == 4, FragmentationParams(min_patch_ha=0.5, dispersal_m=2_000.0))
    assert math.isclose(float(pg.pair_d[0]), math.hypot(2, 2) * RES, rel_tol=1e-9)


# -- PC, IIC, ECA ---------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_straight_line_indices_match_reference(toy, seed):
    lc = _landscape(seed)
    project = toy(lc, RES)
    params = FragmentationParams(min_patch_ha=0.5, dispersal_m=D0)
    pg = patch_graph(project, lc == 4, params)
    out = analyse(project, lc == 4, np.ones(lc.shape, dtype=np.float32), params)

    d = _distances(pg)
    pc, eca = reference.pc_reference(pg.a, d, D0, pg.landscape_ha, p_min=0.01)
    assert math.isclose(out["pc"], pc, abs_tol=6e-7)  # the API rounds to 6 decimals
    assert math.isclose(out["eca_ha"], eca, abs_tol=0.06)  # ... and to 0.1 ha
    assert math.isclose(out["iic"], reference.iic_reference(pg.a, d, D0, pg.landscape_ha), abs_tol=6e-7)


@pytest.mark.parametrize("seed", range(3))
def test_p_min_cutoff_changes_pc_by_less_than_one_percent(toy, seed):
    """PC drops pairs with p < 0.01 to bound the work; the reference keeps every pair.

    A short dispersal distance is used on purpose: with d0 = 800 m the cut-off (3.7 km) is wider
    than the 3.6 km frame and no pair would be dropped at all.
    """
    d0 = 200.0
    lc = _landscape(seed)
    pg = patch_graph(toy(lc, RES), lc == 4, FragmentationParams(min_patch_ha=0.5, dispersal_m=d0))
    far = reference.edge_distances_exact(pg.labels, pg.ids, RES, 50.0 * d0)
    assert len(far) > len(pg.pair_d)  # some pairs really are beyond the cut-off
    cut, _ = reference.pc_reference(pg.a, _distances(pg), d0, pg.landscape_ha, p_min=0.01)
    full, _ = reference.pc_reference(pg.a, far, d0, pg.landscape_ha, p_min=0.0)
    assert 0.0 <= (full - cut) / full < 0.01


# -- cost-weighted variants ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_cost_distances_and_cost_pc_match_networkx_reference(toy, seed):
    rng = np.random.default_rng(seed)
    lc = _landscape(seed, n=28, share=0.38)
    resistance = (np.where(lc == 4, 5.0, 40.0) * np.exp(gaussian_filter(rng.normal(size=lc.shape), 1.0))).astype(np.float32)
    project = toy(lc, RES)
    params = FragmentationParams(min_patch_ha=0.5, dispersal_m=D0, reference_resistance=10.0)
    pg = patch_graph(project, lc == 4, params)
    graph = RasterGraph(resistance, RES)

    limit = pg.r_pair * params.reference_resistance
    fast = _cost_distances(graph, pg.labels, pg.ids, limit)
    slow = reference.cost_distances_reference(reference.cost_graph_reference(resistance.astype(float), RES), pg.labels, pg.ids, limit)
    assert set(fast) == set(slow)
    assert max(abs(fast[k] - slow[k]) / max(slow[k], 1.0) for k in fast) < 1e-9

    out = analyse(project, lc == 4, resistance, params, graph=graph)
    effective = {k: max(v / params.reference_resistance - RES, 0.0) for k, v in slow.items()}
    pc, eca = reference.pc_reference(pg.a, effective, D0, pg.landscape_ha, p_min=0.01)
    assert math.isclose(out["pc_cost"], pc, abs_tol=6e-7)
    assert math.isclose(out["eca_cost_ha"], eca, abs_tol=0.06)


# -- Conefor export ---------------------------------------------------------------------------------


def test_toy_case_expected_values_by_hand(tmp_path):
    meta = interop.write_toy_case(tmp_path)
    areas, d, d0, al = interop.toy_case()
    p = {k: math.exp(-v / d0) for k, v in d.items()}
    # best paths: 0-1 direct, 1-2 direct, 2-3 direct, 0-2 via 1, 1-3 via 2, 0-3 via 1 and 2 (beats the direct 0.082)
    pstar = {
        (0, 1): p[(0, 1)], (1, 2): p[(1, 2)], (2, 3): p[(2, 3)],
        (0, 2): p[(0, 1)] * p[(1, 2)], (1, 3): p[(1, 2)] * p[(2, 3)],
        (0, 3): max(p[(0, 3)], p[(0, 1)] * p[(1, 2)] * p[(2, 3)]),
    }
    total = float(np.sum(areas**2)) + 2 * sum(areas[i] * areas[j] * v for (i, j), v in pstar.items())
    assert math.isclose(meta["expected"]["pc"], total / al**2, rel_tol=1e-12)
    assert math.isclose(meta["expected"]["eca_ha"], math.sqrt(total), rel_tol=1e-12)
    assert math.isclose(meta["expected"]["eca_ha"], al * math.sqrt(meta["expected"]["pc"]), rel_tol=1e-12)


def test_export_files_are_consistent_with_the_patch_graph(toy, tmp_path):
    lc = _landscape(1)
    project = toy(lc, RES)
    params = FragmentationParams(min_patch_ha=0.5, dispersal_m=D0)
    pg = patch_graph(project, lc == 4, params)
    meta = interop.export_conefor(pg, tmp_path)

    nodes = [line.split("\t") for line in (tmp_path / "nodes.txt").read_text().splitlines()]
    links = [line.split("\t") for line in (tmp_path / "distances.txt").read_text().splitlines()]
    assert len(nodes) == meta["nodes"] == len(pg.ids)
    assert len(links) == meta["connections"] == len(pg.pair_d)
    assert [int(n[0]) for n in nodes] == list(range(1, len(nodes) + 1))
    assert all(int(a) < int(b) for a, b, _ in links)  # every undirected pair once
    out = analyse(project, lc == 4, np.ones(lc.shape, dtype=np.float32), params)
    exp = json.loads((tmp_path / "expected.json").read_text())["expected"]
    assert math.isclose(exp["pc"], out["pc"], abs_tol=6e-7)
    assert math.isclose(exp["iic"], out["iic"], abs_tol=6e-7)
    assert "0.367879" in (tmp_path / "conefor_settings.txt").read_text(encoding="utf-8")


def test_compare_flags_a_mismatch():
    expected = {"pc": 0.1, "iic": 0.08, "eca_ha": 100.0}
    rows = interop.compare(expected, {"pc": 0.1000004, "iic": 0.09}, rel_tol=1e-3)
    assert [r["index"] for r in rows] == ["pc", "iic"]
    assert rows[0]["ok"] and not rows[1]["ok"]
