"""Closed-form checks of the landscape indices on tiny landscapes (1 cell = 1 ha)."""

from __future__ import annotations

import math

import numpy as np

from ecocorridor.fragmentation import FragmentationParams, analyse

RES = 100.0  # 100 m cells -> 1 ha each


def _run(project, habitat, **kw):
    params = FragmentationParams(min_patch_ha=0.5, **kw)
    return analyse(project, habitat, np.ones(project.shape, dtype=np.float32), params)


def test_mesh_and_division_use_landscape_area(toy):
    # 20 x 20 ha landscape cut by a one-cell barrier into 200 ha + 180 ha.
    lc = np.full((20, 20), 4)
    lc[:, 10] = 9
    p = toy(lc, RES)
    f = _run(p, lc == 4)
    assert f["landscape_area_ha"] == 400.0
    assert math.isclose(f["mesh_ha"], (200**2 + 180**2) / 400, rel_tol=1e-3)  # Jaeger 2000
    assert math.isclose(f["division"], 1 - (0.5**2 + 0.45**2), rel_tol=1e-3)
    assert math.isclose(f["lpi"], 50.0, rel_tol=1e-3)
    assert math.isclose(f["habitat_share"], 0.95, rel_tol=1e-3)


def test_iic_two_patches_includes_intrapatch_term(toy):
    # Two 1-ha patches 3 cells apart inside a 100 ha landscape; d0 large enough to link them.
    lc = np.full((10, 10), 6)
    lc[5, 2] = 4
    lc[5, 6] = 4
    f = _run(toy(lc, RES), lc == 4, dispersal_m=1_000.0)
    # IIC = (a1^2 + a2^2 + 2 * a1 a2 / (1 + 1)) / A_L^2 = 3 / 100^2
    assert math.isclose(f["iic"], 3 / 100**2, abs_tol=1e-6)
    assert f["components"] == 1


def test_iic_counts_stepping_stones_beyond_direct_reach(toy):
    # Three patches on a line, 4 cells between neighbours; the ends are 9 cells apart.
    lc = np.full((3, 21), 6)
    lc[1, 2] = lc[1, 7] = lc[1, 12] = 4
    f = _run(toy(lc, RES), lc == 4, dispersal_m=600.0)
    A = 63.0
    # nl: A-B 1, B-C 1, A-C 2 (via B), diagonal 0
    expected = (3 + 2 * (1 / 2 + 1 / 2 + 1 / 3)) / A**2
    assert math.isclose(f["iic"], expected, abs_tol=1e-6)
    assert f["components"] == 1


def test_pc_uses_maximum_product_probability(toy):
    lc = np.full((3, 21), 6)
    lc[1, 2] = lc[1, 7] = lc[1, 12] = 4
    d0 = 600.0
    f = _run(toy(lc, RES), lc == 4, dispersal_m=d0)
    # edge-to-edge gaps: 4 cells between patches -> 400 m; ends 9 cells apart -> 900 m
    p_ab = math.exp(-400 / d0)
    p_ac_direct = math.exp(-900 / d0)
    p_ac = max(p_ac_direct, p_ab * p_ab)  # stepping stone beats the direct link here
    assert p_ab * p_ab > p_ac_direct
    A = 63.0
    expected = (3 + 2 * (2 * p_ab + p_ac)) / A**2
    assert math.isclose(f["pc"], expected, abs_tol=1e-6)
    assert math.isclose(f["eca_ha"], A * math.sqrt(expected), abs_tol=0.06)


def test_removing_habitat_never_increases_pc_or_iic(toy):
    """With A_L fixed, losing a patch cannot raise connectivity (Saura & Pascual-Hortal)."""
    lc = np.full((20, 40), 6)
    lc[5:15, 2:12] = 4  # large patch (100 ha)
    lc[10, 30] = 4  # small remote patch
    p = toy(lc, RES)
    before = _run(p, lc == 4, dispersal_m=2_000.0)
    lc2 = lc.copy()
    lc2[10, 30] = 6
    after = _run(toy(lc2, RES), lc2 == 4, dispersal_m=2_000.0)
    assert after["pc"] <= before["pc"] + 1e-12
    assert after["iic"] <= before["iic"] + 1e-12
    assert after["mesh_ha"] <= before["mesh_ha"] + 1e-9


def test_dpc_of_bridge_patch_is_large(toy):
    lc = np.full((3, 31), 6)
    lc[:, 1:4] = 4  # 9 ha west
    lc[:, 27:30] = 4  # 9 ha east
    lc[1, 14] = 4  # single-cell stepping stone in the middle
    f = _run(toy(lc, RES), lc == 4, dispersal_m=1_500.0)
    by_area = {round(t["area_ha"], 1): t["dpc"] for t in f["top_patches"]}
    assert by_area[1.0] > 0.0  # the stone carries some of the connectivity
    assert by_area[9.0] > by_area[1.0]


def test_fully_connected_landscape_has_pc_one(toy):
    lc = np.full((10, 10), 4)
    f = _run(toy(lc, RES), lc == 4)
    assert math.isclose(f["pc"], 1.0, rel_tol=1e-6)
    assert math.isclose(f["iic"], 1.0, rel_tol=1e-6)
    assert math.isclose(f["eca_ha"], 100.0, rel_tol=1e-6)


def test_no_habitat_is_reported_unavailable(toy):
    lc = np.full((5, 5), 6)
    assert _run(toy(lc, RES), lc == 4) == {"available": False}
