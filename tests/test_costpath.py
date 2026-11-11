"""Least-cost path graph: analytic cases, corner cutting and grid anisotropy."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecocorridor.costpath import RasterGraph


def _cost(resistance, res, start, end, **kw):
    g = RasterGraph(resistance, res, **kw)
    dist, _ = g.distances(np.array([g.flat(*start)], dtype=np.int32))
    return float(dist[g.flat(*end)])


def test_straight_line_in_uniform_terrain():
    r = np.full((5, 30), 2.0)
    assert math.isclose(_cost(r, 10.0, (2, 0), (2, 20)), 10.0 * 20 * 2.0)


def test_orthogonal_step_averages_the_two_cells():
    r = np.array([[1.0, 5.0]])
    assert math.isclose(_cost(r, 10.0, (0, 0), (0, 1)), 10.0 * (1 + 5) / 2)


def test_diagonal_step_in_uniform_terrain():
    r = np.ones((10, 10))
    assert math.isclose(_cost(r, 1.0, (0, 0), (5, 5)), 5 * math.sqrt(2))


@pytest.mark.parametrize("neighbours", [8, 16])
def test_one_cell_diagonal_barrier_is_not_free_to_cross(neighbours):
    n = 9
    r = np.ones((n, n))
    for i in range(n):
        r[i, i] = 1000.0
    leaky = _cost(r, 100.0, (0, 1), (1, 0), neighbours=neighbours, corner_mode="pair")
    sealed = _cost(r, 100.0, (0, 1), (1, 0), neighbours=neighbours, corner_mode="block")
    assert leaky < 200.0  # endpoint averaging: the barrier cells are simply skipped
    assert sealed > 50 * leaky  # block averaging charges for the crossing


def test_diagonal_crossing_never_cheaper_than_orthogonal_crossing_by_orders_of_magnitude():
    r = np.ones((9, 9))
    r[:, 4] = 1000.0  # vertical one-cell road
    straight = _cost(r, 100.0, (4, 3), (4, 5))
    r2 = np.ones((9, 9))
    for i in range(9):
        r2[i, i] = 1000.0
    diagonal = _cost(r2, 100.0, (3, 5), (5, 3), corner_mode="block")
    assert diagonal > 0.3 * straight


def test_isotropic_elongation_matches_literature_bounds():
    # Maximum elongation of shortest grid paths: 1.082 (8 neighbours), 1.027 (16 neighbours).
    r = np.ones((161, 161))
    worst = {8: 0.0, 16: 0.0}
    for nb in (8, 16):
        g = RasterGraph(r, 1.0, neighbours=nb)
        dist, _ = g.distances(np.array([g.flat(80, 80)], dtype=np.int32))
        for dy in range(1, 70):
            for dx in range(1, 70):
                worst[nb] = max(worst[nb], dist[g.flat(80 + dy, 80 + dx)] / math.hypot(dx, dy))
    assert 1.07 < worst[8] <= 1.0825
    assert 1.02 < worst[16] <= 1.0276


def test_invalid_cells_are_not_traversable():
    r = np.ones((3, 5))
    r[:, 2] = np.nan
    g = RasterGraph(r, 1.0)
    dist, _ = g.distances(np.array([g.flat(1, 0)], dtype=np.int32))
    assert not np.isfinite(dist[g.flat(1, 4)])
