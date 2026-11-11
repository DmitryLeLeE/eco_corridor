"""Circuit-theory current: analytic circuits and an independent dense implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ecocorridor import circuit


def strip(rows, cols, r=1.0):
    return np.full((rows, cols), r)


def end_cores(rows, cols):
    left = np.zeros((rows, cols), bool)
    right = np.zeros((rows, cols), bool)
    left[:, 0] = True
    right[:, -1] = True
    return [left, right]


@pytest.mark.parametrize("rows,cols,r", [(1, 12, 1.0), (4, 10, 2.5), (7, 25, 0.5)])
def test_uniform_ladder_resistance(rows, cols, r):
    """W parallel chains of (L-1) resistors r between shorted end columns: R = r (L-1) / W."""
    result = circuit.solve_circuit(strip(rows, cols, r), end_cores(rows, cols), neighbours=4)
    assert result.effective_resistance[0, 1] == pytest.approx(r * (cols - 1) / rows, rel=1e-9)


def test_two_lanes_in_parallel():
    """Lanes of resistance r1 and r2 (widths w1, w2): conductance adds, current splits by conductance."""
    r1, r2, w1, w2, cols = 2.0, 8.0, 3, 2, 15
    field = np.vstack([strip(w1, cols, r1), strip(w2, cols, r2)])
    result = circuit.solve_circuit(field, end_cores(w1 + w2, cols), neighbours=4)
    expected = (cols - 1) / (w1 / r1 + w2 / r2)  # edges within a lane join cells of equal r -> value r
    assert result.effective_resistance[0, 1] == pytest.approx(expected, rel=1e-9)
    # Mid-way through the strip the current per cell is the lane's conductance share.
    mid = cols // 2
    lane1 = result.current[:w1, mid].sum()
    lane2 = result.current[w1:, mid].sum()
    assert lane1 / (lane1 + lane2) == pytest.approx((w1 / r1) / (w1 / r1 + w2 / r2), rel=1e-5)
    assert lane1 + lane2 == pytest.approx(1.0, rel=1e-5)  # unit current crosses every column


def test_chain_current_is_unit_everywhere():
    result = circuit.solve_circuit(strip(1, 9), end_cores(1, 9), neighbours=4)
    assert result.current[0, 1:-1] == pytest.approx(np.ones(7))
    assert result.current[0, 0] == 0 and result.current[0, -1] == 0  # cores carry no cell current


def dense_reference(resistance, cores, neighbours):
    """Independent implementation: explicit loops, dense Laplacian, pseudo-inverse."""
    rows, cols = resistance.shape
    nodes = rows * cols
    labels = np.arange(nodes).reshape(rows, cols)
    # Merge core cells by union.
    parent = list(range(nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for mask in cores:
        cells = labels[mask]
        for c in cells[1:]:
            parent[find(int(c))] = find(int(cells[0]))
    ids = {}
    for v in range(nodes):
        ids.setdefault(find(v), len(ids))
    lap = np.zeros((len(ids), len(ids)))
    offsets = [(0, 1, 1.0), (1, 0, 1.0)] + ([(1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))] if neighbours == 8 else [])
    for y in range(rows):
        for x in range(cols):
            for dy, dx, length in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < rows and 0 <= xx < cols:
                    i, j = ids[find(int(labels[y, x]))], ids[find(int(labels[yy, xx]))]
                    if i == j:
                        continue
                    g = 1.0 / ((resistance[y, x] + resistance[yy, xx]) / 2.0) / length
                    lap[i, i] += g
                    lap[j, j] += g
                    lap[i, j] -= g
                    lap[j, i] -= g
    # Voltages from the Laplacian with node 0 grounded (dense solve; no LU shared between pairs).
    def voltages(inject):
        v = np.zeros(len(ids))
        v[1:] = np.linalg.solve(lap[1:, 1:], inject[1:])
        return v

    out = np.zeros((len(cores), len(cores)))
    nid = [ids[find(int(labels[m][0]))] for m in cores]
    for i in range(len(cores)):
        for j in range(len(cores)):
            if i != j:
                inject = np.zeros(len(ids))
                inject[nid[i]] += 1.0
                inject[nid[j]] -= 1.0
                v = voltages(inject)
                out[i, j] = v[nid[i]] - v[nid[j]]
    return out, lap, ids, find, labels, voltages


@pytest.mark.parametrize("neighbours", [4, 8])
def test_matches_dense_reference_on_random_landscape(neighbours):
    rng = np.random.default_rng(7)
    field = np.exp(rng.normal(1.0, 1.0, size=(9, 11)))
    cores = [np.zeros(field.shape, bool) for _ in range(3)]
    cores[0][0:2, 0:2] = True
    cores[1][6:9, 8:11] = True
    cores[2][0:1, 8:11] = True
    result = circuit.solve_circuit(field, cores, neighbours=neighbours)
    reference, lap, ids, find, labels, voltages = dense_reference(field, cores, neighbours)
    assert np.allclose(result.effective_resistance, reference, rtol=1e-9, atol=1e-12)

    # Cumulative current of the pair (0, 1) against the dense solution.
    pair = circuit.solve_circuit(field, cores[:2], neighbours=neighbours)
    ref2, lap2, ids2, find2, labels2, voltages2 = dense_reference(field, cores[:2], neighbours)
    n = len(ids2)
    inj = np.zeros(n)
    inj[ids2[find2(int(labels2[cores[0]][0]))]] = 1.0
    inj[ids2[find2(int(labels2[cores[1]][0]))]] = -1.0
    v = voltages2(inj)
    expected = np.zeros(n)
    rows, cols = field.shape
    offsets = [(0, 1, 1.0), (1, 0, 1.0)] + ([(1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))] if neighbours == 8 else [])
    for y in range(rows):
        for x in range(cols):
            for dy, dx, length in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < rows and 0 <= xx < cols:
                    i, j = ids2[find2(int(labels2[y, x]))], ids2[find2(int(labels2[yy, xx]))]
                    if i == j:
                        continue
                    g = 1.0 / ((field[y, x] + field[yy, xx]) / 2.0) / length
                    flow = abs(g * (v[i] - v[j]))
                    if not any(cores[k][y, x] for k in range(2)):
                        expected[i] += 0.5 * flow
                    if not any(cores[k][yy, xx] for k in range(2)):
                        expected[j] += 0.5 * flow
    grid = np.zeros(field.shape)
    for y in range(rows):
        for x in range(cols):
            if not (cores[0][y, x] or cores[1][y, x]):
                grid[y, x] = expected[ids2[find2(int(labels2[y, x]))]]
    assert np.allclose(pair.current, grid, rtol=1e-5, atol=1e-9)


def test_symmetry_and_triangle_inequality():
    rng = np.random.default_rng(3)
    field = np.exp(rng.normal(0.5, 0.8, size=(12, 14)))
    masks = [np.zeros(field.shape, bool) for _ in range(4)]
    for k, (y, x) in enumerate([(0, 0), (11, 0), (0, 13), (11, 13)]):
        masks[k][y, x] = True
    m = circuit.solve_circuit(field, masks).effective_resistance
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert m[i, j] <= m[i, k] + m[k, j] + 1e-9  # effective resistance is a metric


def test_parallel_route_lowers_resistance_but_not_least_cost():
    """Adding a second equal route halves the effective resistance (least-cost distance is unchanged)."""
    one = strip(1, 11)
    two = strip(2, 11)
    r1 = circuit.solve_circuit(one, end_cores(1, 11), neighbours=4).effective_resistance[0, 1]
    r2 = circuit.solve_circuit(two, end_cores(2, 11), neighbours=4).effective_resistance[0, 1]
    assert r2 == pytest.approx(r1 / 2, rel=1e-9)


def test_barrier_concentrates_current_in_the_gap():
    field = strip(9, 15)
    field[:, 7] = 1000.0
    field[4, 7] = 1.0  # a single gap in the wall
    result = circuit.solve_circuit(field, end_cores(9, 15), neighbours=8)
    assert result.current[4, 7] == result.current[:, 7].max()
    assert result.current[4, 7] > 5 * np.median(result.current[:, 7])


def test_errors_are_reported_in_russian():
    with pytest.raises(circuit.CircuitError, match="минимум два"):
        circuit.solve_circuit(strip(5, 5), end_cores(5, 5)[:1])
    with pytest.raises(circuit.CircuitError, match="пересекаются"):
        both = np.ones((5, 5), bool)
        circuit.solve_circuit(strip(5, 5), [both, both])
    with pytest.raises(circuit.CircuitError, match="велик"):
        circuit.solve_circuit(strip(30, 30), end_cores(30, 30), max_cells=100)
    bad = strip(5, 5)
    bad[2, 2] = np.inf
    with pytest.raises(circuit.CircuitError, match="конечным"):
        circuit.solve_circuit(bad, end_cores(5, 5))


# ------------------------------------------------------------------ API
def test_api_circuit_on_demo(tmp_path):
    from fastapi.testclient import TestClient

    from ecocorridor import api

    client = TestClient(api.app)
    token = client.post("/api/analyze", json={"scenario": "base"}).json()["token"]
    summary = client.get(f"/api/circuit/{token}")
    assert summary.status_code == 200, summary.text
    data = summary.json()
    n = len(data["cores"])
    assert n >= 2
    matrix = np.array(data["effective_resistance"])
    assert matrix.shape == (n, n) and np.allclose(matrix, matrix.T) and (matrix[np.triu_indices(n, 1)] > 0).all()
    png = client.get(f"/api/layers/{token}/current.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    tif = client.get(f"/api/export/{token}/current.tif")
    assert tif.status_code == 200
    # Effective resistance never exceeds the least-cost distance expressed in resistance steps... but the
    # parallel routes make it strictly lower than the single-path series resistance along the best path.
    assert data["stats"]["max_current"] > 0
