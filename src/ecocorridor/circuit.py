"""Circuit-theory connectivity (McRae et al. 2008): current flow between cores.

The landscape is an electrical network.  Every cell is a node; neighbouring
cells are joined by a resistor whose value is the mean of the two cell
resistances (8 neighbours; a diagonal step has the conductance of an
orthogonal one divided by sqrt(2), the convention of Circuitscape).  Each core
is shorted into a single node.  For every pair of cores a unit current is
injected into one and withdrawn from the other; the cumulative current through
a cell, summed over all pairs, is high where movement is concentrated or has no
alternative.  Unlike a least-cost path, current spreads over all routes in
proportion to their conductance, so the map also shows *how wide* a connection is.

``effective_resistance[i, j]`` is the voltage between cores *i* and *j* at unit
current: a distance measure that, unlike the least-cost distance, falls when
parallel routes are added (McRae et al. 2008).  Its unit is the resistance of
the surface (a uniform cell resistance *r* gives *r* per step).

The linear system is solved exactly by one sparse LU factorisation shared by all
pairs.  Memory grows faster than the number of cells (measured: 270 thousand cells take ~4 s and ~1 GB, a million
cells ~30 s and ~4 GB), so territories above ``MAX_CELLS`` are refused with a message asking for a coarser resolution.

Not implemented: advanced modes (one-to-all, focal-node grounding with leakage),
barrier-effect or "current-flow betweenness" scaling, and the normalisation of
current by the expected current of a uniform landscape.  Results are verified
against analytic circuits and a dense linear-algebra implementation
(``tests/test_circuit.py``); they have not been compared with Circuitscape itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

MAX_CELLS = 400_000
MAX_CORES = 40
_CHUNK = 24  # pairs solved per batch (bounds the size of the E x m current block)
SQRT2 = math.sqrt(2.0)


class CircuitError(RuntimeError):
    """Raised when the input cannot be solved; the message is meant for the user."""


@dataclass
class CircuitResult:
    current: np.ndarray  # (rows, cols) cumulative current; 0 inside cores
    effective_resistance: np.ndarray  # (n, n); diagonal 0
    pairs: int
    neighbours: int

    def stats(self) -> dict:
        n = len(self.effective_resistance)
        upper = self.effective_resistance[np.triu_indices(n, 1)]
        positive = self.current[self.current > 0]
        return {
            "cores": n,
            "pairs": self.pairs,
            "mean_effective_resistance": round(float(upper.mean()), 2) if upper.size else None,
            "min_effective_resistance": round(float(upper.min()), 2) if upper.size else None,
            "max_current": round(float(self.current.max()), 4),
            "p99_current": round(float(np.percentile(positive, 99)), 4) if positive.size else 0.0,
        }


def _edges(resistance: np.ndarray, neighbours: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected edges (a, b, conductance) of the raster graph."""
    rows, cols = resistance.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    r = resistance.astype(np.float64)
    moves = [(0, 1, 1.0), (1, 0, 1.0)]
    if neighbours == 8:
        moves += [(1, 1, SQRT2), (1, -1, SQRT2)]
    elif neighbours != 4:
        raise CircuitError("Число соседей должно быть 4 или 8.")
    ea, eb, eg = [], [], []
    for dy, dx, length in moves:
        y0, y1 = 0, rows - dy
        x0, x1 = (0, cols - dx) if dx >= 0 else (-dx, cols)
        a = index[y0:y1, x0:x1]
        b = index[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        ra = r[y0:y1, x0:x1]
        rb = r[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        ea.append(a.ravel())
        eb.append(b.ravel())
        eg.append((1.0 / ((ra + rb) / 2.0) / length).ravel())
    return np.concatenate(ea), np.concatenate(eb), np.concatenate(eg)


def solve_circuit(
    resistance: np.ndarray,
    cores: list[np.ndarray],
    neighbours: int = 8,
    max_cells: int = MAX_CELLS,
) -> CircuitResult:
    """Cumulative current and pairwise effective resistance between ``cores`` (boolean masks)."""
    resistance = np.asarray(resistance, dtype=np.float64)
    if resistance.ndim != 2:
        raise CircuitError("Ожидается двумерный растр сопротивления.")
    if not np.all(np.isfinite(resistance)) or resistance.min() <= 0:
        raise CircuitError("Сопротивление должно быть конечным и положительным во всех ячейках.")
    rows, cols = resistance.shape
    n_cells = rows * cols
    if n_cells > max_cells:
        raise CircuitError(
            f"Растр велик для расчёта по току ({n_cells:,} ячеек, предел {max_cells:,}); загрузите территорию с более грубым разрешением.".replace(",", " ")
        )
    masks = [np.asarray(m, dtype=bool) for m in cores if np.asarray(m).any()]
    n = len(masks)
    if n < 2:
        raise CircuitError("Для расчёта по току нужно минимум два ядра.")
    if n > MAX_CORES:
        raise CircuitError(f"Ядер больше {MAX_CORES}: оставьте самые крупные.")
    stacked = np.stack(masks)
    if (stacked.sum(axis=0) > 1).any():
        raise CircuitError("Ядра пересекаются: каждая ячейка может принадлежать только одному ядру.")

    # Node ids: free cells keep a compact index, each core becomes one node after them.
    node = np.full(n_cells, -1, dtype=np.int64)
    in_core = stacked.any(axis=0).ravel()
    free = np.flatnonzero(~in_core)
    node[free] = np.arange(free.size)
    for k, mask in enumerate(masks):
        node[np.flatnonzero(mask.ravel())] = free.size + k
    n_nodes = free.size + n

    ea, eb, g = _edges(resistance, neighbours)
    a, b = node[ea], node[eb]
    keep = a != b
    a, b, g = a[keep], b[keep], g[keep]
    laplacian = coo_matrix(
        (np.concatenate([g, g, -g, -g]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
        shape=(n_nodes, n_nodes),
    ).tocsc()

    # Ground the last core; unknowns are all other nodes.
    ground = n_nodes - 1
    reduced = laplacian[:ground, :ground]
    try:
        lu = splu(csc_matrix(reduced))
    except RuntimeError as exc:  # singular: a part of the raster is cut off from the cores
        raise CircuitError("Сеть несвязна: часть растра не соединена с ядрами.") from exc

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    core_node = np.arange(n) + free.size
    effective = np.zeros((n, n))
    node_current = np.zeros(n_nodes)
    for start in range(0, len(pairs), _CHUNK):
        chunk = pairs[start : start + _CHUNK]
        rhs = np.zeros((ground, len(chunk)))
        for col, (i, j) in enumerate(chunk):
            for sign, core in ((1.0, i), (-1.0, j)):
                if core_node[core] != ground:
                    rhs[core_node[core], col] += sign
        volt = np.zeros((n_nodes, len(chunk)))
        volt[:ground] = lu.solve(rhs)
        for col, (i, j) in enumerate(chunk):
            effective[i, j] = effective[j, i] = volt[core_node[i], col] - volt[core_node[j], col]
        flow = np.abs(g[:, None] * (volt[a] - volt[b]))  # (E, m) current through every edge
        total = flow.sum(axis=1)
        node_current += np.bincount(a, weights=total, minlength=n_nodes) * 0.5
        node_current += np.bincount(b, weights=total, minlength=n_nodes) * 0.5

    current = np.zeros(n_cells)
    current[free] = node_current[: free.size]
    return CircuitResult(
        current=current.reshape(rows, cols).astype(np.float32),
        effective_resistance=effective,
        pairs=len(pairs),
        neighbours=neighbours,
    )
