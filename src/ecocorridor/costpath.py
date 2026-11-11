"""Least-cost paths on a raster graph.

The raster is converted into an 8-connected sparse graph.  Moving from cell *a*
to a neighbouring cell *b* costs

    c(a, b) = d(a, b) * (R_a + R_b) / 2

where ``d`` is the planar distance between cell centres (cell size or its
diagonal) and ``R`` the resistance.  Accumulated cost therefore has units of
"metres of ideal habitat equivalent".  Paths are found with Dijkstra's
algorithm (``scipy.sparse.csgraph``), which is exact for this graph and fast
enough to recompute interactively for a few thousand by a few hundred cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class CostPath:
    cost: float
    cells: np.ndarray  # (n, 2) row/col
    coords: list[tuple[float, float]]
    length_m: float
    mean_resistance: float

    def as_dict(self) -> dict:
        return {
            "cost": round(self.cost, 2),
            "length_m": round(self.length_m, 1),
            "mean_resistance": round(self.mean_resistance, 2),
            "coords": [[round(x, 2), round(y, 2)] for x, y in self.coords],
        }


class RasterGraph:
    """8-connected weighted graph over a raster, built once and reused."""

    def __init__(self, resistance: np.ndarray, res: float):
        self.shape = resistance.shape
        self.res = float(res)
        self.n = int(resistance.size)
        self._build(resistance)

    def _build(self, resistance: np.ndarray) -> None:
        h, w = self.shape
        r = np.asarray(resistance, dtype=np.float64).ravel()
        valid = np.isfinite(r) & (r > 0)
        idx = np.arange(self.n, dtype=np.int32).reshape(h, w)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        diag = self.res * math.sqrt(2.0)

        for dy, dx, step in ((0, 1, self.res), (1, 0, self.res), (1, 1, diag), (1, -1, diag)):
            ys_a, ys_b = slice(0, h - dy), slice(dy, h)
            xs_a = slice(0, w - dx) if dx >= 0 else slice(-dx, w)
            xs_b = slice(dx, w) if dx >= 0 else slice(0, w + dx)

            a = idx[ys_a, xs_a].ravel()
            b = idx[ys_b, xs_b].ravel()
            ok = valid[a] & valid[b]
            if not ok.any():
                continue
            a, b = a[ok], b[ok]
            cost = step * (r[a] + r[b]) / 2.0
            rows.extend([a, b])
            cols.extend([b, a])
            data.extend([cost, cost])

        if not rows:
            raise ValueError("Resistance raster has no traversable cells")
        self.graph = csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n),
        )
        self.graph.sort_indices()

    # -- index helpers ------------------------------------------------------
    def flat(self, row: int, col: int) -> int:
        return int(row) * self.shape[1] + int(col)

    def rowcol(self, flat: int) -> tuple[int, int]:
        return int(flat // self.shape[1]), int(flat % self.shape[1])

    def nearest_cell(self, x: float, y: float, project) -> tuple[int, int]:
        col, row = ~project.transform * (x, y)
        return min(max(int(row), 0), self.shape[0] - 1), min(max(int(col), 0), self.shape[1] - 1)

    def cells_of(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask.ravel()).astype(np.int32)

    # -- shortest paths -----------------------------------------------------
    def distances(self, sources: np.ndarray, return_predecessors: bool = True):
        sources = np.asarray(sources, dtype=np.int32)
        if sources.size == 0:
            raise ValueError("No source cells")
        if sources.size == 1:
            dist, pred = dijkstra(
                self.graph, indices=int(sources[0]), return_predecessors=True, directed=False
            )
            return dist, pred.astype(np.int32)
        if not return_predecessors:
            dist = dijkstra(
                self.graph,
                indices=sources,
                min_only=True,
                return_predecessors=False,
                directed=False,
            )
            return np.asarray(dist), None
        dist, pred, _sources = dijkstra(
            self.graph,
            indices=sources,
            min_only=True,
            return_predecessors=True,
            directed=False,
        )
        return np.asarray(dist), np.asarray(pred, dtype=np.int32)

    def cost_surface(self, sources: np.ndarray) -> np.ndarray:
        dist, _ = self.distances(sources, return_predecessors=False)
        dist = np.asarray(dist, dtype=np.float64)
        return dist.reshape(self.shape)

    def _backtrack(self, pred: np.ndarray, target: int) -> list[int]:
        chain = [target]
        node = target
        for _ in range(self.n):
            nxt = int(pred[node]) if node < pred.size else -9999
            if nxt == -9999 or nxt == node or nxt < 0:
                break
            chain.append(nxt)
            node = nxt
        chain.reverse()
        return chain

    def _path_from_dist(self, dist, pred, target: int, resistance: np.ndarray | None = None) -> CostPath | None:
        total = float(dist[target])
        if not np.isfinite(total):
            return None
        cells = self._backtrack(pred, target)
        arr = np.array([self.rowcol(c) for c in cells], dtype=np.int32)
        return CostPath(cost=total, cells=arr, coords=[], length_m=0.0, mean_resistance=0.0)

    def least_cost(self, project, src_mask: np.ndarray, dst_mask: np.ndarray) -> CostPath | None:
        """Least-cost path between two raster masks (cost measured between edges)."""
        sources = self.cells_of(src_mask)
        targets = self.cells_of(dst_mask)
        if sources.size == 0 or targets.size == 0:
            return None
        dist, pred = self.distances(sources)
        target_dist = dist[targets]
        if not np.isfinite(target_dist).any():
            return None
        target = int(targets[int(np.argmin(target_dist))])
        path = self._path_from_dist(dist, pred, target, None)  # type: ignore[arg-type]
        if path is None:
            return None
        return self._finalise(path, project)

    def least_cost_points(self, project, start: tuple[float, float], end: tuple[float, float]) -> CostPath | None:
        sr, sc = self.nearest_cell(*start, project)
        er, ec = self.nearest_cell(*end, project)
        src = np.array([self.flat(sr, sc)], dtype=np.int32)
        dist, pred = self.distances(src)
        target = self.flat(er, ec)
        path = self._path_from_dist(dist, pred, target, None)  # type: ignore[arg-type]
        if path is None:
            return None
        return self._finalise(path, project)

    def _finalise(self, path: CostPath, project) -> CostPath:
        coords = [project.cell_centre(int(r), int(c)) for r, c in path.cells]
        if len(coords) > 1:
            arr = np.asarray(coords)
            length = float(np.hypot(*np.diff(arr, axis=0).T).sum())
        else:
            length = 0.0
        return CostPath(cost=path.cost, cells=path.cells, coords=coords, length_m=length, mean_resistance=0.0)


def path_resistance_stats(project, resistance: np.ndarray, cells: np.ndarray) -> dict:
    """Summary of the resistance values along a path."""
    if len(cells) == 0:
        return {"mean": 0.0, "max": 0.0, "p90": 0.0}
    values = resistance[cells[:, 0], cells[:, 1]].astype(np.float64)
    return {
        "mean": round(float(values.mean()), 2),
        "max": round(float(values.max()), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
    }
