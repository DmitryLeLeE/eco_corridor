"""Least-cost paths on a raster graph.

The raster is converted into a sparse graph (8 or 16 neighbours).  Moving from
cell *a* to a neighbouring cell *b* costs

    c(a, b) = d(a, b) * mean(R over the cells the step crosses)

where ``d`` is the planar distance between cell centres and ``R`` the
resistance.  For orthogonal steps the crossed cells are *a* and *b*; for
diagonal and knight steps the cells that the segment passes next to are
included as well ("block" averaging).  Plain endpoint averaging lets a path cut
the corner between two diagonal barrier cells and cross a one-cell-wide road at
the price of two forest cells; averaging over the crossed block removes that
leak.  Accumulated cost has units of "metres of ideal habitat equivalent".
Paths are found with Dijkstra's algorithm (``scipy.sparse.csgraph``), which is
exact for this graph.

The 8-neighbour graph over-estimates Euclidean distance by up to ~8 % in
homogeneous terrain (elongation 1.082); the 16-neighbour graph reduces this to
~2.7 % (1.027) at about twice the memory.  Both figures are checked by the tests; the
directional bias of regular raster graphs is discussed by Huber & Church (1985) and
Etherington (2012).
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


# (dy, dx) steps of the 8- and 16-neighbour graphs, listed once per undirected edge.
_MOVES_8 = ((0, 1), (1, 0), (1, 1), (1, -1))
_MOVES_16 = _MOVES_8 + ((1, 2), (2, 1), (1, -2), (2, -1))


def _crossed_offsets(dy: int, dx: int) -> list[tuple[int, int]]:
    """Offsets (relative to the start cell) of the cells a step passes through.

    Endpoints are excluded.  Orthogonal steps cross nothing else; a diagonal
    step touches the two cells that share its corner; a knight step crosses the
    two cells in the middle of the 2x3 block.
    """
    if (dy, dx) in ((0, 1), (1, 0)):
        return []
    if abs(dy) == 1 and abs(dx) == 1:
        return [(0, dx), (dy, 0)]
    if abs(dy) == 1 and abs(dx) == 2:
        mid = dx // 2
        return [(0, mid), (dy, mid)]
    if abs(dy) == 2 and abs(dx) == 1:
        mid = dy // 2
        return [(mid, 0), (mid, dx)]
    raise ValueError(f"unsupported step {(dy, dx)}")


class RasterGraph:
    """Weighted graph over a raster (8 or 16 neighbours), built once and reused."""

    def __init__(self, resistance: np.ndarray, res: float, neighbours: int = 8, corner_mode: str = "block"):
        if neighbours not in (8, 16):
            raise ValueError("neighbours must be 8 or 16")
        if corner_mode not in ("block", "pair"):
            raise ValueError("corner_mode must be 'block' or 'pair'")
        self.shape = resistance.shape
        self.res = float(res)
        self.n = int(resistance.size)
        self.neighbours = neighbours
        self.corner_mode = corner_mode
        self._build(resistance)

    def _build(self, resistance: np.ndarray) -> None:
        h, w = self.shape
        r2 = np.asarray(resistance, dtype=np.float64)
        valid2 = np.isfinite(r2) & (r2 > 0)
        r2z = np.where(valid2, r2, 0.0)
        idx = np.arange(self.n, dtype=np.int32).reshape(h, w)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []

        def window(arr: np.ndarray, dy: int, dx: int, oy: int, ox: int) -> np.ndarray:
            """arr sampled at (start cell + (oy, ox)) for every start cell of the step (dy, dx)."""
            y0, y1 = 0, h - dy
            x0, x1 = (0, w - dx) if dx >= 0 else (-dx, w)
            return arr[y0 + oy : y1 + oy, x0 + ox : x1 + ox]

        for dy, dx in _MOVES_16 if self.neighbours == 16 else _MOVES_8:
            if dy >= h or abs(dx) >= w:
                continue
            step = self.res * math.hypot(dy, dx)
            ok = window(valid2, dy, dx, 0, 0) & window(valid2, dy, dx, dy, dx)
            if not ok.any():
                continue
            total = window(r2z, dy, dx, 0, 0) + window(r2z, dy, dx, dy, dx)
            count = np.full(total.shape, 2.0)
            if self.corner_mode == "block":
                for oy, ox in _crossed_offsets(dy, dx):
                    v = window(valid2, dy, dx, oy, ox)
                    total = total + np.where(v, window(r2z, dy, dx, oy, ox), 0.0)
                    count = count + v
            a = window(idx, dy, dx, 0, 0)[ok]
            b = window(idx, dy, dx, dy, dx)[ok]
            cost = step * (total / count)[ok]
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
