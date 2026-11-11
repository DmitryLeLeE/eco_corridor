"""Naive reference implementations for cross-checking the production code.

Nothing here is used by the application.  Each function recomputes a quantity the fast code
computes, by a deliberately different and slower route, so that an error in one of the two
shows up as a disagreement:

=====================  ===========================  ===================================
quantity               production code              reference
=====================  ===========================  ===================================
p*_ij (PC)             Dijkstra on -ln p (scipy)    Dijkstra on products of p (heapq)
nl_ij (IIC)            scipy BFS on a sparse graph  networkx all-pairs shortest paths
edge-to-edge distance  KD-tree of boundary cells    all cell pairs, exact gap of squares
least-cost distance    vectorised sparse graph      per-edge networkx graph
=====================  ===========================  ===================================

The reference shares the *definitions* with the production code (they are the published
ones: Saura & Pascual-Hortal 2007 for PC and IIC), so agreement shows that the definitions
are implemented as written; it cannot show that Conefor or Graphab would give the same
numbers, because it is not their code.  ``interop`` exports the same graph in Conefor's
input format for that comparison.
"""

from __future__ import annotations

import heapq
import math

import networkx as nx
import numpy as np


def max_product_probabilities(n: int, edges: dict[tuple[int, int], float]) -> np.ndarray:
    """p*_ij: the largest product of direct probabilities over all paths i -> j (1 on the diagonal)."""
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), p in edges.items():
        adjacency[i].append((j, p))
        adjacency[j].append((i, p))
    out = np.zeros((n, n))
    for source in range(n):
        best = [0.0] * n
        best[source] = 1.0
        heap = [(-1.0, source)]
        while heap:
            neg, u = heapq.heappop(heap)
            p = -neg
            if p < best[u]:
                continue
            for v, w in adjacency[u]:
                q = p * w
                if q > best[v]:
                    best[v] = q
                    heapq.heappush(heap, (-q, v))
        out[source] = best
    return out


def pc_reference(
    areas: np.ndarray,
    distances: dict[tuple[int, int], float],
    d0: float,
    landscape_ha: float,
    p_min: float = 0.0,
) -> tuple[float, float]:
    """(PC, ECA in ha) with p_ij = exp(-d_ij / d0); pairs with p < p_min are dropped."""
    edges = {pair: math.exp(-d / d0) for pair, d in distances.items()}
    edges = {pair: p for pair, p in edges.items() if p >= p_min}
    prob = max_product_probabilities(len(areas), edges)
    total = float(areas @ prob @ areas)
    return total / landscape_ha**2, math.sqrt(total)


def iic_reference(areas: np.ndarray, distances: dict[tuple[int, int], float], d0: float, landscape_ha: float) -> float:
    """IIC with a link wherever d_ij <= d0 and nl_ij the number of links on the shortest path."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(areas)))
    graph.add_edges_from(pair for pair, d in distances.items() if d <= d0)
    total = 0.0
    for i, lengths in nx.all_pairs_shortest_path_length(graph):
        for j, hops in lengths.items():
            total += areas[i] * areas[j] / (1.0 + hops)
    return total / landscape_ha**2


def edge_distances_exact(labels: np.ndarray, ids: list[int], res: float, max_dist: float) -> dict[tuple[int, int], float]:
    """Exact distance between the closest points of two patches made of square cells.

    For two cells whose centres are (dr, dc) cells apart the closest points of the squares are
    hypot(max(|dr|-1, 0), max(|dc|-1, 0)) cells apart.  All cell pairs are compared.
    """
    cells = {i: np.argwhere(labels == i) for i in ids}
    out: dict[tuple[int, int], float] = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            pa, pb = cells[ids[a]], cells[ids[b]]
            dr = np.abs(pa[:, None, 0] - pb[None, :, 0])
            dc = np.abs(pa[:, None, 1] - pb[None, :, 1])
            gap = np.hypot(np.maximum(dr - 1, 0), np.maximum(dc - 1, 0)) * res
            best = float(gap.min())
            if best <= max_dist:
                out[(a, b)] = best
    return out


_STEPS = ((0, 1), (1, 0), (1, 1), (1, -1))


def cost_graph_reference(resistance: np.ndarray, res: float) -> nx.Graph:
    """8-neighbour cell graph; a step costs length * mean(R over its end cells and the cells it grazes)."""
    h, w = resistance.shape
    graph = nx.Graph()
    for r in range(h):
        for c in range(w):
            for dr, dc in _STEPS:
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < h and 0 <= c2 < w):
                    continue
                block = [(r, c), (r2, c2)]
                if dr and dc:  # diagonal: the two cells sharing the corner are grazed
                    block += [(r, c2), (r2, c)]
                mean_r = float(np.mean([resistance[x] for x in block]))
                graph.add_edge((r, c), (r2, c2), weight=res * math.hypot(dr, dc) * mean_r)
    return graph


def cost_distances_reference(
    graph: nx.Graph, labels: np.ndarray, ids: list[int], limit: float
) -> dict[tuple[int, int], float]:
    """Least accumulated cost between patches (centre of a cell of one to centre of a cell of the other)."""
    cells = {i: [tuple(x) for x in np.argwhere(labels == i)] for i in ids}
    out: dict[tuple[int, int], float] = {}
    for a, lab in enumerate(ids):
        lengths = nx.multi_source_dijkstra_path_length(graph, cells[lab], cutoff=limit)
        for b in range(a + 1, len(ids)):
            reached = [lengths[c] for c in cells[ids[b]] if c in lengths]
            if reached:
                out[(a, b)] = float(min(reached))
    return out
