"""Landscape structure and connectivity indices.

All indices are computed on the natural-habitat mask (a set of land-cover
classes) and, where relevant, on the graph of habitat patches.

Reported metrics
----------------
NP, total habitat area, mean/median patch size, largest patch index (LPI),
effective mesh size (MESH, Jaeger 2000), landscape division index, edge density,
number of components and the connectivity indices IIC and PC.

IIC (integral index of connectivity, Pascual-Hortal & Saura 2006):

    IIC = sum_i sum_j (a_i * a_j) / (1 + n_ij) / A^2

with ``a`` patch areas, ``n_ij`` the number of links in the shortest patch-graph
path and ``A`` the total habitat area.

PC (probability of connectivity, Saura & Pascual-Hortal 2006) in its direct
form with an exponential dispersal kernel:

    PC = sum_i sum_j (a_i * a_j * p_ij) / A^2,   p_ij = exp(-d_ij / d0),  p_ii = 1

where ``d_ij`` is the edge-to-edge distance between patches.  The dPC index of a
patch is the relative loss of PC when that patch is removed; it ranks patches by
their contribution to landscape connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree

from .dataset import Project


@dataclass
class FragmentationParams:
    min_patch_ha: float = 5.0
    dispersal_m: float = 5_000.0
    link_threshold_m: float = 2_500.0
    natural_codes: tuple[int, ...] = (2, 3, 4, 5, 7, 10)

    def as_dict(self) -> dict:
        return {
            "min_patch_ha": self.min_patch_ha,
            "dispersal_m": self.dispersal_m,
            "link_threshold_m": self.link_threshold_m,
            "natural_codes": list(self.natural_codes),
        }


def _patch_boundary_points(mask: np.ndarray, labels: np.ndarray, res: float) -> dict[int, np.ndarray]:
    boundary = mask & ~binary_erosion(mask, iterations=1, border_value=0)
    rows, cols = np.nonzero(boundary)
    labs = labels[rows, cols]
    order = np.argsort(labs, kind="stable")
    rows, cols, labs = rows[order], cols[order], labs[order]
    splits = np.nonzero(np.diff(labs))[0] + 1
    out: dict[int, np.ndarray] = {}
    for chunk_rows, chunk_cols, chunk_labs in zip(
        np.split(rows, splits), np.split(cols, splits), np.split(labs, splits)
    ):
        if chunk_labs.size == 0:
            continue
        idx = int(chunk_labs[0])
        points = np.column_stack([chunk_cols * res, chunk_rows * res]).astype(np.float64)
        if len(points) > 1_500:
            step = int(np.ceil(len(points) / 1_500))
            points = points[::step]
        out[idx] = points
    return out


def _pairwise_edge_distances(points: dict[int, np.ndarray], centroids: dict[int, np.ndarray], radius: float) -> dict[tuple[int, int], float]:
    ids = sorted(points)
    if not ids:
        return {}
    centre_arr = np.array([centroids[i] for i in ids])
    tree = cKDTree(centre_arr)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    result: dict[tuple[int, int], float] = {}
    for i_idx, j_idx in pairs:
        i, j = ids[int(i_idx)], ids[int(j_idx)]
        if i == j:
            continue
        tree_i = cKDTree(points[i])
        dist, _ = tree_i.query(points[j], k=1, workers=-1)
        result[(i, j)] = float(dist.min())
    return result


def analyse(
    project: Project,
    habitat_mask: np.ndarray,
    resistance: np.ndarray,
    params: FragmentationParams | None = None,
) -> dict:
    params = params or FragmentationParams()
    res = project.res
    cell_ha = res * res / 10_000.0

    labels, n_labels = label(habitat_mask, structure=np.ones((3, 3)))
    areas_cells = np.bincount(labels.ravel(), minlength=n_labels + 1)
    areas_cells[0] = 0
    areas_ha = areas_cells * cell_ha
    total_ha = float(areas_ha.sum())
    keep = np.flatnonzero(areas_ha >= params.min_patch_ha)
    keep = keep[keep > 0]

    if total_ha <= 0:
        return {"available": False}

    order = keep[np.argsort(areas_ha[keep])[::-1]]
    largest = float(areas_ha[order[0]]) if len(order) else 0.0
    mesh = float(np.sum(areas_ha[keep] ** 2) / total_ha)
    division = float(1.0 - np.sum((areas_ha[keep] / total_ha) ** 2))

    horizontal = np.count_nonzero(habitat_mask[:, 1:] != habitat_mask[:, :-1]) * res
    vertical = np.count_nonzero(habitat_mask[1:, :] != habitat_mask[:-1, :]) * res
    edge_length = float(horizontal + vertical)
    edge_density = edge_length / max(total_ha, 1e-9)  # metres of edge per hectare

    # -- patch graph --------------------------------------------------------
    points = _patch_boundary_points(habitat_mask, labels, res)
    rows = np.repeat(np.arange(project.height), project.width)
    cols = np.tile(np.arange(project.width), project.height)
    flat = labels.ravel()
    sum_rows = np.bincount(flat, weights=rows, minlength=n_labels + 1)
    sum_cols = np.bincount(flat, weights=cols, minlength=n_labels + 1)
    centroids: dict[int, np.ndarray] = {}
    for patch in keep:
        if areas_cells[patch] == 0:
            continue
        centroids[int(patch)] = np.array([sum_cols[patch] / areas_cells[patch] * res, sum_rows[patch] / areas_cells[patch] * res])
    points = {i: p for i, p in points.items() if i in centroids}
    max_radius = params.dispersal_m + 2.0 * np.sqrt(max(areas_ha[keep].max(), 1.0) * 10_000.0 / np.pi)
    pair_dist = _pairwise_edge_distances(points, centroids, max_radius)

    graph = nx.Graph()
    for patch in centroids:
        graph.add_node(int(patch), area=float(areas_ha[patch]))
    for (i, j), distance in pair_dist.items():
        if distance <= params.dispersal_m:
            graph.add_edge(i, j, distance=distance)

    components = list(nx.connected_components(graph)) if graph.number_of_nodes() else []
    largest_component_share = (
        float(max(np.sum([areas_ha[i] for i in comp]) for comp in components) / total_ha) if components else 0.0
    )

    # Hop distances for IIC.
    hop: dict[int, dict[int, int]] = {}
    for node in graph.nodes:
        hop[node] = nx.single_source_shortest_path_length(graph, node, cutoff=6)

    area = areas_ha
    iic_sum = 0.0
    pc_sum = 0.0
    for i in graph.nodes:
        for j, n_hops in hop.get(i, {}).items():
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            distance = pair_dist.get(pair)
            if distance is None:
                continue
            weight_i = float(area[i] * area[j])
            iic_sum += weight_i / (1.0 + n_hops)
            pc_sum += weight_i * np.exp(-distance / params.dispersal_m)
    for i in graph.nodes:
        pc_sum += float(area[i] ** 2)

    iic = float(iic_sum / total_ha**2)
    pc = float(pc_sum / total_ha**2)

    # dPC: relative contribution of each patch (exact removal of its links).
    dpc: dict[int, float] = {}
    if pc > 0:
        for node in graph.nodes:
            removed = float(area[node] ** 2)
            for other in graph.neighbors(node):
                pair = (node, other) if node < other else (other, node)
                distance = pair_dist.get(pair)
                if distance is None:
                    continue
                removed += 2.0 * float(area[node] * area[other]) * np.exp(-distance / params.dispersal_m)
            dpc[node] = float(max(0.0, removed / pc_sum) * 100.0)

    top_patches = []
    for patch in order[:12]:
        top_patches.append(
            {
                "id": int(patch),
                "area_ha": round(float(areas_ha[patch]), 1),
                "dpc": round(dpc.get(int(patch), 0.0), 2),
                "mean_resistance": round(float(resistance[labels == patch].mean()), 1),
            }
        )

    sizes = areas_ha[keep]
    bins = [0, 5, 25, 100, 500, 2_500, 10_000, float(max(sizes.max(), 10_001))]
    hist, _ = np.histogram(sizes, bins=bins)

    return {
        "available": True,
        "patch_count": int(len(keep)),
        "patch_count_all": int(n_labels),
        "total_area_ha": round(total_ha, 1),
        "mean_patch_ha": round(float(sizes.mean()), 1),
        "median_patch_ha": round(float(np.median(sizes)), 1),
        "largest_patch_ha": round(largest, 1),
        "lpi": round(largest / total_ha * 100.0, 2),
        "mesh_ha": round(mesh, 1),
        "division": round(division, 4),
        "edge_density_m_per_ha": round(edge_density, 2),
        "components": len(components),
        "largest_component_share": round(largest_component_share, 4),
        "iic": round(iic, 4),
        "pc": round(pc, 4),
        "dispersal_m": params.dispersal_m,
        "min_patch_ha": params.min_patch_ha,
        "size_histogram": {
            "bins": [round(b, 1) for b in bins],
            "counts": [int(c) for c in hist],
        },
        "top_patches": top_patches,
        "params": params.as_dict(),
    }
