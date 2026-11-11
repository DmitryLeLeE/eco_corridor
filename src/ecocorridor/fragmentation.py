"""Landscape structure and connectivity indices.

All indices are computed on a habitat mask (a set of land-cover classes) and,
where relevant, on the graph of habitat patches.  Every area-normalised index
uses the *total landscape area* ``A_L`` (habitat + non-habitat) as denominator,
as in the original definitions; normalising by habitat area would make the
values incomparable between scenarios that change the amount of habitat.

Reported metrics
----------------
NP, habitat area and share, mean/median patch size, largest patch index (LPI),
effective mesh size (MESH, Jaeger 2000), landscape division index, edge density,
number of components and the connectivity indices IIC, PC and ECA.

MESH (Jaeger 2000):          MESH = sum_i a_i^2 / A_L
DIVISION:                    D = 1 - sum_i (a_i / A_L)^2
LPI (FRAGSTATS):             LPI = max_i a_i / A_L * 100

IIC (integral index of connectivity, Pascual-Hortal & Saura 2006), binary graph
in which two patches are linked when their edge-to-edge distance is at most the
dispersal distance ``d0``:

    IIC = sum_i sum_j a_i * a_j / (1 + nl_ij) / A_L^2,   nl_ii = 0

where ``nl_ij`` is the number of links on the shortest path of the patch graph
(the double sum runs over all ordered pairs, including i = j and pairs that are
connected only through stepping-stone patches).

PC (probability of connectivity, Saura & Pascual-Hortal 2007), exponential
dispersal kernel ``p_ij = exp(-d_ij / d0)``:

    PC = sum_i sum_j a_i * a_j * p*_ij / A_L^2,   p*_ii = 1

where ``p*_ij`` is the *maximum product probability* over all paths in the patch
graph (direct link or via stepping stones).  Edge-to-edge distances between
patches do not obey the triangle inequality (a large intermediate patch can be
crossed at no cost), so stepping stones genuinely raise ``p*`` above the direct
probability.  ``ECA = A_L * sqrt(PC)`` (equivalent connected area, Saura et al.
2011) expresses PC in hectares.

dPC (Saura & Rubio 2010) of a patch is the relative loss of PC when that patch
is removed from the network, recomputed from scratch (stepping-stone effects
included).

Known simplifications
---------------------
* Distances between patches are Euclidean edge-to-edge distances, not
  cost-weighted distances; the resistance surface enters the corridor model but
  not these indices.
* The landscape is the raster window; patches cut by the window boundary are
  treated as if the boundary were a barrier (the "CUT" procedure of Jaeger 2000,
  see Moser et al. 2007 for the resulting boundary effect).
* The habitat mask is a single, species-agnostic set of classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
from scipy.spatial import cKDTree

from .dataset import Project

# Probabilities below this value are treated as zero when building the PC graph
# (exp(-4.6) = 0.01), which bounds the number of patch pairs that must be examined.
_P_MIN = 0.01
_TOP_DPC = 12


@dataclass
class FragmentationParams:
    min_patch_ha: float = 5.0
    dispersal_m: float = 5_000.0
    natural_codes: tuple[int, ...] = (2, 3, 4, 5, 7, 10)
    max_patches: int = 2_000

    def as_dict(self) -> dict:
        return {
            "min_patch_ha": self.min_patch_ha,
            "dispersal_m": self.dispersal_m,
            "natural_codes": list(self.natural_codes),
            "max_patches": self.max_patches,
        }


HABITAT_PRESETS: dict[str, tuple[int, ...]] = {
    # Everything that is not water, cropland, built-up or road.
    "natural": (2, 3, 4, 5, 7, 10),
    # Forest-dependent species: closed woodland only.
    "forest": (3, 4),
    # Forest plus floodplain wetlands and open woodland.
    "forest_wetland": (2, 3, 4, 7),
}


def _boundary_points(mask: np.ndarray, labels: np.ndarray, res: float) -> dict[int, np.ndarray]:
    """Boundary cell centres (metres) of every labelled patch."""
    boundary = mask & ~binary_erosion(mask, iterations=1, border_value=0)
    rows, cols = np.nonzero(boundary)
    labs = labels[rows, cols]
    order = np.argsort(labs, kind="stable")
    rows, cols, labs = rows[order], cols[order], labs[order]
    splits = np.nonzero(np.diff(labs))[0] + 1
    out: dict[int, np.ndarray] = {}
    for chunk_rows, chunk_cols, chunk_labs in zip(np.split(rows, splits), np.split(cols, splits), np.split(labs, splits)):
        if chunk_labs.size == 0:
            continue
        out[int(chunk_labs[0])] = np.column_stack([(chunk_cols + 0.5) * res, (chunk_rows + 0.5) * res]).astype(np.float64)
    return out


def _edge_distances(
    ids: list[int],
    points: dict[int, np.ndarray],
    centroids: np.ndarray,
    radii: np.ndarray,
    max_dist: float,
    res: float,
) -> dict[tuple[int, int], float]:
    """Exact edge-to-edge distances (positions in ``ids``) for pairs within ``max_dist``."""
    if len(ids) < 2:
        return {}
    tree = cKDTree(centroids)
    reach = max_dist + 2.0 * float(radii.max())
    candidates = tree.query_pairs(reach, output_type="ndarray")
    if candidates.size == 0:
        return {}
    i_pos, j_pos = candidates[:, 0], candidates[:, 1]
    gap = np.hypot(*(centroids[i_pos] - centroids[j_pos]).T)
    keep = gap <= max_dist + radii[i_pos] + radii[j_pos]
    i_pos, j_pos = i_pos[keep], j_pos[keep]

    trees: dict[int, cKDTree] = {}
    result: dict[tuple[int, int], float] = {}
    for a, b in zip(i_pos.tolist(), j_pos.tolist()):
        pts_a, pts_b = points[ids[a]], points[ids[b]]
        if len(pts_a) < len(pts_b):  # query the smaller set against the larger tree
            a, b, pts_a, pts_b = b, a, pts_b, pts_a
        if a not in trees:
            trees[a] = cKDTree(pts_a)
        dist, _ = trees[a].query(pts_b, k=1, distance_upper_bound=max_dist + 2.0 * res)
        best = float(dist.min())
        if np.isfinite(best):
            # Boundary-cell centres are one cell apart across a one-cell gap.
            edge = max(best - res, 0.0)
            if edge <= max_dist:
                result[(min(a, b), max(a, b))] = edge
    return result


def _pc_sum(areas: np.ndarray, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray) -> float:
    """sum_ij a_i a_j p*_ij for a patch graph whose edge weights are -ln p."""
    n = len(areas)
    if n == 0:
        return 0.0
    if len(rows) == 0:
        return float(np.sum(areas**2))
    graph = csr_matrix((np.maximum(weights, 1e-9), (rows, cols)), shape=(n, n))
    dist = dijkstra(graph, directed=False)
    prob = np.exp(-dist)  # exp(-inf) = 0, exp(0) = 1 on the diagonal
    return float(areas @ prob @ areas)


def analyse(
    project: Project,
    habitat_mask: np.ndarray,
    resistance: np.ndarray,
    params: FragmentationParams | None = None,
) -> dict:
    params = params or FragmentationParams()
    res = project.res
    cell_ha = res * res / 10_000.0
    landscape_ha = float(project.height * project.width * cell_ha)
    d0 = float(params.dispersal_m)

    labels, n_labels = label(habitat_mask, structure=np.ones((3, 3)))
    areas_cells = np.bincount(labels.ravel(), minlength=n_labels + 1)
    areas_cells[0] = 0
    areas_ha = areas_cells * cell_ha
    habitat_ha = float(areas_ha.sum())
    if habitat_ha <= 0:
        return {"available": False}

    # -- structure (all habitat patches, FRAGSTATS / Jaeger definitions) -------
    all_patches = np.flatnonzero(areas_ha > 0)
    largest = float(areas_ha.max())
    mesh = float(np.sum(areas_ha**2) / landscape_ha)
    division = float(1.0 - np.sum((areas_ha / landscape_ha) ** 2))
    horizontal = np.count_nonzero(habitat_mask[:, 1:] != habitat_mask[:, :-1]) * res
    vertical = np.count_nonzero(habitat_mask[1:, :] != habitat_mask[:-1, :]) * res
    edge_density = float(horizontal + vertical) / landscape_ha  # metres of edge per hectare of landscape

    keep = all_patches[areas_ha[all_patches] >= params.min_patch_ha]
    order = keep[np.argsort(areas_ha[keep])[::-1]]
    if len(order) > params.max_patches:
        order = order[: params.max_patches]
    nodes = order  # patch labels, largest first
    n = len(nodes)

    iic = pc = eca = 0.0
    components = 0
    largest_component_share = 0.0
    dpc: dict[int, float] = {}

    if n:
        a = areas_ha[nodes].astype(np.float64)
        rows_g = np.repeat(np.arange(project.height), project.width)
        cols_g = np.tile(np.arange(project.width), project.height)
        flat = labels.ravel()
        cnt = np.maximum(areas_cells, 1).astype(np.float64)
        cx = np.bincount(flat, weights=cols_g + 0.5, minlength=n_labels + 1) / cnt * res
        cy = np.bincount(flat, weights=rows_g + 0.5, minlength=n_labels + 1) / cnt * res
        centroids = np.column_stack([cx[nodes], cy[nodes]])

        points_all = _boundary_points(habitat_mask, labels, res)
        ids = [int(x) for x in nodes]
        points = {i: points_all[i] for i in ids if i in points_all}
        ids = [i for i in ids if i in points]
        # Conservative reach of each patch from its centroid: its bounding-box diagonal.
        bbox_diag = np.zeros(n)
        for pos, i in enumerate(ids):
            pts = points[i]
            bbox_diag[pos] = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))) + res

        r_pair = d0 * float(np.log(1.0 / _P_MIN))
        dist_pairs = _edge_distances(ids, points, centroids, bbox_diag, r_pair, res)

        pair_i = np.array([p[0] for p in dist_pairs], dtype=np.int64)
        pair_j = np.array([p[1] for p in dist_pairs], dtype=np.int64)
        pair_d = np.array(list(dist_pairs.values()), dtype=np.float64)

        # PC: -ln p = d / d0 on every pair within the cut-off.
        weights = pair_d / d0
        pc_total = _pc_sum(a, pair_i, pair_j, weights)
        pc = pc_total / landscape_ha**2
        eca = float(np.sqrt(pc_total))

        # IIC: binary links at the dispersal distance, hop counts over all paths.
        linked = pair_d <= d0
        li, lj = pair_i[linked], pair_j[linked]
        adjacency = csr_matrix((np.ones(len(li)), (li, lj)), shape=(n, n))
        hops = shortest_path(adjacency, method="D", directed=False, unweighted=True)
        iic_sum = float(a @ (1.0 / (1.0 + hops)) @ a)  # 1/(1+inf) = 0, diagonal = a_i^2
        iic = iic_sum / landscape_ha**2

        n_comp, comp_labels = connected_components(adjacency, directed=False)
        components = int(n_comp)
        comp_area = np.bincount(comp_labels, weights=a, minlength=n_comp)
        largest_component_share = float(comp_area.max() / habitat_ha)

        # dPC for the largest patches, recomputed without the patch.
        if pc_total > 0:
            for pos in range(min(n, _TOP_DPC)):
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[pos] = False
                remap = np.cumsum(keep_mask) - 1
                edge_ok = keep_mask[pair_i] & keep_mask[pair_j]
                reduced = _pc_sum(a[keep_mask], remap[pair_i[edge_ok]], remap[pair_j[edge_ok]], weights[edge_ok])
                dpc[int(ids[pos])] = float(max(0.0, (pc_total - reduced) / pc_total * 100.0))

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

    sizes = areas_ha[keep] if len(keep) else np.array([0.0])
    bins = [0, 5, 25, 100, 500, 2_500, 10_000, float(max(sizes.max(), 10_001))]
    hist, _ = np.histogram(sizes, bins=bins)

    return {
        "available": True,
        "patch_count": int(len(keep)),
        "patch_count_all": int(len(all_patches)),
        "landscape_area_ha": round(landscape_ha, 1),
        "total_area_ha": round(habitat_ha, 1),
        "habitat_share": round(habitat_ha / landscape_ha, 4),
        "mean_patch_ha": round(float(sizes.mean()), 1),
        "median_patch_ha": round(float(np.median(sizes)), 1),
        "largest_patch_ha": round(largest, 1),
        "lpi": round(largest / landscape_ha * 100.0, 2),
        "mesh_ha": round(mesh, 1),
        "division": round(division, 4),
        "edge_density_m_per_ha": round(edge_density, 2),
        "components": components,
        "largest_component_share": round(largest_component_share, 4),
        "iic": round(iic, 6),
        "pc": round(pc, 6),
        "eca_ha": round(eca, 1),
        "dispersal_m": d0,
        "min_patch_ha": params.min_patch_ha,
        "size_histogram": {
            "bins": [round(b, 1) for b in bins],
            "counts": [int(c) for c in hist],
        },
        "top_patches": top_patches,
        "params": params.as_dict(),
    }
