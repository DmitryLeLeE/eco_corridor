"""Corridor network: least-cost links between cores, flows and pinch points.

Procedure
---------
1. Every pair of core areas closer than ``max_link_km`` is a candidate link.
2. For each core, one multi-source Dijkstra run (all core cells as sources)
   yields the exact least-cost path to every other core.
3. The network is built with a minimum spanning tree over link cost, then
   redundant links are added when they are at most ``detour_ratio`` times more
   expensive than the corresponding tree path.  This separates the backbone
   (necessary for connectivity) from alternative corridors (resilience).
4. Corridor cells accumulate flow (number of links using the cell).  The local
   corridor width is measured with a distance transform; narrow cells carrying
   high flow are reported as pinch points, the places where a corridor is most
   vulnerable to a single intervention.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from shapely.geometry import LineString, Point

from .costpath import RasterGraph
from .dataset import Project


@dataclass
class CorridorParams:
    max_link_km: float = 34.0
    corridor_width_m: float = 250.0
    detour_ratio: float = 1.25
    alternate_links: bool = True
    neighbours: int = 8

    def as_dict(self) -> dict:
        return {
            "max_link_km": self.max_link_km,
            "corridor_width_m": self.corridor_width_m,
            "detour_ratio": self.detour_ratio,
            "alternate_links": self.alternate_links,
            "neighbours": self.neighbours,
        }


@dataclass
class CorridorNetwork:
    nodes: list[dict]
    links: list[dict]
    lines: gpd.GeoDataFrame
    polygons: gpd.GeoDataFrame
    pinchpoints: gpd.GeoDataFrame
    flow: np.ndarray
    stats: dict


def _core_masks(project: Project) -> list[tuple[dict, np.ndarray]]:
    cores = project.cores
    if cores.empty:
        return []
    result: list[tuple[dict, np.ndarray]] = []
    for idx, row in cores.reset_index(drop=True).iterrows():
        mask = project.rasterize(gpd.GeoDataFrame([row], crs=cores.crs, geometry="geometry"))
        if not mask.any():
            continue
        props = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if k != "geometry"}
        props["id"] = int(idx)
        props["name"] = str(props.get("name", f"Ядро {idx + 1}"))
        props["area_ha"] = round(float(row.geometry.area / 10_000.0), 1)
        props["centroid"] = [float(row.geometry.centroid.x), float(row.geometry.centroid.y)]
        result.append((props, mask))
    return result


def build_network(
    project: Project,
    graph: RasterGraph,
    resistance: np.ndarray,
    params: CorridorParams | None = None,
) -> CorridorNetwork:
    params = params or CorridorParams()
    cores = _core_masks(project)
    if len(cores) < 2:
        empty = gpd.GeoDataFrame({"geometry": []}, crs=project.crs)
        return CorridorNetwork([], [], empty, empty, empty, np.zeros(project.shape, dtype=np.float32), {"link_count": 0})

    # Candidate pairs by geometric separation (fast upper bound).
    idx_to_geom = {
        int(idx): geom
        for idx, geom in enumerate(project.cores.reset_index(drop=True).geometry)
        if geom is not None and not geom.is_empty
    }
    ids = [p["id"] for p, _ in cores]
    candidates: list[tuple[int, int]] = []
    for i_pos, i in enumerate(ids):
        for j in ids[i_pos + 1 :]:
            gi, gj = idx_to_geom.get(i), idx_to_geom.get(j)
            if gi is None or gj is None:
                continue
            if gi.distance(gj) <= params.max_link_km * 1000.0:
                candidates.append((i, j))

    masks = {props["id"]: mask for props, mask in cores}
    coords_of: dict[int, dict[int, tuple[float, float]]] = {}
    paths: dict[tuple[int, int], dict] = {}

    for i in ids:
        targets = [(j, masks[j]) for (a, j) in candidates if a == i] + [(a, masks[a]) for (a, j) in candidates if j == i]
        if not targets:
            continue
        dist, pred = graph.distances(graph.cells_of(masks[i]))
        for j, target_mask in targets:
            target_cells = graph.cells_of(target_mask)
            target_dist = dist[target_cells]
            if not np.isfinite(target_dist).any():
                continue
            best = int(target_cells[int(np.argmin(target_dist))])
            cells = np.array([graph.rowcol(c) for c in graph._backtrack(pred, best)], dtype=np.int32)
            coords = [project.cell_centre(int(r), int(c)) for r, c in cells]
            if len(coords) < 2:
                continue
            line = LineString(coords)
            values = resistance[cells[:, 0], cells[:, 1]]
            key = (min(i, j), max(i, j))
            paths[key] = {
                "id": f"{key[0]}-{key[1]}",
                "from": int(i) if i < j else int(j),
                "to": int(j) if i < j else int(i),
                "cost": float(np.min(target_dist)),
                "length_m": float(line.length),
                "length_km": round(float(line.length) / 1000.0, 2),
                "mean_resistance": float(values.mean()),
                "max_resistance": float(values.max()),
                "geometry": line,
                "cells": cells,
            }
            coords_of.setdefault(key[0], {})[key[1]] = coords[0]

    # Backbone: minimum spanning tree over least-cost link costs.
    net = nx.Graph()
    for props, _mask in cores:
        net.add_node(int(props["id"]))
    for key, link in paths.items():
        net.add_edge(key[0], key[1], weight=link["cost"])
    tree = nx.minimum_spanning_tree(net, weight="weight")
    tree_edges = {tuple(sorted(edge)) for edge in tree.edges()}

    selected: list[dict] = []
    for key, link in paths.items():
        is_tree = key in tree_edges
        if is_tree:
            link["type"] = "backbone"
            selected.append(link)
            continue
        if not params.alternate_links:
            continue
        try:
            tree_cost = nx.shortest_path_length(tree, key[0], key[1], weight="weight")
        except nx.NetworkXNoPath:
            continue
        if link["cost"] <= tree_cost * params.detour_ratio:
            link["type"] = "alternative"
            selected.append(link)

    # Flow accumulation over the corridor raster.
    flow = np.zeros(project.shape, dtype=np.float32)
    for link in selected:
        flow[link["cells"][:, 0], link["cells"][:, 1]] += 1.0
    for link in selected:
        cells = link["cells"]
        values = flow[cells[:, 0], cells[:, 1]]
        link["shared_pct"] = round(float((values > 1.0).mean() * 100.0), 1) if cells.size else 0.0

    # Betweenness on the cost network ranks stepping-stone cores.
    betweenness = (
        nx.betweenness_centrality(net, weight="weight") if net.number_of_nodes() > 2 else {n: 0.0 for n in net.nodes}
    )

    lines = gpd.GeoDataFrame(
        [
            {
                "id": link["id"],
                "from_core": link["from"],
                "to_core": link["to"],
                "type": link["type"],
                "cost": round(link["cost"], 1),
                "length_km": round(link["length_m"] / 1000.0, 2),
                "mean_resistance": round(link["mean_resistance"], 1),
                "max_resistance": round(link["max_resistance"], 1),
                "shared_pct": link.get("shared_pct", 0.0),
                "detour_pct": round((link["cost"] / max(_tree_cost(tree, link["from"], link["to"]), 1e-9) - 1.0) * 100.0, 1),
                "geometry": link["geometry"].simplify(60.0),
            }
            for link in selected
        ],
        crs=project.crs,
    )

    polygons = gpd.GeoDataFrame(
        [
            {
                "id": row.id,
                "from_core": row.from_core,
                "to_core": row.to_core,
                "type": row.type,
                "flow": 1,
                "geometry": row.geometry.buffer(params.corridor_width_m / 2.0, cap_style=2, join_style=2),
            }
            for row in lines.itertuples()
        ],
        crs=project.crs,
    )

    pinchpoints = _pinchpoints(project, flow, params)

    total_length = float(lines.geometry.length.sum()) if len(lines) else 0.0
    road_crossings = 0
    if len(lines) and not project.roads.empty:
        road_crossings = _count_crossings(lines, project.roads)

    stats = {
        "link_count": len(selected),
        "backbone_links": sum(1 for link in selected if link["type"] == "backbone"),
        "alternative_links": sum(1 for link in selected if link["type"] == "alternative"),
        "total_length_km": round(total_length / 1000.0, 1),
        "network_cost": round(float(sum(link["cost"] for link in selected)), 1),
        "corridor_area_ha": round(float(polygons.geometry.area.sum() / 10_000.0), 1) if len(polygons) else 0.0,
        "mean_resistance": round(
            float(np.average([link["mean_resistance"] for link in selected], weights=[link["length_m"] for link in selected])), 1
        )
        if selected
        else 0.0,
        "max_resistance": round(float(max((link["max_resistance"] for link in selected), default=0.0)), 1),
        "road_crossings": int(road_crossings),
        "redundancy": round(
            sum(1 for link in selected if link["type"] == "alternative") / max(len(selected), 1), 3
        ),
        "pinchpoint_count": int(len(pinchpoints)),
        "corridor_width_m": params.corridor_width_m,
    }

    nodes = []
    for props, _mask in cores:
        node_id = int(props["id"])
        degree = net.degree(node_id)
        nodes.append(
            {
                **props,
                "degree": int(degree),
                "betweenness": round(float(betweenness.get(node_id, 0.0)), 4),
                "mean_resistance": round(float(resistance[_mask].mean()), 1),
                "in_tree": any(node_id in edge for edge in tree_edges),
            }
        )

    return CorridorNetwork(
        nodes=nodes,
        links=[{k: v for k, v in link.items() if k not in ("cells", "geometry")} for link in selected],
        lines=lines,
        polygons=polygons,
        pinchpoints=pinchpoints,
        flow=flow,
        stats=stats,
    )


def _tree_cost(tree: nx.Graph, a: int, b: int) -> float:
    try:
        return float(nx.shortest_path_length(tree, a, b, weight="weight"))
    except nx.NetworkXNoPath:
        return float("inf")


def _count_crossings(lines: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> int:
    """Number of places where corridor axes cross the road network.

    Roads are dissolved first, so a road split into many OSM ways is counted
    once per crossing, and a corridor running along a road counts once.
    """
    union = roads.geometry.union_all() if hasattr(roads.geometry, "union_all") else roads.geometry.unary_union
    crossings = 0
    for geom in lines.geometry:
        if geom is None or geom.is_empty:
            continue
        inter = geom.intersection(union)
        if inter.is_empty:
            continue
        parts = getattr(inter, "geoms", [inter])
        crossings += len(parts)
    return crossings


def _pinchpoints(project: Project, flow: np.ndarray, params: CorridorParams) -> gpd.GeoDataFrame:
    mask = flow > 0
    if not mask.any():
        return gpd.GeoDataFrame({"geometry": []}, crs=project.crs)
    width = distance_transform_edt(mask, sampling=project.res) * 2.0
    score = flow / np.maximum(width, project.res)
    score = gaussian_filter(score, 1.0)
    local_max = (score == maximum_filter(score, size=9)) & (flow >= 1.0)
    rows, cols = np.nonzero(local_max)
    if rows.size == 0:
        return gpd.GeoDataFrame({"geometry": []}, crs=project.crs)
    values = score[rows, cols]
    order = np.argsort(values)[::-1]
    chosen: list[tuple[int, int, float, float]] = []
    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        x, y = project.cell_centre(r, c)
        if any(np.hypot(x - cx, y - cy) < 1_500.0 for cx, cy, _s, _w in chosen):
            continue
        chosen.append((x, y, float(flow[r, c]), float(width[r, c])))
        if len(chosen) >= 12:
            break
    return gpd.GeoDataFrame(
        [
            {
                "flow": float(f),
                "width_m": round(w, 0),
                "score": round(f / max(w, project.res), 4),
                "lonlat": list(project.to_lonlat(x, y)),
                "geometry": Point(x, y),
            }
            for x, y, f, w in chosen
        ],
        crs=project.crs,
    )
