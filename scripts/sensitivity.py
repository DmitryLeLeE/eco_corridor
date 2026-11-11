#!/usr/bin/env python
"""Monte-Carlo sensitivity of EcoCorridor results to the (uncalibrated) resistance values.

Resistance values and weights are expert guesses, so a result is only worth
reporting if it survives plausible changes to them (Beier et al. 2009;
Rayfield et al. 2010).  Each draw multiplies every land-cover class resistance
by exp(N(0, sigma)) and every surface weight by U(1 - w, 1 + w), re-runs the
whole pipeline, and compares with the unperturbed run:

* backbone stability   - how often each link is part of the minimum spanning tree, and the
                         Jaccard similarity of the backbone with the baseline backbone
* corridor location    - intersection-over-union of the strips (one corridor width) around all corridor axes
                         (the cost-weighted paths are what actually react to resistance)
* PC / MESH variation  - reported for completeness: both use straight-line distances between
                         patches, so they are invariant to resistance *by construction*
* scenario ranking     - how often the scenarios keep the baseline order of change in the cost-weighted PC

Usage::

    python scripts/sensitivity.py                       # bundled demo
    python scripts/sensitivity.py --cached t1a2b3c4d5   # a territory from data/cache
    python scripts/sensitivity.py --n 50 --sigma 0.7 --json out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecocorridor import analysis  # noqa: E402
from ecocorridor.analysis import ModelParams  # noqa: E402
from ecocorridor.dataset import CACHE_DIR, Project, load_demo  # noqa: E402
from ecocorridor.scenarios import default_scenarios  # noqa: E402


def links_of(result) -> frozenset[tuple[int, int]]:
    return frozenset((int(l["from"]), int(l["to"])) for l in result.network.links)


def backbone_of(result) -> frozenset[tuple[int, int]]:
    return frozenset((int(l["from"]), int(l["to"])) for l in result.network.links if l["type"] == "backbone")


def corridor_cells(result, project: Project, params: ModelParams) -> np.ndarray:
    """Cells within one corridor width of any corridor axis (axes are one cell wide, so raw overlap is too harsh)."""
    from scipy.ndimage import binary_dilation

    steps = max(1, int(round(params.corridors.corridor_width_m / project.res)))
    return binary_dilation(result.network.flow > 0, iterations=steps)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


def jaccard(a: frozenset, b: frozenset) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def perturbed(project: Project, rng: np.random.Generator, sigma: float, tag: str) -> Project:
    classes = [dataclasses.replace(c, resistance=float(c.resistance * np.exp(rng.normal(0.0, sigma)))) for c in project.classes]
    return dataclasses.replace(project, id=f"{project.id}-{tag}", classes=classes)


def perturbed_params(base: ModelParams, rng: np.random.Generator, spread: float) -> ModelParams:
    r = base.resistance
    jitter = lambda v: float(v * rng.uniform(1.0 - spread, 1.0 + spread))  # noqa: E731
    new = dataclasses.replace(r, slope_weight=jitter(r.slope_weight), road_weight=jitter(r.road_weight), urban_weight=jitter(r.urban_weight))
    return dataclasses.replace(base, resistance=new)


def evaluate(project: Project, params: ModelParams):
    scenarios = {s.id: s for s in default_scenarios(project)}
    runs = {sid: analysis.run(project, s, params) for sid, s in scenarios.items()}
    base_pc = runs["base"].fragmentation["pc"]
    delta = {sid: (r.fragmentation["pc"] - base_pc) / base_pc * 100.0 if base_pc else 0.0 for sid, r in runs.items() if sid != "base"}
    base_cost = runs["base"].fragmentation["pc_cost"]
    delta_cost = {
        sid: (r.fragmentation["pc_cost"] - base_cost) / base_cost * 100.0 if base_cost else 0.0 for sid, r in runs.items() if sid != "base"
    }
    return runs["base"], delta, delta_cost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cached", help="id of a downloaded territory in the cache directory")
    ap.add_argument("--n", type=int, default=30, help="number of Monte-Carlo draws")
    ap.add_argument("--sigma", type=float, default=0.5, help="SD of the log-normal factor on class resistances")
    ap.add_argument("--weight-spread", type=float, default=0.5, help="surface weights are scaled by U(1-s, 1+s)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", help="write the full result to this file")
    args = ap.parse_args()

    project = Project.from_directory(CACHE_DIR / args.cached, project_id=args.cached) if args.cached else load_demo()
    params = ModelParams()
    rng = np.random.default_rng(args.seed)

    base_run, base_delta, base_delta_cost = evaluate(project, params)
    base_links = links_of(base_run)
    base_backbone = backbone_of(base_run)
    base_cells = corridor_cells(base_run, project, params)
    base_pc = base_run.fragmentation["pc"]
    base_pc_cost = base_run.fragmentation["pc_cost"]
    print(f"Проект: {project.name}; базовый PC = {base_pc:.4f}, PC по стоимости = {base_pc_cost:.4f}, связей: {len(base_links)}", file=sys.stderr)

    freq: dict[tuple[int, int], int] = {}
    jac, pcs, pcs_cost, meshes, bb_jac, cell_iou, costs = [], [], [], [], [], [], []
    deltas: dict[str, list[float]] = {k: [] for k in base_delta}
    deltas_cost: dict[str, list[float]] = {k: [] for k in base_delta}
    same_order = 0
    base_order = sorted(base_delta_cost, key=base_delta_cost.get)
    for k in range(args.n):
        proj = perturbed(project, rng, args.sigma, f"s{k}")
        run, delta, delta_cost = evaluate(proj, perturbed_params(params, rng, args.weight_spread))
        found = links_of(run)
        backbone = backbone_of(run)
        for link in backbone:
            freq[link] = freq.get(link, 0) + 1
        jac.append(jaccard(found, base_links))
        bb_jac.append(jaccard(backbone, base_backbone))
        cell_iou.append(iou(corridor_cells(run, proj, params), base_cells))
        costs.append(run.network.stats["network_cost"])
        pcs.append(run.fragmentation["pc"])
        pcs_cost.append(run.fragmentation["pc_cost"])
        meshes.append(run.fragmentation["mesh_ha"])
        for sid, value in delta.items():
            deltas[sid].append(value)
            deltas_cost[sid].append(delta_cost[sid])
        same_order += sorted(delta_cost, key=delta_cost.get) == base_order
        analysis.forget_project(proj.id)
        print(f"  {k + 1}/{args.n}", end="\r", file=sys.stderr)

    def spread(values: list[float]) -> dict:
        m = statistics.fmean(values)
        sd = statistics.pstdev(values)
        return {"mean": m, "sd": sd, "cv": sd / abs(m) if m else None, "min": min(values), "max": max(values)}

    summary = {
        "project": project.name,
        "draws": args.n,
        "sigma": args.sigma,
        "weight_spread": args.weight_spread,
        "baseline": {"pc": base_pc, "pc_cost": base_pc_cost, "delta_pc_cost_pct": base_delta_cost, "links": sorted(base_links), "backbone": sorted(base_backbone), "delta_pc_pct": base_delta},
        "backbone_frequency": {f"{a}-{b}": freq.get((a, b), 0) / args.n for a, b in sorted(set(freq) | set(base_links))},
        "link_set_jaccard": spread(jac),
        "backbone_jaccard": spread(bb_jac),
        "corridor_cells_iou": spread(cell_iou),
        "network_cost": spread(costs),
        "pc": spread(pcs),
        "pc_cost": spread(pcs_cost),
        "mesh_ha": spread(meshes),
        "delta_pc_pct": {sid: spread(v) for sid, v in deltas.items()},
        "delta_pc_cost_pct": {sid: spread(v) for sid, v in deltas_cost.items()},
        "scenario_order_kept": same_order / args.n,
        "baseline_scenario_order": base_order,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n# Чувствительность: {project.name} ({args.n} прогонов, σ={args.sigma}, веса ±{args.weight_spread:.0%})\n")
    print(f"Связей в сети: {len(base_links)} (набор совпал с базовым в среднем на {summary['link_set_jaccard']['mean']:.0%}; "
          f"при малом числе ядер и большом пороге max_link это тривиально — рассматриваются все пары).")
    print(f"Остов (MST): Jaccard с базовым — среднее {summary['backbone_jaccard']['mean']:.2f}, минимум {summary['backbone_jaccard']['min']:.2f}.")
    print(f"Положение коридоров (IoU полос шириной коридора): среднее {summary['corridor_cells_iou']['mean']:.2f}, минимум {summary['corridor_cells_iou']['min']:.2f}.")
    print(f"Стоимость сети: CV {summary['network_cost']['cv']:.0%}.")
    print(f"PC: {summary['pc']['mean']:.4f} ± {summary['pc']['sd']:.4f} (CV {summary['pc']['cv']:.1%}) — не зависит от сопротивления по построению.")
    print(f"PC по стоимости: {summary['pc_cost']['mean']:.4f} ± {summary['pc_cost']['sd']:.4f} (CV {summary['pc_cost']['cv']:.1%}).\n")
    print("| Связь | в остове, доля прогонов | в базовом остове |\n|---|---|---|")
    for key, share in summary["backbone_frequency"].items():
        a, b = (int(x) for x in key.split("-"))
        print(f"| {key} | {share:.0%} | {'да' if (a, b) in base_backbone else 'нет'} |")
    print("\n| Сценарий | ΔPC по стоимости, базовый, % | среднее, % | SD | диапазон |\n|---|---|---|---|---|")
    for sid, s_ in summary["delta_pc_cost_pct"].items():
        print(f"| {sid} | {base_delta_cost[sid]:+.2f} | {s_['mean']:+.2f} | {s_['sd']:.2f} | {s_['min']:+.2f} … {s_['max']:+.2f} |")
    print(f"\nПорядок сценариев по ΔPC (по стоимости) сохранился в {summary['scenario_order_kept']:.0%} прогонов (базовый: {' < '.join(base_order)}).")

if __name__ == "__main__":
    main()
