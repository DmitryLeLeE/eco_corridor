#!/usr/bin/env python
"""Cross-check PC / IIC / ECA against Conefor on one and the same patch graph.

Conefor and Graphab are the reference programs for these indices.  They cannot be run from
the environment this package is developed in, so the comparison is a three-step manual routine
on a Windows machine (see docs/validation.md, section "Сверка с Conefor"):

1. toy case, to confirm the Conefor settings reproduce the closed-form values:

       python scripts/conefor_check.py toy out/toy
       # run Conefor on out/toy/nodes.txt + distances.txt as described in conefor_settings.txt
       python scripts/conefor_check.py compare out/toy/expected.json --pc <PC> --iic <IIC> --pcnum <PCnum>

2. the real landscape (bundled demo or any territory; the scenario selects the habitat mask):

       python scripts/conefor_check.py export out/demo --demo --scenario base
       python scripts/conefor_check.py export out/saratov --query "Saratov" --size-km 30 --res 200

3. enter Conefor's values:

       python scripts/conefor_check.py compare out/demo/expected.json --pc <PC> --iic <IIC> --pcnum <PCnum>

``compare`` exits with 1 when any index differs by more than --tol (relative, default 1e-3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_project_args, load_project  # noqa: E402


def cmd_toy(args) -> int:
    from ecocorridor import interop

    meta = interop.write_toy_case(args.out)
    print(f"Игрушечный пример записан в {args.out}: {meta['nodes']} узла, {meta['connections']} связи.")
    print(json.dumps(meta["expected"], indent=2))
    return 0


def cmd_export(args) -> int:
    from ecocorridor import interop
    from ecocorridor.fragmentation import FragmentationParams, patch_graph
    from ecocorridor.resistance import build_resistance
    from ecocorridor.scenarios import apply_scenario, default_scenarios

    project = load_project(args)
    params = FragmentationParams(min_patch_ha=args.min_patch, dispersal_m=args.dispersal)
    scenario = next((s for s in default_scenarios(project) if s.id == args.scenario), None)
    if scenario is None:
        raise SystemExit(f"неизвестный сценарий {args.scenario!r}")
    surface = build_resistance(project)
    _, _, habitat, _ = apply_scenario(project, surface, scenario, params.natural_codes)
    graph = patch_graph(project, habitat, params)
    if graph is None or not graph.ids:
        raise SystemExit("в проекте нет участков местообитаний нужного размера")
    meta = interop.export_conefor(graph, args.out)
    print(f"Записано в {args.out}: {meta['nodes']} участков, {meta['connections']} связей, "
          f"AL = {meta['landscape_ha']:.1f} га, d0 = {meta['dispersal_m']:g} м")
    print("Ожидаемые значения:", json.dumps(meta["expected"]))
    print(f"Настройки Conefor: {Path(args.out) / 'conefor_settings.txt'}")
    return 0


def cmd_compare(args) -> int:
    from ecocorridor import interop

    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))["expected"]
    eca = args.eca if args.eca is not None else (args.pcnum**0.5 if args.pcnum is not None else None)
    rows = interop.compare(expected, {"pc": args.pc, "iic": args.iic, "eca_ha": eca}, rel_tol=args.tol)
    if not rows:
        raise SystemExit("введите хотя бы одно значение: --pc, --iic, --pcnum или --eca")
    print(f"{'индекс':8s} {'ecocorridor':>14s} {'Conefor':>14s} {'отн. разница':>14s}")
    for r in rows:
        print(f"{r['index']:8s} {r['ecocorridor']:14.6f} {r['conefor']:14.6f} {r['rel_diff']:14.2e}  {'ок' if r['ok'] else 'РАСХОЖДЕНИЕ'}")
    return 0 if all(r["ok"] for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    toy = sub.add_parser("toy", help="записать игрушечный пример с известным ответом")
    toy.add_argument("out")
    toy.set_defaults(func=cmd_toy)

    exp = sub.add_parser("export", help="записать граф участков в формате Conefor")
    exp.add_argument("out")
    add_project_args(exp)
    exp.add_argument("--scenario", default="base", choices=["base", "urban", "road", "restore"])
    exp.add_argument("--dispersal", type=float, default=5_000.0, help="d0, м")
    exp.add_argument("--min-patch", type=float, default=5.0, help="минимальная площадь участка, га")
    exp.set_defaults(func=cmd_export)

    cmp_ = sub.add_parser("compare", help="сравнить значения Conefor с ожидаемыми")
    cmp_.add_argument("expected", help="expected.json из каталога экспорта")
    cmp_.add_argument("--pc", type=float)
    cmp_.add_argument("--iic", type=float)
    cmp_.add_argument("--eca", type=float, help="ECA в га (в Conefor 2.2 нет; см. --pcnum)")
    cmp_.add_argument("--pcnum", type=float, help="PCnum из Conefor; ECA = sqrt(PCnum)")
    cmp_.add_argument("--tol", type=float, default=1e-3)
    cmp_.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
