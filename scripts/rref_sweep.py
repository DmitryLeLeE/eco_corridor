#!/usr/bin/env python
"""Sensitivity of the cost-weighted PC to the reference resistance R_ref.

R_ref converts accumulated path cost into effective distance (d_eff = cost / R_ref - cell size).  It is not
calibrated, so every use of ``pc_cost`` should be read together with this table.

    python scripts/rref_sweep.py                      # bundled demo
    python scripts/rref_sweep.py --cached ta329e77086 --values 5 10 20 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_project_args, load_project  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_project_args(ap)
    ap.add_argument("--values", type=float, nargs="+", default=[5, 10, 20, 40])
    args = ap.parse_args(argv if argv is not None else (None if len(sys.argv) > 1 else ["--demo"]))

    from ecocorridor import analysis
    from ecocorridor.analysis import ModelParams
    from ecocorridor.scenarios import default_scenarios

    project = load_project(args)
    scenarios = default_scenarios(project)
    ids = [s.id for s in scenarios if s.id != "base"]
    print("| R_ref | PC по стоимости, базовый | " + " | ".join(f"Δ {i}, %" for i in ids) + " |")
    print("|---|---|" + "---|" * len(ids))
    for value in args.values:
        params = ModelParams()
        params.fragmentation.reference_resistance = float(value)
        base = analysis.run(project, scenarios[0], params).fragmentation["pc_cost"]
        deltas = []
        for s in scenarios[1:]:
            v = analysis.run(project, s, params).fragmentation["pc_cost"]
            deltas.append(f"{(v / base - 1) * 100:+.2f}")
        print(f"| {value:g} | {base:.3f} | " + " | ".join(deltas) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
