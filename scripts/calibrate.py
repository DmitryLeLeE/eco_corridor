#!/usr/bin/env python
"""Estimate land-cover resistance from animal locations (RSF) or tracks (SSF).

    # presence points: a CSV with columns lon,lat   (or x,y with --xy, in the project CRS)
    python scripts/calibrate.py rsf points.csv --cached ta329e77086 --out calibration.json

    # GPS tracks: a CSV with columns id,lon,lat, rows of one animal in time order
    python scripts/calibrate.py ssf tracks.csv --cached ta329e77086 --boot 200 --save-project

The result is a table of relative resistances with 95 % intervals per class.  ``--save-project``
writes a calibrated copy of the territory next to the original in the cache, so that it appears in the
application's list of territories and can be compared with the uncalibrated one.

Read docs/validation.md ("Калибровка сопротивлений") before using the numbers: selection is not cost of
movement, and the slope / road / built-up weights and the reference resistance R_ref are NOT calibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_project_args, load_project  # noqa: E402


def _xy(df, project, use_xy: bool):
    import numpy as np

    from ecocorridor import calibration

    if use_xy:
        return df[["x", "y"]].to_numpy(dtype=float)
    return calibration.lonlat_to_project(project, df["lon"].to_numpy(float), df["lat"].to_numpy(float))


def _print_table(project, res) -> None:
    names = {c.code: c.name for c in project.classes}
    print(f"\nМетод: {res.method.upper()}; событий: {res.n_events}; фоновых: {res.n_background}; "
          f"погрешности: {res.se_method}")
    print(f"Базовый класс (β = 0): {names[res.reference_class]}; самый предпочитаемый: {names[res.best_class]} (сопротивление 1)\n")
    print(f"{'класс':28s} {'набл.':>6s} {'β':>8s} {'±SE':>7s} {'отн. сопр.':>11s} {'95 % интервал':>16s} {'было':>7s} {'стало':>8s}")
    for c in project.classes:
        code = c.code
        n = res.n_used.get(code, 0)
        if code in res.relative_resistance:
            lo, hi = res.relative_ci95[code]
            print(f"{c.name[:28]:28s} {n:6d} {res.beta[code]:8.2f} {res.beta_se[code]:7.2f} {res.relative_resistance[code]:11.2f} "
                  f"{f'{lo:.2f}–{hi:.2f}':>16s} {res.prior[code]:7.1f} {res.resistance[code]:8.1f}")
        elif n:
            print(f"{c.name[:28]:28s} {n:6d} {'—':>8s} {'—':>7s} {'—':>11s} {'—':>16s} {res.prior[code]:7.1f} {res.prior[code]:8.1f}")
    for code, text in res.flags.items():
        if res.n_used.get(code, 0):
            print(f"  ! {names.get(code, code)}: {text}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["rsf", "ssf"])
    ap.add_argument("csv")
    add_project_args(ap)
    ap.add_argument("--xy", action="store_true", help="колонки x,y в системе координат проекта вместо lon,lat")
    ap.add_argument("--min-used", type=int, default=10, help="минимум наблюдений класса для оценки")
    ap.add_argument("--max-ratio", type=float, default=100.0, help="предел контраста сопротивлений")
    ap.add_argument("--buffer-m", type=float, help="RSF: доступны только ячейки в этом радиусе от точек")
    ap.add_argument("--random-steps", type=int, default=20, help="SSF: случайных шагов на наблюдаемый")
    ap.add_argument("--boot", type=int, default=0, help="SSF: число бутстреп-повторов по животным (нужно ≥ 5 животных)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="записать результат в JSON")
    ap.add_argument("--save-project", action="store_true", help="записать откалиброванную копию территории в кэш")
    args = ap.parse_args(argv)

    import pandas as pd

    from ecocorridor import calibration
    from ecocorridor.dataset import CACHE_DIR

    project = load_project(args)
    df = pd.read_csv(args.csv)
    need = ["x", "y"] if args.xy else ["lon", "lat"]
    if args.method == "ssf":
        need = ["id"] + need
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"в {args.csv} нет колонок: {', '.join(missing)} (есть: {', '.join(df.columns)})")

    try:
        if args.method == "rsf":
            res = calibration.fit_rsf(project, _xy(df, project, args.xy), buffer_m=args.buffer_m, seed=args.seed,
                                      min_used=args.min_used, max_ratio=args.max_ratio)
        else:
            tracks = [_xy(g, project, args.xy) for _, g in df.groupby("id", sort=False)]
            res = calibration.fit_ssf(project, tracks, n_random=args.random_steps, seed=args.seed,
                                      min_used=args.min_used, max_ratio=args.max_ratio, n_boot=args.boot)
    except ValueError as exc:
        raise SystemExit(f"Калибровка невозможна: {exc}") from None

    _print_table(project, res)
    if args.out:
        Path(args.out).write_text(json.dumps(res.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nРезультат записан: {args.out}")
    if args.save_project:
        import dataclasses

        calibrated = calibration.apply_calibration(project, res)
        calibrated = dataclasses.replace(calibrated, id=f"{project.id}cal", name=f"{project.name} (калибровка {res.method.upper()})")
        target = CACHE_DIR / calibrated.id
        calibrated.to_directory(target)
        print(f"Откалиброванная копия: {target} (в приложении — в списке территорий под именем «{calibrated.name}»)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
