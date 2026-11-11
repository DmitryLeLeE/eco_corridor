#!/usr/bin/env python
"""End-to-end check of the online loader against the REAL services.

The automated tests use synthetic tiles and a fake Overpass, so they do not prove that
Nominatim, Overpass, ESA WorldCover and Copernicus DEM are reachable and parse correctly.
Run this once on a machine with internet access:

    python scripts/smoke_online.py                      # Saratov, 30 km frame, 200 m cells
    python scripts/smoke_online.py --query "Kazan" --size-km 40 --res 200

It downloads into a temporary cache, runs the full analysis on every default scenario and
prints a short report.  Exit code 0 = everything worked, 1 = a step failed (the message says which).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default="Saratov", help="place name for Nominatim")
    ap.add_argument("--center", help="lon,lat of the centre instead of a place name (skips Nominatim)")
    ap.add_argument("--size-km", type=float, default=30.0)
    ap.add_argument("--res", type=float, default=200.0, help="cell size, m")
    ap.add_argument("--keep-cache", action="store_true", help="do not delete the temporary cache")
    args = ap.parse_args(argv)

    cache = Path(tempfile.mkdtemp(prefix="ecocorridor_smoke_"))
    import os

    os.environ["ECOCORRIDOR_CACHE"] = str(cache)
    from ecocorridor import analysis, dataset, online
    from ecocorridor.scenarios import default_scenarios

    # The cache location is read at import time; point every module at the temporary one.
    for module in (dataset, online):
        module.CACHE_DIR = cache

    steps: list[tuple[str, str]] = []

    def progress(stage: str, fraction: float, message: str) -> None:
        print(f"  [{fraction:4.0%}] {message}")

    try:
        t0 = time.perf_counter()
        where = args.center or repr(args.query)
        print(f"1/3 Загрузка: {where}, кадр {args.size_km:g} км, ячейка {args.res:g} м")
        if args.center:
            lon, lat = (float(v) for v in args.center.split(","))
            request = online.TerritoryRequest(center=(lon, lat), size_km=args.size_km, resolution_m=args.res)
        else:
            request = online.TerritoryRequest(query=args.query, size_km=args.size_km, resolution_m=args.res)
        project = online.load_territory(request, progress=progress)
        steps.append(("загрузка", f"{time.perf_counter() - t0:.0f} с"))
        print(f"    проект: {project.name}; {project.width} × {project.height} ячеек; "
              f"ядер: {len(project.cores)}; дорог: {len(project.roads)}; рек: {len(project.rivers)}")
        meta = cache.glob("*/meta.json")
        for path in meta:
            print(f"    meta: {path}")
        if len(project.cores) < 2:
            print("    ПРЕДУПРЕЖДЕНИЕ: найдено меньше двух ядер — сеть коридоров не будет построена.")

        print("2/3 Расчёт по сценариям")
        for scenario in default_scenarios(project):
            t1 = time.perf_counter()
            result = analysis.run(project, scenario)
            f = result.fragmentation
            print(f"    {scenario.id:8s} PC={f.get('pc')}  PC(стоим.)={f.get('pc_cost')}  MESH={f.get('mesh_ha')} га  "
                  f"связей={result.network.stats.get('link_count')}  {time.perf_counter() - t1:.1f} с")
            steps.append((f"сценарий {scenario.id}", "ok"))

        print("3/3 Повторная загрузка из кэша (без сети)")
        t2 = time.perf_counter()
        online.load_territory(request, progress=lambda *_: None)
        print(f"    {time.perf_counter() - t2:.1f} с")
    except Exception as exc:  # noqa: BLE001 - report any failure with the step reached
        print(f"\nОШИБКА на шаге после «{steps[-1][0] if steps else 'старт'}»: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep_cache:
            print(f"Кэш сохранён: {cache}")
        else:
            shutil.rmtree(cache, ignore_errors=True)

    print("\nOK: загрузчик и расчёт работают на реальных данных. Проверьте карту визуально в приложении.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
