"""Shared helpers of the command-line scripts: choosing and loading a project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def add_project_args(parser: argparse.ArgumentParser) -> None:
    """--demo | --cached ID | --query NAME | --center LON,LAT  (+ --size-km, --res for online loading)."""
    parser.add_argument("--demo", action="store_true", help="встроенный демонстрационный проект")
    parser.add_argument("--cached", metavar="ID", help="территория, уже загруженная в кэш")
    parser.add_argument("--query", help="название места (Nominatim); нужна сеть")
    parser.add_argument("--center", help="lon,lat центра вместо названия")
    parser.add_argument("--size-km", type=float, default=30.0)
    parser.add_argument("--res", type=float, default=200.0, help="размер ячейки, м")


def load_project(args):
    from ecocorridor import dataset, online

    if args.demo:
        return dataset.load_demo()
    if args.cached:
        return dataset.Project.from_directory(dataset.CACHE_DIR / args.cached, project_id=args.cached)
    if args.center:
        lon, lat = (float(v) for v in args.center.split(","))
        request = online.TerritoryRequest(center=(lon, lat), size_km=args.size_km, resolution_m=args.res)
    elif args.query:
        request = online.TerritoryRequest(query=args.query, size_km=args.size_km, resolution_m=args.res)
    else:
        raise SystemExit("укажите --demo, --cached, --query или --center")
    return online.load_territory(request, progress=lambda *_: None)
