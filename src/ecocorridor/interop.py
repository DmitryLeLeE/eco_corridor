"""Export of the patch graph for Conefor, and comparison of the numbers it returns.

Conefor (Saura & Torné 2009) and Graphab are the reference programs for PC and IIC.
Neither can be run from the sandbox this package is developed in, so the comparison
is done in two steps that the user performs once on a Windows machine:

1. ``export_conefor`` writes the node file, the connection (distance) file and a
   description of the settings that reproduce this package's definitions;
2. Conefor is run on them by hand and its values are entered into ``compare``.

Before comparing a real landscape, run the toy case (``write_toy_case``): its results are
known in closed form, so a mismatch there means the Conefor settings (units, probability
convention, whether a connection is listed once or twice) differ, not the landscape.

What is compared: the *indices* computed from one and the same set of patch areas and
patch-to-patch distances.  The distances themselves (straight-line edge-to-edge here;
Conefor takes them from the file) are not recomputed by Conefor.

File layout follows the Conefor manual as the author remembers it: a node file with
``node_id  attribute`` per line and a connection file with ``from  to  distance`` per line
(tab-separated, one line per undirected pair).  Check it against the manual of the version
you use; the toy case catches a wrong assumption immediately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .fragmentation import PatchGraph
from .reference import iic_reference, pc_reference

# Probability of a single dispersal step of length d0 under p = exp(-d / d0).
P_AT_D0 = math.exp(-1.0)

_SETTINGS = """\
Настройки Conefor (Sensinode), воспроизводящие определения пакета ecocorridor
=============================================================================

Файлы
  nodes.txt      : номер участка <TAB> площадь, га
  distances.txt  : от <TAB> к <TAB> расстояние «край–край», м (каждая пара один раз)

Вход
  Тип связей              : расстояния (distances), файл distances.txt
  Площадь ландшафта (AL)  : {landscape_ha:.4f} га (площадь всего кадра, не только местообитаний)

Индексы
  PC  (вероятностный)     : расстояние {d0:g} м, вероятность {p:.6f}
                            (это p = exp(−d / {d0:g}); вероятность 0,5 здесь НЕ подходит)
  IIC (бинарный)          : пороговое расстояние {d0:g} м
  ECA получается из PC как √(Σ a_i a_j p*_ij) = AL · √PC

Ожидаемые значения (по независимой эталонной реализации, ecocorridor.reference):
  PC  = {pc:.6f}
  IIC = {iic:.6f}
  ECA = {eca_ha:.2f} га

Расхождение допускается в пределах округления, которое выводит Conefor (обычно 4–6 знаков).
Если на игрушечном примере (toy/) значения не совпали, настройки выше интерпретированы неверно.
"""


def _expected(areas: np.ndarray, distances: dict[tuple[int, int], float], d0: float, landscape_ha: float) -> dict:
    pc, eca = pc_reference(areas, distances, d0, landscape_ha, p_min=0.01)
    return {
        "pc": pc,
        "iic": iic_reference(areas, distances, d0, landscape_ha),
        "eca_ha": eca,
    }


def export_conefor(graph: PatchGraph, out_dir: str | Path) -> dict:
    """Write ``nodes.txt``, ``distances.txt``, ``expected.json`` and ``conefor_settings.txt``."""
    out = Path(out_dir)
    areas = np.asarray(graph.a, dtype=float)
    distances = {(int(i), int(j)): float(d) for i, j, d in zip(graph.pair_i, graph.pair_j, graph.pair_d)}
    d0 = float(graph.dispersal_m)
    return _write(out, areas, distances, d0, graph.landscape_ha)


def _write(out: Path, areas: np.ndarray, distances: dict[tuple[int, int], float], d0: float, landscape_ha: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "nodes.txt").write_text(
        "".join(f"{pos + 1}\t{a:.6f}\n" for pos, a in enumerate(areas)), encoding="utf-8"
    )
    (out / "distances.txt").write_text(
        "".join(f"{i + 1}\t{j + 1}\t{d:.4f}\n" for (i, j), d in sorted(distances.items())), encoding="utf-8"
    )
    expected = _expected(areas, distances, d0, landscape_ha)
    meta = {
        "nodes": int(len(areas)),
        "connections": int(len(distances)),
        "landscape_ha": landscape_ha,
        "dispersal_m": d0,
        "probability_at_d0": P_AT_D0,
        "expected": expected,
    }
    (out / "expected.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "conefor_settings.txt").write_text(
        _SETTINGS.format(landscape_ha=landscape_ha, d0=d0, p=P_AT_D0, **expected), encoding="utf-8"
    )
    return meta


def toy_case() -> tuple[np.ndarray, dict[tuple[int, int], float], float, float]:
    """Four nodes, one stepping-stone path and one direct but poor link; AL = 100 ha (all habitat)."""
    areas = np.array([10.0, 20.0, 30.0, 40.0])
    distances = {(0, 1): 500.0, (1, 2): 800.0, (2, 3): 300.0, (0, 3): 2_500.0}
    return areas, distances, 1_000.0, 100.0


def write_toy_case(out_dir: str | Path) -> dict:
    areas, distances, d0, landscape = toy_case()
    return _write(Path(out_dir), areas, distances, d0, landscape)


def compare(expected: dict, observed: dict, rel_tol: float = 1e-3) -> list[dict]:
    """Relative differences between ``expected`` and values read from Conefor.

    ``observed`` may contain any of ``pc``, ``iic``, ``eca_ha``; missing ones are skipped.
    """
    rows = []
    for key in ("pc", "iic", "eca_ha"):
        if key not in observed or observed[key] is None:
            continue
        ours, theirs = float(expected[key]), float(observed[key])
        rel = abs(theirs - ours) / ours if ours else abs(theirs - ours)
        rows.append({"index": key, "ecocorridor": ours, "conefor": theirs, "rel_diff": rel, "ok": rel <= rel_tol})
    return rows
