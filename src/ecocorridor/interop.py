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

File layout follows the Conefor Sensinode 2.2 user's manual (June 2007, sections 4.2 and 5.3): a node
file with ``node_id  attribute`` per line, a connection file with ``from  to  distance`` per line, tab
separated, every undirected pair once, no headings, point as the decimal separator, and both files
ending with a blank line.  The connection file lists only pairs within reach, so Conefor must be told
that it is a PARTIAL file.  Conefor 2.2 reports PC and the numerator PCnum, not ECA; ECA = sqrt(PCnum).
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
Настройки Conefor Sensinode 2.2, воспроизводящие определения пакета ecocorridor
================================================================================
(по руководству CS22manual.pdf, разделы 4.2, 5.2, 5.3, 6.2.3, 6.3)

Файлы
  nodes.txt      : номер участка <TAB> площадь, га            (Node file)
  distances.txt  : от <TAB> к <TAB> расстояние «край–край», м (Connection file)
  Оба файла заканчиваются пустой строкой, как требует Conefor.

Главный экран Conefor
  Connection file format  : PARTIAL   (в файле только пары в пределах досягаемости; остальные = не связаны)
  Type of connections     : Distances
  Maximum landscape attribute (AL) : {landscape_ha:.4f}
                            (площадь всего кадра в гектарах, не только местообитаний)

Индексы
  PC  (Probabilistic): distance = {d0:g}, probability = {p:.6f}
                            (в пакете p = exp(−d / {d0:g}); вероятность 0,5 здесь НЕ подходит)
                            В «PC more» оставьте CHECK ALL PATHS (по умолчанию).
  IIC (Binary):        distance threshold = {d0:g}   (связь при расстоянии ≤ порога)
  Число в полях distance / probability вводится с десятичным разделителем ваших региональных
  настроек Windows (в русской локали — запятая: 0,367879); тысячные разделители не нужны.

Что читать в результатах (Overall indices)
  PC и IIC — значения индексов; PCnum — числитель ΣΣ aᵢ aⱼ p*ᵢⱼ.
  ECA в Conefor 2.2 нет: ECA = √PCnum (в compare можно передать --pcnum).

Ожидаемые значения (независимая эталонная реализация, ecocorridor.reference):
  PC    = {pc:.6f}
  PCnum = {pcnum:.4f}
  IIC   = {iic:.6f}
  ECA   = {eca_ha:.2f} га

Расхождение допускается в пределах округления вывода Conefor.
Если на игрушечном примере (toy) значения не совпали, настройки выше выполнены неверно.
"""


def _expected(areas: np.ndarray, distances: dict[tuple[int, int], float], d0: float, landscape_ha: float) -> dict:
    pc, eca = pc_reference(areas, distances, d0, landscape_ha, p_min=0.01)
    return {
        "pc": pc,
        "pcnum": eca**2,
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
    # Conefor requires a blank line after the last record (manual, sections 4.2 and 5.3).
    (out / "nodes.txt").write_text(
        "".join(f"{pos + 1}\t{a:.6f}\n" for pos, a in enumerate(areas)) + "\n", encoding="utf-8"
    )
    (out / "distances.txt").write_text(
        "".join(f"{i + 1}\t{j + 1}\t{d:.4f}\n" for (i, j), d in sorted(distances.items())) + "\n", encoding="utf-8"
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

    ``observed`` may contain any of ``pc``, ``pcnum``, ``iic``, ``eca_ha``; missing ones are skipped.
    """
    rows = []
    for key in ("pc", "pcnum", "iic", "eca_ha"):
        if key not in observed or observed[key] is None:
            continue
        ours, theirs = float(expected[key]), float(observed[key])
        rel = abs(theirs - ours) / ours if ours else abs(theirs - ours)
        rows.append({"index": key, "ecocorridor": ours, "conefor": theirs, "rel_diff": rel, "ok": rel <= rel_tol})
    return rows
