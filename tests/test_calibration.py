"""Recovery of known selection from simulated animals, and the plumbing around it."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from ecocorridor import calibration

RES = 100.0
# True relative resistance: forest 1, meadow 3, cropland 8  ->  selection = 1 / resistance.
TRUE_R = {4: 1.0, 5: 3.0, 6: 8.0}


def _landscape(seed: int = 3, n: int = 140) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=(n, n)), 4.0)
    lo, hi = np.quantile(field, [0.35, 0.7])
    lc = np.where(field < lo, 4, np.where(field < hi, 5, 6))
    lc[rng.random((n, n)) < 0.001] = 8  # a handful of built-up cells: too rare to estimate
    return lc


def _cell_xy(project, rows, cols):
    t = project.transform
    return np.column_stack([t.c + (cols + 0.5) * t.a, t.f + (rows + 0.5) * t.e])


def _simulate_points(project, n, seed):
    rng = np.random.default_rng(seed)
    w = np.vectorize(lambda c: 1.0 / TRUE_R.get(int(c), np.inf))(project.landcover)
    p = (w / w.sum()).ravel()
    flat = rng.choice(p.size, size=n, p=p)
    rows, cols = np.divmod(flat, project.width)
    jitter = rng.uniform(-0.5, 0.5, size=(n, 2)) * RES
    return _cell_xy(project, rows, cols) + jitter * np.array([1, -1])


def _simulate_tracks(project, animals, steps, seed, proposals=25):
    rng = np.random.default_rng(seed)
    tracks = []
    for _ in range(animals):
        while True:
            pos = _cell_xy(project, rng.integers(10, project.height - 10), rng.integers(10, project.width - 10))[0]
            if calibration.classes_at(project, pos)[0] in TRUE_R:
                break
        path = [pos]
        for _ in range(steps):
            length = rng.gamma(2.0, 150.0, size=proposals)
            ang = rng.uniform(0, 2 * np.pi, size=proposals)
            cand = pos + np.column_stack([length * np.cos(ang), length * np.sin(ang)])
            code = calibration.classes_at(project, cand)
            w = np.array([1.0 / TRUE_R.get(int(c), np.inf) for c in code])
            if w.sum() == 0:
                break
            pos = cand[rng.choice(proposals, p=w / w.sum())]
            path.append(pos)
        tracks.append(np.array(path))
    return tracks


def _assert_recovered(res: calibration.CalibrationResult):
    best = res.best_class
    assert best == 4
    for code in (5, 6):
        truth = math.log(TRUE_R[code] / TRUE_R[4])  # beta_best - beta_c
        gap = res.beta[best] - res.beta[code]
        lo, hi = res.relative_ci95[code]
        # The 95 % interval of the relative resistance contains the truth ...
        assert lo <= TRUE_R[code] <= hi, (code, lo, TRUE_R[code], hi)
        # ... and the point estimate is within 15 % of it, so the interval is not just wide.
        assert abs(gap - truth) < 0.15


def test_rsf_recovers_known_selection(toy):
    project = toy(_landscape(), RES)
    res = calibration.fit_rsf(project, _simulate_points(project, 4_000, seed=1), seed=2)
    _assert_recovered(res)
    assert res.method == "rsf" and res.relative_resistance[4] == 1.0


def test_ssf_recovers_known_selection(toy):
    project = toy(_landscape(), RES)
    tracks = _simulate_tracks(project, animals=30, steps=120, seed=5)
    res = calibration.fit_ssf(project, tracks, n_random=30, seed=6)
    _assert_recovered(res)
    assert res.method == "ssf"


def test_ssf_cluster_bootstrap_gives_wider_or_comparable_errors(toy):
    project = toy(_landscape(), RES)
    tracks = _simulate_tracks(project, animals=30, steps=60, seed=7)
    model = calibration.fit_ssf(project, tracks, n_random=20, seed=8)
    boot = calibration.fit_ssf(project, tracks, n_random=20, seed=8, n_boot=60)
    assert boot.se_method == "cluster bootstrap" and model.se_method == "model"
    assert boot.beta == model.beta  # same point estimate
    assert all(v > 0 for c, v in boot.beta_se.items() if c != boot.reference_class)


def test_rare_class_keeps_its_prior_and_is_flagged(toy):
    project = toy(_landscape(), RES)
    res = calibration.fit_rsf(project, _simulate_points(project, 3_000, seed=1), seed=2)
    assert 8 in res.flags and "не оценено" in res.flags[8]
    assert res.resistance[8] == project.class_resistance()[8]
    assert 8 not in res.relative_resistance


def test_calibrated_resistance_is_anchored_to_the_prior_of_the_best_class(toy):
    project = toy(_landscape(), RES)
    res = calibration.fit_rsf(project, _simulate_points(project, 3_000, seed=1), seed=2)
    prior = project.class_resistance()
    assert res.resistance[res.best_class] == pytest.approx(prior[res.best_class])
    calibrated = calibration.apply_calibration(project, res)
    assert calibrated.class_resistance()[5] == pytest.approx(res.resistance[5])
    assert project.class_resistance()[5] == prior[5]  # the original project is not modified


def test_too_few_classes_is_an_error(toy):
    project = toy(_landscape(), RES)
    only_forest = _cell_xy(project, *np.nonzero(project.landcover == 4))[:200]
    with pytest.raises(ValueError, match="два класса"):
        calibration.fit_rsf(project, only_forest)


def test_points_outside_the_frame_are_dropped(toy):
    project = toy(_landscape(), RES)
    pts = _simulate_points(project, 2_000, seed=1)
    far = pts + 1e7
    res = calibration.fit_rsf(project, np.vstack([pts, far]), seed=2)
    assert res.n_events <= 2_000


def test_reference_resistance_is_the_habitat_median():
    r = np.array([[2.0, 4.0], [100.0, 6.0]])
    habitat = np.array([[True, True], [False, True]])
    assert calibration.reference_resistance(r, habitat) == 4.0
    with pytest.raises(ValueError):
        calibration.reference_resistance(r, np.zeros_like(habitat))


# -- command-line script ------------------------------------------------------------------------------


def _load_script(name):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _demo_points_csv(tmp_path, n=3_000, seed=0):
    import pandas as pd
    from pyproj import Transformer

    from ecocorridor import dataset

    project = dataset.load_demo()
    rng = np.random.default_rng(seed)
    resist = project.class_resistance()
    w = np.vectorize(lambda c: 1.0 / resist.get(int(c), 1e6))(project.landcover)
    flat = rng.choice(w.size, size=n, p=(w / w.sum()).ravel())
    rows, cols = np.divmod(flat, project.width)
    xy = _cell_xy(project, rows, cols)
    lon, lat = Transformer.from_crs(project.crs, "EPSG:4326", always_xy=True).transform(xy[:, 0], xy[:, 1])
    path = tmp_path / "points.csv"
    pd.DataFrame({"lon": lon, "lat": lat}).to_csv(path, index=False)
    return path


def test_calibrate_script_runs_and_writes_a_loadable_project(tmp_path, monkeypatch, capsys):
    import json

    from ecocorridor import dataset

    monkeypatch.setattr(dataset, "CACHE_DIR", tmp_path / "cache")
    script = _load_script("calibrate")
    csv = _demo_points_csv(tmp_path)
    out = tmp_path / "cal.json"
    code = script.main(["rsf", str(csv), "--demo", "--out", str(out), "--save-project"])
    printed = capsys.readouterr().out
    assert code == 0, printed
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["method"] == "rsf" and result["n_events"] > 1_000
    saved = dataset.Project.from_directory(tmp_path / "cache" / "democal", project_id="democal")
    assert "калибровка" in saved.name
    # the lowest-resistance class of the prior is the most selected one in this simulation
    prior = dataset.load_demo().class_resistance()
    assert prior[result["best_class"]] == min(prior[int(c)] for c in result["relative_resistance"])
    assert "Базовый класс" in printed


def test_calibrate_script_reports_missing_columns(tmp_path):
    import pandas as pd

    script = _load_script("calibrate")
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(SystemExit, match="нет колонок"):
        script.main(["rsf", str(tmp_path / "bad.csv"), "--demo"])


def test_rsf_interval_coverage_over_repeated_simulations(toy):
    """The 95 % intervals should contain the true resistances of both classes in most repeats."""
    project = toy(_landscape(), RES)
    hits = 0
    for k in range(30):
        res = calibration.fit_rsf(project, _simulate_points(project, 1_500, seed=100 + k), seed=k)
        hits += all(res.relative_ci95[c][0] <= TRUE_R[c] <= res.relative_ci95[c][1] for c in (5, 6))
    assert hits >= 23  # 0.95 per class, ~0.90 for two classes at once; 28 of 30 were observed
