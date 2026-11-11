"""Calibration of land-cover resistance from animal locations or tracks.

The class resistances shipped with the package are expert guesses.  This module estimates
them from data the user supplies:

``fit_rsf``  presence points against random available locations (a resource selection function),
             fitted by logistic regression on land-cover classes;
``fit_ssf``  GPS tracks (a step selection function, Fortin et al. 2005): every observed step is
             compared with random steps from the same start, fitted by conditional logistic
             regression on the land-cover class at the end of the step.

Both return a coefficient beta_c per class (log relative selection, the reference class has 0).
Resistance is the inverse of selection, r_c = exp(beta_best - beta_c), so that the most
selected class has the lowest resistance and a class selected half as often is twice as hard to
cross.  This is the simplest of the transformations reviewed by Zeller et al. (2012); the review
also lists the limits that apply here: selection of a habitat is not the same as the cost of
moving through it, presence-only points say nothing about movement at all, and SSF coefficients
describe step endpoints, not the path between them (Zeller et al. 2015 compare the approaches).

What is *not* calibrated: the slope, road and built-up weights of the resistance surface, the
reference resistance ``R_ref`` of the cost-weighted PC, and the dispersal distance.  Classes with
too few observations keep their prior value and are flagged.

The estimators are tested on simulated animals whose true selection is known (tests/test_calibration.py);
that shows the code recovers a known answer, not that the answer is right for any real species.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from .dataset import LandCoverClass, Project


@dataclass
class CalibrationResult:
    method: str  # "rsf" or "ssf"
    reference_class: int
    best_class: int  # most selected class: relative resistance 1
    beta: dict[int, float]  # log selection relative to the reference class
    beta_se: dict[int, float]
    relative_resistance: dict[int, float]  # best class = 1, capped at max_ratio
    relative_ci95: dict[int, tuple[float, float]]
    resistance: dict[int, float]  # anchored: the best class keeps its prior resistance
    prior: dict[int, float]
    n_used: dict[int, int]  # observed locations (RSF) or step ends (SSF) per class
    flags: dict[int, str] = field(default_factory=dict)
    n_events: int = 0
    n_background: int = 0
    se_method: str = "model"  # "model" (inverse information) or "cluster bootstrap"

    def as_dict(self) -> dict:
        def r(d, k=4):
            return {int(c): (round(v, k) if isinstance(v, float) else v) for c, v in d.items()}

        return {
            "method": self.method,
            "reference_class": self.reference_class,
            "best_class": self.best_class,
            "beta": r(self.beta),
            "beta_se": r(self.beta_se),
            "relative_resistance": r(self.relative_resistance, 3),
            "relative_ci95": {int(c): [round(lo, 3), round(hi, 3)] for c, (lo, hi) in self.relative_ci95.items()},
            "resistance": r(self.resistance, 2),
            "prior": r(self.prior, 2),
            "n_used": {int(c): int(v) for c, v in self.n_used.items()},
            "flags": {int(c): v for c, v in self.flags.items()},
            "n_events": self.n_events,
            "n_background": self.n_background,
            "se_method": self.se_method,
        }


# -- raster access --------------------------------------------------------------------------------


def classes_at(project: Project, xy: np.ndarray) -> np.ndarray:
    """Land-cover code under each (x, y) in project coordinates; 0 outside the raster."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    t = project.transform
    col = np.floor((xy[:, 0] - t.c) / t.a).astype(np.int64)
    row = np.floor((xy[:, 1] - t.f) / t.e).astype(np.int64)
    inside = (row >= 0) & (row < project.height) & (col >= 0) & (col < project.width)
    out = np.zeros(len(xy), dtype=np.int64)
    out[inside] = project.landcover[row[inside], col[inside]]
    return out


def lonlat_to_project(project: Project, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(n, 2) array of project-CRS coordinates for WGS84 longitudes and latitudes."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", project.crs, always_xy=True).transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([x, y])


# -- estimation -----------------------------------------------------------------------------------


def _logit(X: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton's method; the intercept (column 0) is not penalised."""
    p = X.shape[1]
    pen = np.full(p, ridge)
    pen[0] = 0.0
    beta = np.zeros(p)
    for _ in range(100):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (y - mu) - pen * beta
        info = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(info, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    info = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(pen)
    return beta, np.linalg.inv(info)


def _clogit(X: np.ndarray, valid: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Conditional logistic regression.  X is (strata, candidates, p) with the used one first."""
    s, k, p = X.shape
    beta = np.zeros(p)
    for _ in range(100):
        eta = np.where(valid, X @ beta, -np.inf)
        eta = eta - eta.max(axis=1, keepdims=True)
        w = np.exp(eta)
        w /= w.sum(axis=1, keepdims=True)
        xbar = np.einsum("sk,skp->sp", w, X)
        grad = (X[:, 0, :] - xbar).sum(axis=0) - ridge * beta
        second = np.einsum("sk,skp,skq->pq", w, X, X) - np.einsum("sp,sq->pq", xbar, xbar)
        info = second + ridge * np.eye(p)
        step = np.linalg.solve(info, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.where(valid, X @ beta, -np.inf)
    eta = eta - eta.max(axis=1, keepdims=True)
    w = np.exp(eta)
    w /= w.sum(axis=1, keepdims=True)
    xbar = np.einsum("sk,skp->sp", w, X)
    info = np.einsum("sk,skp,skq->pq", w, X, X) - np.einsum("sp,sq->pq", xbar, xbar) + ridge * np.eye(p)
    return beta, np.linalg.inv(info)


def _dummies(codes: np.ndarray, estimated: list[int], reference: int) -> np.ndarray:
    others = [c for c in estimated if c != reference]
    return np.stack([(codes == c).astype(float) for c in others], axis=-1) if others else np.zeros(codes.shape + (0,))


def _choose_classes(
    project: Project, used_codes: np.ndarray, min_used: int
) -> tuple[list[int], int, dict[int, int], dict[int, str]]:
    counts = {int(c.code): int(np.sum(used_codes == c.code)) for c in project.classes}
    estimated = [c for c, n in counts.items() if n >= min_used]
    flags = {c: f"меньше {min_used} наблюдений: сопротивление не оценено, оставлено прежнее" for c, n in counts.items() if n < min_used}
    if len(estimated) < 2:
        raise ValueError(
            f"нужно хотя бы два класса с {min_used}+ наблюдениями, найдено: "
            + ", ".join(f"{c}: {n}" for c, n in counts.items() if n)
        )
    reference = max(estimated, key=lambda c: counts[c])
    return estimated, reference, counts, flags


def _finish(
    project: Project,
    method: str,
    estimated: list[int],
    reference: int,
    coef: np.ndarray,
    cov: np.ndarray,
    counts: dict[int, int],
    flags: dict[int, str],
    max_ratio: float,
    n_events: int,
    n_background: int,
    draws: np.ndarray | None = None,
) -> CalibrationResult:
    others = [c for c in estimated if c != reference]
    beta = {reference: 0.0, **{c: float(coef[i]) for i, c in enumerate(others)}}
    index = {c: i for i, c in enumerate(others)}
    best = max(estimated, key=lambda c: beta[c])

    if draws is not None:  # bootstrap replicates of the coefficients, columns as ``others``
        full = {reference: np.zeros(len(draws)), **{c: draws[:, i] for i, c in enumerate(others)}}
        se = {c: float(np.std(full[c], ddof=1)) for c in estimated}

        def var_contrast(c: int) -> float:
            return float(np.var(full[best] - full[c], ddof=1)) if c != best else 0.0
    else:  # model-based covariance of the coefficients (the reference class is fixed at 0)
        se = {reference: 0.0, **{c: float(math.sqrt(max(cov[i, i], 0.0))) for i, c in enumerate(others)}}

        def var_contrast(c: int) -> float:
            """Variance of beta_best - beta_c."""
            if c == best:
                return 0.0
            vb = cov[index[best], index[best]] if best in index else 0.0
            vc = cov[index[c], index[c]] if c in index else 0.0
            cb = cov[index[best], index[c]] if best in index and c in index else 0.0
            return float(vb + vc - 2 * cb)

    prior = project.class_resistance()
    cap = math.log(max_ratio)
    rel, ci = {}, {}
    for c in estimated:
        gap = beta[best] - beta[c]  # >= 0
        spread = math.sqrt(max(var_contrast(c), 0.0))
        rel[c] = math.exp(min(gap, cap))
        ci[c] = (math.exp(min(max(gap - 1.96 * spread, 0.0), cap)), math.exp(min(gap + 1.96 * spread, cap)))
        if gap > cap:
            flags[c] = f"контраст ограничен ×{max_ratio:g}"
    anchor = prior[best]
    resistance = dict(prior)
    for c in estimated:
        resistance[c] = round(anchor * rel[c], 3)
    return CalibrationResult(
        method=method, reference_class=reference, best_class=best, beta=beta, beta_se=se,
        relative_resistance=rel, relative_ci95=ci, resistance=resistance, prior=prior,
        n_used=counts, flags=flags, n_events=n_events, n_background=n_background,
    )


def fit_rsf(
    project: Project,
    points_xy: np.ndarray,
    n_available: int | None = None,
    buffer_m: float | None = None,
    seed: int = 0,
    ridge: float = 1e-3,
    min_used: int = 10,
    max_ratio: float = 100.0,
) -> CalibrationResult:
    """Resource selection function from presence points (x, y in the project CRS).

    Available locations are drawn uniformly from the raster, or from the cells within ``buffer_m``
    of any presence point.  Points outside the raster are dropped.
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(points_xy, dtype=float)
    used = classes_at(project, pts)
    inside = used > 0
    pts, used = pts[inside], used[inside]
    if len(used) < 2 * min_used:
        raise ValueError(f"слишком мало точек внутри кадра: {len(used)}")

    allowed = np.ones(project.shape, dtype=bool)
    if buffer_m is not None:
        t = project.transform
        seeds = np.zeros(project.shape, dtype=bool)
        r = np.clip(np.floor((pts[:, 1] - t.f) / t.e).astype(int), 0, project.height - 1)
        c = np.clip(np.floor((pts[:, 0] - t.c) / t.a).astype(int), 0, project.width - 1)
        seeds[r, c] = True
        allowed = distance_transform_edt(~seeds, sampling=project.res) <= buffer_m
    allowed &= project.landcover > 0
    rows, cols = np.nonzero(allowed)
    n_av = n_available or max(10 * len(used), 5_000)
    pick = rng.integers(0, len(rows), size=n_av)
    avail = project.landcover[rows[pick], cols[pick]].astype(np.int64)

    estimated, reference, counts, flags = _choose_classes(project, used, min_used)
    keep_u, keep_a = np.isin(used, estimated), np.isin(avail, estimated)
    codes = np.concatenate([used[keep_u], avail[keep_a]])
    y = np.concatenate([np.ones(keep_u.sum()), np.zeros(keep_a.sum())])
    X = np.column_stack([np.ones(len(codes)), _dummies(codes, estimated, reference)])
    beta, cov = _logit(X, y, ridge)
    return _finish(
        project, "rsf", estimated, reference, beta[1:], cov[1:, 1:], counts, flags, max_ratio,
        n_events=int(keep_u.sum()), n_background=int(keep_a.sum()),
    )


def fit_ssf(
    project: Project,
    tracks: list[np.ndarray],
    n_random: int = 20,
    seed: int = 0,
    ridge: float = 1e-3,
    min_used: int = 10,
    max_ratio: float = 100.0,
    n_boot: int = 0,
) -> CalibrationResult:
    """Step selection function from tracks, each an (n, 2) array of fixes in time order (project CRS).

    Random steps keep the start of the observed step; their lengths are drawn from the pooled observed
    step lengths and their directions uniformly (no turning-angle persistence is modelled).  With
    ``n_boot`` > 0 and at least five tracks, standard errors come from a bootstrap that resamples whole tracks,
    which is the honest choice when consecutive steps of one animal are not independent.
    """
    rng = np.random.default_rng(seed)
    starts, ends, owner = [], [], []
    for k, track in enumerate(tracks):
        track = np.asarray(track, dtype=float)
        if len(track) < 2:
            continue
        starts.append(track[:-1])
        ends.append(track[1:])
        owner.append(np.full(len(track) - 1, k))
    if not starts:
        raise ValueError("нет ни одного шага: нужны треки минимум из двух точек")
    start, end, owner_arr = np.vstack(starts), np.vstack(ends), np.concatenate(owner)
    lengths = np.hypot(*(end - start).T)

    used_code = classes_at(project, end)
    start_ok = classes_at(project, start) > 0
    ok = (used_code > 0) & start_ok & (lengths > 0)
    start, end, owner_arr, lengths, used_code = start[ok], end[ok], owner_arr[ok], lengths[ok], used_code[ok]
    if len(lengths) < 2 * min_used:
        raise ValueError(f"слишком мало шагов внутри кадра: {len(lengths)}")

    s = len(lengths)
    length_r = rng.choice(lengths, size=(s, n_random))
    angle = rng.uniform(0.0, 2 * np.pi, size=(s, n_random))
    cand_xy = np.stack([start[:, 0, None] + length_r * np.cos(angle), start[:, 1, None] + length_r * np.sin(angle)], axis=-1)
    cand_code = classes_at(project, cand_xy.reshape(-1, 2)).reshape(s, n_random)
    codes = np.concatenate([used_code[:, None], cand_code], axis=1)  # (s, 1 + n_random), used first

    estimated, reference, counts, flags = _choose_classes(project, used_code, min_used)
    valid = np.isin(codes, estimated)
    keep = valid[:, 0]  # a step that ends in a class that is not estimated carries no information here
    codes, valid, owner_arr = codes[keep], valid[keep], owner_arr[keep]
    X = _dummies(codes, estimated, reference)
    beta, cov = _clogit(X, valid, ridge)

    draws = None
    if n_boot > 0 and len(np.unique(owner_arr)) >= 5:
        animals = np.unique(owner_arr)
        draws_list = []
        for _ in range(n_boot):
            chosen = rng.choice(animals, size=len(animals), replace=True)
            idx = np.concatenate([np.flatnonzero(owner_arr == a) for a in chosen])
            try:
                b, _ = _clogit(X[idx], valid[idx], ridge)
            except np.linalg.LinAlgError:
                continue
            draws_list.append(b)
        if len(draws_list) >= max(10, n_boot // 2):
            draws = np.array(draws_list)

    result = _finish(
        project, "ssf", estimated, reference, beta, cov, counts, flags, max_ratio,
        n_events=int(len(codes)), n_background=int(valid[:, 1:].sum()), draws=draws,
    )
    if draws is not None:
        result.se_method = "cluster bootstrap"
    return result


# -- using the result -------------------------------------------------------------------------------


def apply_calibration(project: Project, result: CalibrationResult) -> Project:
    """A copy of ``project`` whose class resistances are the calibrated ones."""
    classes = [
        LandCoverClass(c.code, c.name, c.color, float(result.resistance.get(c.code, c.resistance)))
        for c in project.classes
    ]
    return dataclasses.replace(project, classes=classes)


def reference_resistance(resistance: np.ndarray, habitat_mask: np.ndarray) -> float:
    """Median resistance of habitat cells: a defensible value for ``FragmentationParams.reference_resistance``.

    The dispersal distance d0 of PC is normally understood as a distance covered through habitat.
    Setting R_ref to the habitat resistance makes the cost-weighted PC equal the straight-line PC on a
    landscape that is uniformly habitat, and lower wherever the way is harder than that.  This is a convention,
    not a calibration: it fixes the scale of the effective distance, not the species' real cost of movement.
    """
    values = np.asarray(resistance)[np.asarray(habitat_mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("в маске нет местообитаний")
    return float(np.median(values))
