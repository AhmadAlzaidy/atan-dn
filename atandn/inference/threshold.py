"""Threshold selection (fitted on the validation block carved from the training split; never on test)."""
from __future__ import annotations

import logging

import numpy as np

from atandn.inference.postprocess import min_consecutive_filter

log = logging.getLogger(__name__)

GRID_LOW, GRID_HIGH, GRID_STEP = 0.05, 0.95, 0.005


def _check_finite(s: np.ndarray) -> None:
    """Raise a clear error for NaN / infinite scores (a NaN would silently count as 'not above any threshold')."""
    if not np.all(np.isfinite(s)):
        raise ValueError(f"scores contain {int((~np.isfinite(s)).sum())} NaN/inf value(s) of {s.size}: the model "
                         "produced non-finite scores, no threshold can be selected")


def f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray, fallback_quantile: float | None = None) -> float:
    """Return the threshold maximising F1 for the rule ``pred = score > threshold``.

    Call this with TRAINING-split scores and labels only; the returned value is then frozen and applied to the
    test scores. Candidates are midpoints between consecutive distinct scores (plus one below the minimum).

    Args:
        scores: Window scores (higher = more anomalous).
        labels: 0/1 window labels of the same windows.
        fallback_quantile: If the labels contain no positive, return this quantile of ``scores`` instead of
            raising (e.g. 0.999); with ``None`` a ``ValueError`` is raised because F1 is undefined.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and of equal shape")
    _check_finite(s)
    if not y.any():
        if fallback_quantile is None:
            raise ValueError("No positive labels in the fitting data: F1-optimal threshold is undefined")
        return float(np.quantile(s, fallback_quantile))
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    # Only cut between distinct scores (ties are predicted together).
    last_of_group = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp, fp, sv = tp[last_of_group], fp[last_of_group], s_sorted[last_of_group]
    f1 = 2 * tp / (tp + fp + y.sum())          # 2TP / (2TP + FP + FN) with FN = P - TP
    best = int(np.argmax(f1))                    # first = highest threshold among ties in F1
    lower = sv[best + 1] if best + 1 < len(sv) else sv[best] - 1.0
    return float((sv[best] + lower) / 2.0)


def grid_f1_search(scores: np.ndarray, labels: np.ndarray, low: float = GRID_LOW, high: float = GRID_HIGH,
                   step: float = GRID_STEP, min_len: int = 1) -> tuple[float, float]:
    """Grid search of the F1-optimal threshold; returns ``(threshold, validation_f1)``.

    Rule ``pred = score > threshold``; with ``min_len > 1`` the predictions of EVERY candidate are passed through
    :func:`atandn.inference.postprocess.min_consecutive_filter` before F1 is computed, so the threshold is optimal
    for the post-processed predictions that the test split will get. ``scores``/``labels`` must be consecutive
    windows of one block in time order (runs are only meaningful then). Ties in F1 resolve to the smallest
    threshold. Scores are not clipped: a saturated score of 1.0 is above every grid point (a positive window is
    found at every threshold), a score of 0.0 (e.g. a masked window) below all of them.

    Raises:
        ValueError: NaN/inf scores, no positive label, or empty / mismatched inputs.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if s.shape != y.shape or s.size == 0:
        raise ValueError("scores and labels must be non-empty and of equal shape")
    _check_finite(s)
    if not y.any():
        raise ValueError("No positive labels in the validation block: the grid F1-optimal threshold is undefined")
    grid = np.linspace(low, high, int(round((high - low) / step)) + 1)
    if min_len <= 1:
        sp, sn = np.sort(s[y]), np.sort(s[~y])
        tp = (len(sp) - np.searchsorted(sp, grid, side="right")).astype(np.float64)   # #(score > thr) among positives
        fp = (len(sn) - np.searchsorted(sn, grid, side="right")).astype(np.float64)
    else:
        tp, fp = np.zeros(len(grid)), np.zeros(len(grid))
        for i, t in enumerate(grid):
            pred = min_consecutive_filter((s > t).astype(int), min_len).astype(bool)
            tp[i], fp[i] = (pred & y).sum(), (pred & ~y).sum()
    f1 = 2 * tp / (tp + fp + y.sum())
    best = int(np.argmax(f1))
    if f1[best] == 0.0:
        log.warning("validation F1 is 0 at every grid threshold (positives never exceed the grid or no true positive "
                    "after filtering); the threshold %.3f is arbitrary", grid[best])
    elif best in (0, len(grid) - 1):
        log.warning("the F1-optimal threshold %.3f sits on the edge of the grid [%.2f, %.2f]; scores may be "
                    "saturated or the optimum lies outside the grid", grid[best], low, high)
    return float(grid[best]), float(f1[best])


def grid_f1_threshold(scores: np.ndarray, labels: np.ndarray, low: float = GRID_LOW, high: float = GRID_HIGH,
                      step: float = GRID_STEP) -> float:
    """F1-optimal threshold on the grid ``low, low + step, ..., high`` (the paper's procedure).

    Rule ``pred = score > threshold``. Ties in F1 resolve to the smallest threshold. Call it with VALIDATION
    scores/labels only. See :func:`grid_f1_search` (edge cases: F1 == 0 and grid-boundary results log a WARNING).

    Raises:
        ValueError: if the labels contain no positive (F1 is undefined), the scores are NaN/inf, or the inputs are
            empty / mismatched.
    """
    return grid_f1_search(scores, labels, low, high, step)[0]


def select_threshold_filtered(scores: np.ndarray, labels: np.ndarray, method: str = "grid", min_len: int = 1,
                              fallback_quantile: float | None = None) -> tuple[float, float]:
    """Select the threshold on a validation block with the spike filter inside the search; ``(threshold, F1)``.

    ``method="grid"``: :func:`grid_f1_search` (filter applied per candidate). ``method="exact"``: the unfiltered
    exact optimum is the only candidate when ``min_len <= 1``; otherwise the candidates are the exact optimum plus
    the 0.005 grid, and the one with the best filtered F1 wins (ties: smallest).
    """
    if method == "grid":
        return grid_f1_search(scores, labels, min_len=min_len)
    if method != "exact":
        raise ValueError("method must be 'grid' or 'exact'")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    thr0 = f1_optimal_threshold(s, y, fallback_quantile)
    cands = [thr0] if min_len <= 1 else sorted({thr0, *np.linspace(GRID_LOW, GRID_HIGH, 181).tolist()})
    best_t, best_f = thr0, -1.0
    for t in cands:
        pred = min_consecutive_filter((s > t).astype(int), min_len).astype(bool)
        tp, fp = (pred & y).sum(), (pred & ~y).sum()
        f = 2.0 * tp / (tp + fp + y.sum()) if y.any() else 0.0
        if f > best_f + 1e-15:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def select_threshold(scores: np.ndarray, labels: np.ndarray, method: str = "grid",
                     fallback_quantile: float | None = None) -> float:
    """Dispatch to :func:`grid_f1_threshold` (default) or :func:`f1_optimal_threshold` (``method="exact"``)."""
    if method == "grid":
        return grid_f1_threshold(scores, labels)
    if method == "exact":
        return f1_optimal_threshold(scores, labels, fallback_quantile)
    raise ValueError("method must be 'grid' or 'exact'")


def nominal_score_quantile(scores: np.ndarray, labels: np.ndarray, q: float = 0.99) -> float:
    """``q``-quantile of the scores of the NORMAL windows (label 0); training windows only, no test data.

    Raises:
        ValueError: no normal window, or non-finite scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    normal = np.asarray(labels) <= 0
    if s.shape != normal.shape or not normal.any():
        raise ValueError("nominal_score_quantile needs scores and labels of equal shape with at least one normal window")
    _check_finite(s[normal])
    return float(np.quantile(s[normal], q))


def apply_threshold_floor(threshold: float, mode: str, fit_scores: np.ndarray, fit_labels: np.ndarray,
                          q: float = 0.99) -> tuple[float, dict]:
    """Raise a validated threshold to a label-free floor: ``tau = max(tau*, floor)``.

    ``mode="nominal_p99"``: floor = 99th percentile of the scores of the NORMAL windows of the fitting part of the
    TRAINING split (the same chain as at test time: masked, before the spike filter). This is a threshold floor, not a
    label-aware suppression: it uses training labels only to pick the normal training windows, and nothing from the
    test split. ``mode="none"`` returns the threshold unchanged.

    Returns:
        ``(threshold, info)`` with ``info`` = mode, quantile, floor, threshold before the floor, ``applied`` (the floor
        raised the threshold) and the data source.
    """
    if mode == "none":
        return float(threshold), {"mode": "none", "applied": False, "floor": None}
    if mode != "nominal_p99":
        raise ValueError("threshold floor mode must be 'none' or 'nominal_p99'")
    floor = nominal_score_quantile(fit_scores, fit_labels, q)
    new = max(float(threshold), floor)
    info = {"mode": mode, "quantile": q, "floor": floor, "threshold_before_floor": float(threshold),
            "applied": bool(floor > threshold), "source": "fitting part of the training split, normal windows",
            "n_normal_windows": int((np.asarray(fit_labels) <= 0).sum())}
    if info["applied"]:
        log.info("threshold floor raised the threshold from %.4f to %.4f (P%d of nominal fit-part scores)", threshold,
                 new, round(100 * q))
    return new, info
