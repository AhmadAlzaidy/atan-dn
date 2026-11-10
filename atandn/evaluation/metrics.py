"""Window-level and sample-level metrics (no point-adjustment in the headline metrics)."""
from __future__ import annotations

import warnings

import numpy as np


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    tp, fp = int((t & p).sum()), int((~t & p).sum())
    fn, tn = int((t & ~p).sum()), int((~t & ~p).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _rates(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        "precision": p, "recall": r,
        "fpr": _div(fp, fp + tn), "fnr": _div(fn, fn + tp),
        "f1": _div(2 * p * r, p + r),
        "f0_5": _div(1.25 * p * r, 0.25 * p + r),   # F_beta with beta = 0.5
        **c,
    }


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion-matrix metrics of window decisions.

    Returns:
        dict with precision, recall, fpr, fnr, f1, f0_5 (``1.25PR/(0.25P+R)``), tp, fp, tn, fn. Undefined
        ratios (zero denominator) are reported as 0.0.
    """
    return _rates(_confusion(y_true, y_pred))


def windows_to_samples(
    window_pred: np.ndarray, end_index: np.ndarray, n_samples: int, stride: int = 10, window_size: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Rule A: give the decision of the window ending at sample ``t`` to samples ``t-stride+1 .. t``.

    Returns:
        ``(sample_pred, valid_mask)``. ``valid_mask`` is False for the first ``window_size - 1`` samples (no
        full window yet) and for samples not covered by any window (e.g. the tail after the last window).
    """
    pred = np.zeros(n_samples, dtype=np.int64)
    covered = np.zeros(n_samples, dtype=bool)
    for w, t in zip(np.asarray(window_pred), np.asarray(end_index)):
        lo = max(int(t) - stride + 1, 0)
        hi = min(int(t) + 1, n_samples)
        pred[lo:hi] = int(w)
        covered[lo:hi] = True
    valid = covered.copy()
    valid[: max(window_size - 1, 0)] = False
    return pred, valid


def pointwise_metrics(y_true_samples: np.ndarray, y_pred_samples: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """Sample-level metrics (same keys as :func:`window_metrics`) on the samples where ``valid_mask`` is True."""
    t, p = np.asarray(y_true_samples), np.asarray(y_pred_samples)
    if valid_mask is not None:
        m = np.asarray(valid_mask).astype(bool)
        t, p = t[m], p[m]
    return window_metrics(t, p)


def adjust_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Standard point-adjusted (PA) prediction adjustment matching ESA-ADB Supplementary Table 9.

    For each contiguous segment of true anomalies, if any sample inside the segment is predicted
    as anomalous, the entire contiguous segment is marked as detected (set to 1).
    Non-anomalous (nominal) segments are unchanged.

    Args:
        y_true: Binary ground-truth labels (0 = nominal, 1 = anomaly).
        y_pred: Binary model predictions.

    Returns:
        Adjusted binary prediction array of the same shape.
    """
    t = np.asarray(y_true).astype(int)
    adj = np.asarray(y_pred).astype(int).copy()
    diff = np.diff(np.r_[0, t, 0])
    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        if adj[s:e].any():
            adj[s:e] = 1
    return adj


def point_adjusted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point-adjusted (PA) event metrics matching ESA-ADB Supplementary Table 9.

    Evaluates predictions under standard benchmark point-adjusted event scoring:
    if an anomaly segment is touched by a detected alarm, the entire segment is marked detected.

    Args:
        y_true: Binary ground-truth labels.
        y_pred: Binary predictions before adjustment.

    Returns:
        dict with precision, recall, fpr, fnr, f1, f0_5, tp, fp, tn, fn.
    """
    adj = adjust_predictions(y_true, y_pred)
    return window_metrics(y_true, adj)
