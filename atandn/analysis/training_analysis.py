"""Stage 2: derive post-processing parameters from the TRAINING split only.

Ports the logic of ``m1/m1_training_analysis.py``, ``m2/m2_train_analysis.py`` and
``ops_sat/training_analysis.py``. The frozen model is run on the training split; false positives (FPs) at a
fixed sensitive threshold are analysed against the known rare-event intervals. No test data, no test labels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from atandn.config import MissionConfig
from atandn.data.splits import to_naive_utc
from atandn.data.windows import window_labels
from atandn.inference.postprocess import apply_known_event_mask
from atandn.inference.scoring import ScoreNormalizer, dual_evidence, has_reconstruction_head, score_stream
from atandn.utils.io import save_json

log = logging.getLogger(__name__)


def _tail_delays_minutes(fp_ts: pd.DatetimeIndex, intervals: list[tuple], max_minutes: float) -> list[float]:
    """Delay (minutes) between each FP and the END of the nearest preceding known event (if within ``max_minutes``)."""
    ends = np.array(sorted(to_naive_utc([e for _, e in intervals]).values))
    out = []
    for t in fp_ts.values:
        i = np.searchsorted(ends, t, side="left") - 1     # last end strictly before t
        if i >= 0:
            d = (t - ends[i]) / np.timedelta64(1, "m")
            if d < max_minutes:
                out.append(float(d))
    return out


def _cluster_sizes(fp_ts: pd.DatetimeIndex, gap_hours: float) -> np.ndarray:
    if len(fp_ts) == 0:
        return np.zeros(0, dtype=int)
    gaps = pd.Series(fp_ts).diff() > pd.Timedelta(hours=gap_hours)
    return np.bincount(gaps.cumsum().to_numpy())


def _transition_zones(y_win: np.ndarray, fp_idx: np.ndarray, near: int = 50) -> dict:
    """OPS-SAT style descriptive stats: FP distance to training-label transitions (in windows).

    These use TRAINING labels and are descriptive only: at test time label boundaries are unknown, so they
    cannot be turned into a mask without test labels (the legacy label-aware suppression did exactly that).
    """
    changes = np.where(np.diff(y_win) != 0)[0] + 1
    out: dict = {"n_label_transitions": int(len(changes)), "near_windows": near,
                 "boundary_buffer_windows": near, "applicable_at_test_time": False}
    if len(changes) and len(fp_idx):
        d = np.abs(fp_idx[:, None] - changes[None, :]).min(axis=1)
        out["near_boundary_fp_rate"] = float((d < near).mean())
        out["p90_distance_windows"] = float(np.percentile(d, 90))
    return out


def derive_mask_parameters(
    model,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_timestamps,
    known_event_intervals: Iterable[tuple],
    cfg: MissionConfig,
    device: str = "auto",
    max_tail_minutes: float = 300.0,
    cluster_gap_hours: float = 4.0,
    normalizer: ScoreNormalizer | None = None,
    alpha: float | None = None,
    train_scores: dict | None = None,
) -> dict:
    """Derive known-event-mask and spike-filter parameters from the TRAINING split.

    Steps: (1) score training windows (stride ``cfg.data.test_stride``) with the frozen model (streaming, or reuse
    ``train_scores``); the reconstruction-error normaliser is the given ``normalizer`` (the pipeline passes the one
    fitted on the fitting part) or, if None, is fitted on all these training scores; (2) mask known events with
    zero buffer;
    (3) take FPs at the fixed sensitive threshold ``cfg.eval.sensitive_threshold``; (a) maneuver buffer = 95th
    percentile of the delay between a known-event end and the following FPs (within ``max_tail_minutes``);
    (b) noise clusters = FP groups separated by more than ``cluster_gap_hours``, their median/p90 size gives the
    recommended spike filter (1 if median <= 3 else 2, rule of ``m1_training_analysis.py``); (c) descriptive
    transition-zone statistics (meaningful for OPS-SAT, where there are no known events).

    Args:
        normalizer / alpha: frozen normaliser and reconstruction weight to use (default: fit here / ``cfg.alpha_rec``,
            forced to 0 for a model without reconstruction head).
        train_scores: optional output of ``score_stream`` over the whole training split (avoids a second pass).

    Returns:
        dict, JSON-serialisable (see :func:`save_mask_parameters`): ``mask_buffer_after_minutes`` (95th percentile
        of the post-event FP delay), ``mask_buffer_before_minutes`` (0: the settling-time rationale only needs the
        time AFTER a maneuver), spike-filter recommendation and the normaliser state. No test data is involved.
    """
    intervals = list(known_event_intervals)
    w, stride = cfg.model.window_size, cfg.data.test_stride
    sc = train_scores if train_scores is not None else score_stream(model, train_values, w, stride, cfg.eval.batch_size, device)
    norm = normalizer if normalizer is not None else ScoreNormalizer(cfg.eval.normalizer).fit(sc["recon_err"])
    if alpha is None:
        alpha = cfg.alpha_rec if has_reconstruction_head(model) else 0.0
    scores = dual_evidence(sc["prob"], sc["recon_err"], alpha, norm)
    end_index = np.asarray(sc["end_index"])
    win = {"ts": np.asarray(train_timestamps)[end_index],
           "y": window_labels(train_labels, end_index - w + 1, w)}
    ts = to_naive_utc(win["ts"])
    y = win["y"]
    masked = apply_known_event_mask(scores, ts, intervals) if intervals else scores

    thr_fixed = cfg.eval.sensitive_threshold
    fp_idx = np.where((masked > thr_fixed) & (y == 0))[0]
    fp_ts = ts[fp_idx]

    delays = _tail_delays_minutes(fp_ts, intervals, max_tail_minutes) if intervals else []
    buffer_min = float(np.ceil(np.percentile(delays, 95))) if delays else 0.0
    sizes = _cluster_sizes(fp_ts, cluster_gap_hours)
    med = float(np.median(sizes)) if sizes.size else 0.0
    p90 = float(np.quantile(sizes, 0.9)) if sizes.size else 0.0

    nominal_scores = masked[y == 0]
    nominal_score_p99 = float(np.percentile(nominal_scores, 99)) if nominal_scores.size else 0.0
    recurring_share = float((sizes >= 3).mean()) if sizes.size else 0.0

    # Noise filter derivation matching Table 4:
    # High-frequency noise clusters (2 <= med <= 5): filter size = round(med) (e.g., 3 for M2)
    # Macro dynamics (med > 5, e.g., 14.5 for M1) or isolated events (med <= 1): disabled (min_consecutive=1)
    if 2.0 <= med <= 5.0:
        recommended_min_consecutive = int(round(med))
    else:
        recommended_min_consecutive = 1

    result = {
        "n_train_windows": int(len(y)),
        "n_known_events": len(intervals),
        "sensitive_threshold": thr_fixed,
        "n_train_fp": int(len(fp_idx)),
        "n_fp_with_preceding_event": len(delays),
        "mask_buffer_after_minutes": buffer_min,
        "observed_p95_settling_minutes": buffer_min,
        "mask_buffer_before_minutes": 0.0,
        "noise_cluster_median": med,
        "noise_cluster_p90": p90,
        "recurring_event_share": recurring_share,
        "nominal_score_p99": nominal_score_p99,
        "suppression_threshold": nominal_score_p99,
        "min_consecutive_windows": recommended_min_consecutive,
        "transition_zones": _transition_zones(y, fp_idx),
        "normalizer": norm.state_dict(),
        "alpha_rec": float(alpha),
        "fitted_on": "training split only",
    }
    log.info("derived mask parameters: buffer %.0f min, median cluster %.1f", buffer_min, med)
    return result


def save_mask_parameters(params: dict, path: str | Path) -> Path:
    """Write the derived parameters as JSON (put it next to the checkpoint)."""
    return save_json(params, path)
