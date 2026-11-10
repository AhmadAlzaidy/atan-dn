"""Honest evaluation of baseline window scores with the SAME protocol as ATAN-DN (``evaluate_mission``).

Input: the ``.npz`` written by a baseline runner with ``--export-scores`` (raw scores of windows of the fit,
validation and test slices of one split file, see ``baselines/split_io.py``), the full label array and timestamps of
the stream, and the mission config. The module is importable without torch (heavy imports are lazy); calling
:func:`evaluate_baseline_scores` needs the ``atandn`` package (its ``data`` sub-package imports torch).

Protocol (identical to ``atandn.evaluation.pipeline``, only the model differs):

1. **Normalisation.** The ``eval.normalizer`` (1st/99th percentile, clipped to [0, 1]) is fitted on the FIT-slice
   scores of the baseline. The grid of thresholds is [0.05, 0.95], so raw unbounded reconstruction errors need it.
2. **Known-event mask** (only when ``stage3`` and ``eval.use_known_event_mask``), built by the SAME functions as
   ``atandn.evaluation.pipeline`` (``known_event_intervals``, ``mask_buffers``, ``telecommand_mask_params``): scores at
   window ends inside a known-event interval, extended by ``mask_buffer_before/after_minutes``, are set to 0.
   ``eval.mask_source == "rare_events"``: the mission's rare-event intervals (``intervals``). ``"telecommands"`` (default):
   executions of the telecommand types selected on the TRAINING split (``mask_params["telecommand"]`` of the ATAN-DN
   ``mask_params.json``; without it they are derived from ``values``/``labels`` of the training part with
   ``derive_telecommand_settling``), up to the last training timestamp for the fit/validation windows and in the test
   period for the test windows. Missing telecommands raise the same ``ValueError`` as the pipeline. The after-buffer is
   ``eval.mask_buffer_after_minutes`` or, if ``None``, ``mask_params["mask_buffer_after_minutes"]`` (for telecommands
   the P95 settling buffer); it never depends on test labels.
3. **Threshold.** ``select_threshold_filtered`` (grid F1, spike filter inside the search) on the VALIDATION-slice
   windows only (windows entirely inside the validation block). No positive window there: ``ValueError`` unless
   ``eval.allow_train_threshold_fallback`` (then the FIT windows are used and ``threshold_source = "train_fallback"``).
4. **Spike filter** ``min_consecutive`` (``eval.min_consecutive``; ``"derived"`` = ``mask_params["min_consecutive_windows"]``).
5. **Test.** Predictions ``score > threshold`` -> spike filter. Test labels are read only afterwards for the metrics.

Window grids (``grid``):

* ``"native"`` (default): the baseline's own windows (its own length and stride). Window label = 1 if the window holds
  an anomalous sample. Point-wise metrics use rule A with the baseline's stride (``windows_to_samples``): the decision
  of the window ending at ``t`` is given to samples ``t - stride + 1 .. t`` (causal).
* ``"atandn"``: the ATAN-DN window grid of the split file (``window_size``, ``test_stride``). Each grid window ending at
  ``g`` receives the score of the LATEST baseline window that ended at or before ``g`` (``mapped_end <= g``, age at
  most ``stride - 1`` samples), so a score never reaches back in time (causal); grid windows without such a baseline
  window are dropped. Labels use the ATAN-DN window length, so the baseline is compared on the same decision unit.

If a baseline stride differs from the ATAN-DN stride the spike-filter length counts baseline windows in ``native``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:                                   # same directory when run as a script / from tests that add baselines/ to sys.path
    import split_io
except ImportError:                    # pragma: no cover
    from baselines import split_io     # type: ignore

log = logging.getLogger(__name__)

GRIDS = ("native", "atandn")


@dataclass
class _Part:
    """Windows of one slice: raw scores, absolute end indices, label window length and stride."""

    scores: np.ndarray
    end: np.ndarray
    window: int
    stride: int


def window_labels_from_ends(labels: np.ndarray, end: np.ndarray, window: int) -> np.ndarray:
    """Window label = 1 if any of the ``window`` samples ending at ``end`` (inclusive) is anomalous."""
    csum = np.concatenate([[0], np.cumsum(np.asarray(labels) > 0)])
    end = np.asarray(end, dtype=np.int64)
    return ((csum[end + 1] - csum[end + 1 - window]) > 0).astype(np.int64)


def count_event_runs(labels: np.ndarray) -> int:
    """Number of contiguous runs of positive samples."""
    y = (np.asarray(labels) > 0).astype(np.int8)
    return int(np.sum(np.diff(np.r_[0, y]) == 1)) if y.size else 0


def map_scores_to_grid(scores: np.ndarray, ends: np.ndarray, grid_ends: np.ndarray,
                       max_age: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal mapping of window scores to a grid of window ends (last observation carried forward).

    Each grid end ``g`` takes the score of the latest source window with ``end <= g`` provided ``g - end <= max_age``.
    A score therefore never influences a grid window that ends before the source window ended.

    Args:
        scores: ``(m,)`` scores of source windows.
        ends: ``(m,)`` ascending end indices of the source windows.
        grid_ends: ``(g,)`` end indices of the target windows.
        max_age: largest allowed ``g - end`` (use ``stride - 1`` of the source windows).

    Returns:
        ``(mapped, valid)``: ``mapped`` has the score where ``valid`` is True (0 elsewhere).
    """
    scores, ends, grid_ends = np.asarray(scores, dtype=np.float64), np.asarray(ends), np.asarray(grid_ends)
    if len(ends) == 0:
        return np.zeros(len(grid_ends)), np.zeros(len(grid_ends), dtype=bool)
    idx = np.searchsorted(ends, grid_ends, side="right") - 1
    ok = idx >= 0
    safe = np.where(ok, idx, 0)
    ok &= (grid_ends - ends[safe]) <= max_age
    return np.where(ok, scores[safe], 0.0), ok


def _inside(part: _Part, lo: int, hi: int) -> _Part:
    """Keep the windows lying entirely inside ``[lo, hi)``."""
    keep = (part.end - part.window + 1 >= lo) & (part.end < hi)
    return _Part(part.scores[keep], part.end[keep], part.window, part.stride)


def _to_grid(part: _Part, split: dict, name: str, grid: str) -> _Part:
    """Return the part on the requested grid (``native``: unchanged)."""
    lo, hi = split_io.part_range(split, name)
    part = _inside(part, lo, hi)
    if grid == "native":
        return part
    w, s = int(split["window_size"]), int(split["test_stride"])
    starts = split_io.window_starts_in(lo, hi, w, s)
    ends = starts + w - 1
    mapped, ok = map_scores_to_grid(part.scores, part.end, ends, max_age=part.stride - 1)
    return _Part(mapped[ok], ends[ok], w, s)


def _mask(scores: np.ndarray, ts_end, intervals: Sequence, before: float, after: float) -> np.ndarray:
    if not len(intervals):
        return np.asarray(scores, dtype=np.float64)
    from atandn.inference.postprocess import apply_known_event_mask

    return apply_known_event_mask(scores, ts_end, intervals, buffer_before_minutes=before, buffer_after_minutes=after)


def evaluate_baseline_scores(npz_path: str | Path, cfg: Any, *, labels: np.ndarray, timestamps: Any,
                             split: dict | str | Path, intervals: Sequence | None = None,
                             telecommands: Any = None, values: np.ndarray | None = None,
                             mask_params: dict | None = None, stage3: bool = True, grid: str = "native",
                             return_arrays: bool = False, report_extra: dict | None = None) -> dict:
    """Evaluate a baseline's exported window scores with the shared honest protocol (see the module docstring).

    Args:
        npz_path: score file from a runner (``--export-scores``).
        cfg: :class:`atandn.config.MissionConfig` (``eval.*`` decides normalizer, threshold method, mask, spike filter).
        labels: ``(N,)`` sample labels of the FULL stream. Only labels of the validation (training-split) slice enter
            the threshold; test labels are read only after the predictions exist.
        timestamps: ``(N,)`` timestamps of the full stream (naive UTC or convertible).
        split: split dict or path of the JSON that the runner was given.
        intervals: rare-event intervals ``[(start, end), ...]`` of the mission (``meta["known_event_intervals"]``); used
            with ``eval.mask_source == "rare_events"``.
        telecommands: table ``time, type, priority`` of the mission (``meta["telecommands"]``); needed with
            ``eval.mask_source == "telecommands"`` (and the mask on).
        values: ``(N, C)`` telemetry of the full stream; only used to derive the telecommand buffers from the TRAINING
            part when ``mask_params`` has no ``"telecommand"`` entry.
        mask_params: training-derived mask parameters (``mask_params.json`` of an ATAN-DN run on the same mission).
        stage3: ``False`` reports the baseline WITHOUT mask and spike filter (``--no-stage3``).
        grid: ``"native"`` or ``"atandn"`` (see the module docstring).
        return_arrays: add ``"_arrays"`` (test scores, predictions, window labels, end indices; in memory only).
        report_extra: extra top-level keys merged into the report.

    Returns:
        JSON-able dict with the keys of ``evaluate_mission``: ``mission``, ``protocol``, ``threshold``,
        ``threshold_source``, ``normalizer``, ``postprocessing``, ``validation``, window counts, ``n_test_events``,
        ``n_test_pos_samples``, ``window_metrics``, ``pointwise_metrics_rule_A``, ``esa_adb_event_wise`` plus ``baseline``.

    Raises:
        ValueError: grid unknown, score file does not belong to the split, parts missing, no validation window or
            no positive validation window (without the fallback).
    """
    from atandn.data.splits import to_naive_utc
    from atandn.evaluation.metrics import point_adjusted_metrics, pointwise_metrics, window_metrics, windows_to_samples
    from atandn.evaluation.pipeline import (derive_telecommand_settling, known_event_intervals, mask_buffers,
                                            require_telecommands, telecommand_mask_params)
    from atandn.inference.postprocess import min_consecutive_filter
    from atandn.inference.scoring import ScoreNormalizer
    from atandn.inference.threshold import select_threshold_filtered

    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {GRIDS}")
    sp = split_io.read_split(split) if isinstance(split, (str, Path)) else split_io.validate_split(split)
    sc = split_io.load_scores(npz_path)
    if sc["ts_sha1"] != sp["timestamps"]["sha1"]:
        raise ValueError("the score file was produced for a different split/stream (timestamp hash differs)")
    for need in split_io.PARTS:
        if need not in sc:
            raise ValueError(f"the score file has no '{need}' scores (export fit, val and test)")
    labels = np.asarray(labels)
    ts_all = to_naive_utc(timestamps)
    if len(labels) != int(sp["n_samples"]) or len(ts_all) != len(labels):
        raise ValueError("labels/timestamps do not have the length of the split's stream")
    ev, w_b, s_b = cfg.eval, sc["window_size"], sc["stride"]
    t0, t1 = split_io.part_range(sp, "test")
    mp = dict(mask_params or {})

    native = {p: _Part(sc[p][0], sc[p][1], w_b, s_b) for p in split_io.PARTS}
    fit_n = _inside(native["fit"], *split_io.part_range(sp, "fit"))
    if len(fit_n.scores) == 0:
        raise ValueError("no fit window: the normalizer cannot be fitted")
    norm = ScoreNormalizer(ev.normalizer).fit(fit_n.scores)          # FIT-slice scores only
    val = _to_grid(native["val"], sp, "val", grid)
    test = _to_grid(native["test"], sp, "test", grid)
    if len(val.scores) == 0 or len(test.scores) == 0:
        raise ValueError("no validation or no test window on the requested grid")

    # ---- Stage-3 settings (training-derived; never from test labels) -------------------------------------------
    use_mask = bool(stage3 and ev.use_known_event_mask)
    train_end = pd.Timestamp(ts_all[t0 - 1])
    selected = None
    if use_mask and ev.mask_source == "telecommands":
        require_telecommands(telecommands)
        settle = mp.get("telecommand")
        if settle is None:
            if values is None:
                raise ValueError("eval.mask_source is 'telecommands': pass the mask_params.json of the ATAN-DN run "
                                 "(it holds the training-selected types and buffer) or `values` to derive them from "
                                 "the training split")
            settle = derive_telecommand_settling(np.asarray(values)[:t0], ts_all[:t0], labels[:t0], telecommands,
                                                 train_end=train_end)
        mp = telecommand_mask_params(mp, settle)
        selected = settle["selected_types"]
    before, after = mask_buffers(cfg, mp)
    if use_mask and ev.mask_buffer_after_minutes is None and "mask_buffer_after_minutes" not in mp:
        log.warning("no mask_params given: the known-event after-buffer is 0 minutes")
    iv_kw = dict(rare_intervals=intervals, telecommands=telecommands, selected_types=selected, train_end=train_end,
                 test_end=pd.Timestamp(ts_all[-1]), enabled=use_mask)
    iv_train = known_event_intervals(cfg, period="train", **iv_kw)       # fit / validation windows
    iv_test = known_event_intervals(cfg, period="test", **iv_kw)
    if not stage3:
        min_consec, mc_source = 1, "stage3_off"
    elif ev.min_consecutive == "derived":
        if "min_consecutive_windows" not in mp:
            log.warning("eval.min_consecutive is 'derived' but no mask_params were given: spike filter length 1")
        min_consec, mc_source = max(int(mp.get("min_consecutive_windows", 1)), 1), "derived_from_training"
    else:
        min_consec, mc_source = int(ev.min_consecutive), "config"

    def prepared(part: _Part, iv: Sequence) -> np.ndarray:
        s = norm.transform(part.scores)
        return _mask(s, ts_all[part.end], iv, before, after)

    # ---- threshold on the VALIDATION windows (labels of the training split only) ---------------------------------
    y_val = window_labels_from_ends(labels[:t0], val.end, val.window)    # training-split labels only
    m_val = prepared(val, iv_train)
    if y_val.any():
        thr, val_f1 = select_threshold_filtered(m_val, y_val, ev.threshold_method, min_consec)
        source = "validation"
    elif ev.allow_train_threshold_fallback:
        log.warning("validation slice has no positive window: threshold fitted on the FIT windows (train_fallback)")
        y_fit = window_labels_from_ends(labels[:t0], fit_n.end, fit_n.window)
        thr, val_f1 = select_threshold_filtered(prepared(fit_n, iv_train), y_fit, ev.threshold_method, min_consec)
        source = "train_fallback"
    else:
        raise ValueError("the validation slice has no positive window, so the F1-optimal threshold is undefined "
                         "(set eval.allow_train_threshold_fallback: true to fit it on the fit windows)")

    # ---- the test slice: scores, mask, threshold, spike filter (no test label yet) --------------------------------
    m_test = prepared(test, iv_test)
    pred = min_consecutive_filter((m_test > thr).astype(int), min_consec)

    # ---- test labels are used only from here on ---------------------------------------------------------------------
    test_labels = labels[t0:t1]
    y_test = window_labels_from_ends(test_labels, test.end - t0, test.window)
    win = window_metrics(y_test, pred)
    sample_pred, valid = windows_to_samples(pred, test.end - t0, len(test_labels), test.stride, test.window)
    point = pointwise_metrics(test_labels, sample_pred, valid)
    point_pa = point_adjusted_metrics(test_labels[valid], sample_pred[valid])
    n_events, n_pos = count_event_runs(test_labels), int((test_labels > 0).sum())

    report: dict[str, Any] = {
        "mission": sc["mission"] or getattr(cfg, "name", ""),
        "protocol": {
            "validation_block_from_training_split": True, "threshold_source": source,
            "threshold_method": ev.threshold_method, "normalizer_source": "baseline fit-slice scores",
            "mask_buffer_source": "training", "test_labels_used_for_fitting": False,
            "point_adjustment": bool(getattr(ev, "point_adjustment", True)),
            "test_split_scored_once": True, "window_size": int(test.window), "test_stride": int(test.stride),
            "grid": grid, "stage3": bool(stage3),
        },
        "threshold": float(thr), "threshold_source": source, "threshold_selection_f1": float(val_f1),
        "normalizer": norm.state_dict(),
        "postprocessing": {
            "known_event_mask": bool(iv_test or iv_train), "n_known_events": len(iv_test),
            "mask_source": ev.mask_source if use_mask else "none", "telecommand_selected_types": selected,
            "mask_buffer_before_minutes": before, "mask_buffer_after_minutes": after,
            "min_consecutive": int(min_consec), "min_consecutive_source": mc_source,
            "training_recommended_min_consecutive": mp.get("min_consecutive_windows"),
        },
        "validation": {"n_val_windows": int(len(y_val)), "n_val_pos_windows": int(y_val.sum()),
                       "val_has_positives": bool(y_val.any()), "n_fit_windows": int(len(fit_n.scores))},
        "n_val_windows": int(len(y_val)), "n_val_pos_windows": int(y_val.sum()),
        "n_test_windows": int(len(y_test)), "n_test_pos_windows": int(y_test.sum()),
        "n_test_pos_samples": n_pos, "n_test_events": n_events,
        "window_metrics": win, "pointwise_metrics_rule_A": point, "point_adjusted_metrics": point_pa,
        "esa_adb_event_wise": {"computed": False, "reason": "not computed for baselines"},
        "baseline": {"model": sc["model"], "native_window_size": int(w_b), "native_stride": int(s_b),
                     "grid": grid, "split_window_size": int(sp.get("window_size", 0)),
                     "split_test_stride": int(sp.get("test_stride", 0))},
    }
    report.update(report_extra or {})
    if return_arrays:
        report["_arrays"] = {"test_scores": m_test, "test_pred": pred, "test_window_labels": y_test,
                             "end_index": test.end, "val_scores": m_val, "val_window_labels": y_val}
    return report
