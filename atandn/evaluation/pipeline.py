"""The ONE evaluation protocol shared by ``scripts/evaluate.py`` and ``experiments/protocol.py``.

Everything that is fitted comes from the TRAINING split; the test split is scored once with frozen values.

1. ``prepare_protocol_data`` carves the validation block out of the training split
   (:func:`atandn.data.splits.event_aware_validation_split`); the fitting part is the data before the block.
2. ``fit_frozen_protocol`` scores the training split once (streaming) and freezes
   * the reconstruction-error normalizer (1st/99th percentile of the FITTING-part errors),
   * the known-event mask parameters (:func:`atandn.analysis.derive_mask_parameters`, training split),
   * the decision threshold: F1-optimal on the VALIDATION block (grid [0.05, 0.95], step 0.005 by default) of the
     masked dual-evidence score AFTER the spike filter (same chain as the test: mask, threshold, spike filter);
     only windows whose whole span lies inside the block are used,
   * the spike-filter length ``min_consecutive`` (``eval.min_consecutive``: ``derived`` = training-analysis value). No positives in the block: error, unless ``eval.allow_train_threshold_fallback``
     (then the fitting-part scores are used and ``threshold_source = "train_fallback"``).
3. ``evaluate_mission`` scores the test split once (streaming) with the frozen values, applies the known-event
   mask (buffer after, and optionally before, each event) and the spike filter, and computes the metrics.

Parameter derivation per model: the mask parameters are derived from the training split FOR EACH trained model
(every ablation variant and seed gets its own, computed with that model's scores and frozen normalizer).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from atandn.analysis import derive_mask_parameters
from atandn.analysis.telecommand_buffers import derive_settling_buffers, telecommand_intervals
from atandn.config import MissionConfig
from atandn.data.splits import MissionData, ValidationSplit, count_events, event_aware_validation_split, to_naive_utc
from atandn.data.windows import window_labels
from atandn.evaluation.metrics import point_adjusted_metrics, pointwise_metrics, window_metrics, windows_to_samples
from atandn.inference.postprocess import apply_known_event_mask, known_event_mask_flags, min_consecutive_filter
from atandn.inference.scoring import ScoreNormalizer, dual_evidence, has_reconstruction_head, score_stream
from atandn.inference.threshold import apply_threshold_floor, select_threshold_filtered
from atandn.inference.threshold_link import link_report

log = logging.getLogger(__name__)

MIN_TEST_EVENTS_WARN = 10


@dataclass
class ProtocolData:
    """Splits of one mission: ``train``/``test`` plus the validation block inside ``train``.

    Attributes:
        train / test: chronological split of the stream.
        split: :class:`ValidationSplit` (slices index the TRAINING split).
        intervals: known (planned) event intervals of the mission (Rare-Event rows); used only if
            ``eval.use_known_event_mask`` and ``eval.mask_source == "rare_events"``.
        telecommands: table ``time, type, priority`` of the telecommand executions of the whole mission (or None); used
            only with ``eval.mask_source == "telecommands"``. The split into training/test is done by time.
    """

    train: MissionData
    test: MissionData | None
    split: ValidationSplit
    intervals: list
    telecommands: Any = None


def prepare_protocol_data(train: MissionData, test: MissionData | None, cfg: MissionConfig) -> ProtocolData:
    """Build the :class:`ProtocolData` of a mission (event-aware validation block carved from ``train`` only)."""
    w = cfg.model.window_size
    split = event_aware_validation_split(train.labels, cfg.train.val_fraction, w - 1, cfg.train.val_min_events,
                                         min_fit=w, min_fit_fraction=cfg.train.val_min_fit_fraction)
    return ProtocolData(train, test, split, list(train.meta.get("known_event_intervals", [])),
                        train.meta.get("telecommands"))


def data_identity(data: MissionData) -> dict:
    """Cheap fingerprint of a data array set (length, first/last timestamp, SHA-1 of values and labels)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(data.values).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.labels, dtype=np.int8)).tobytes())
    ts = np.asarray(data.timestamps)
    return {"n": int(len(data)), "first_timestamp": str(ts[0]) if len(ts) else None,
            "last_timestamp": str(ts[-1]) if len(ts) else None, "sha1": h.hexdigest()}


def resolve_min_consecutive(setting: int | str, mask_params: dict) -> tuple[int, str]:
    """Spike-filter length: the integer of the config, or (``"derived"``) the training-analysis value.

    Returns:
        ``(min_consecutive, source)`` with source ``"config"`` or ``"derived_from_training"``.
    """
    if setting == "derived":
        return max(int(mask_params.get("min_consecutive_windows", 1)), 1), "derived_from_training"
    return int(setting), "config"


def effective_alpha(model: torch.nn.Module, cfg: MissionConfig) -> float:
    """Reconstruction weight of the dual-evidence score; 0 when the model has no reconstruction head."""
    if not cfg.model.use_reconstruction or not has_reconstruction_head(model):
        return 0.0
    return float(cfg.alpha_rec)


def _active_intervals(data: ProtocolData, cfg: MissionConfig) -> list:
    """Rare-event intervals if the mask is on and ``eval.mask_source == "rare_events"`` (else empty)."""
    if cfg.eval.use_known_event_mask and cfg.eval.mask_source == "rare_events":
        return list(data.intervals)
    return []


def require_telecommands(telecommands: Any) -> pd.DataFrame:
    """The telecommand table, or the clear error shared by the pipeline and the baseline adapter."""
    if telecommands is None:
        raise ValueError("eval.mask_source is 'telecommands' but the data carry no telecommands: load the ESA mission "
                         "with data.telecommand_dir set (atandn.data.load_mission_from_config), or use "
                         "eval.mask_source: rare_events / eval.use_known_event_mask: false")
    return telecommands


def _telecommand_table(data: ProtocolData) -> pd.DataFrame:
    return require_telecommands(data.telecommands)


def derive_telecommand_settling(train_values, train_timestamps, train_labels, telecommands: Any,
                                train_end=None) -> dict:
    """Stage-2 settling buffers from the TRAINING split and its telecommands (shared with the baseline adapter)."""
    tc = require_telecommands(telecommands)
    end = to_naive_utc(train_timestamps)[-1] if train_end is None else train_end
    tc = tc[tc["time"] <= end]
    return derive_settling_buffers(train_values, train_timestamps, tc["time"], slice(None), labels=train_labels,
                                   tc_types=tc["type"].to_numpy())


def telecommand_mask_params(mask_params: dict, settle: dict) -> dict:
    """Mask parameters with the telecommand settling buffer as after-buffer (the rule of :func:`fit_frozen_protocol`)."""
    mask_params = dict(mask_params)
    if "fp_delay_buffer_minutes" not in mask_params:     # keep the FP-delay value of the analysis for the record
        mask_params["fp_delay_buffer_minutes"] = mask_params.get("mask_buffer_after_minutes")
    mask_params["mask_buffer_after_minutes"] = float(settle["buffer_minutes"])
    mask_params["telecommand"] = settle
    mask_params["mask_source"] = "telecommands"
    return mask_params


def mask_buffers(cfg: MissionConfig, mask_params: dict) -> tuple[float, float]:
    """``(before, after)`` buffers in minutes: the config value, or the training-derived one when ``None``."""
    ev = cfg.eval
    after = float(ev.mask_buffer_after_minutes if ev.mask_buffer_after_minutes is not None
                  else mask_params.get("mask_buffer_after_minutes", 0.0))
    return float(ev.mask_buffer_before_minutes), after


def known_event_intervals(cfg: MissionConfig, *, rare_intervals: Any, telecommands: Any, selected_types: Any,
                          train_end, test_end=None, period: str = "train", mask_source: str | None = None,
                          enabled: bool | None = None) -> list:
    """Known-event intervals of the mask for the training (``period="train"``) or test period.

    ``eval.use_known_event_mask`` off: none. ``mask_source == "rare_events"``: the mission's rare-event intervals
    (both periods). ``"telecommands"``: executions of ``selected_types``; ``train``: up to ``train_end``; ``test``:
    in ``(train_end, test_end]``. Raises the :func:`require_telecommands` error when the table is missing.
    ``mask_source`` / ``enabled`` override the config (``evaluate_mission`` passes the frozen values of the checkpoint).
    """
    ev = cfg.eval
    if not (ev.use_known_event_mask if enabled is None else enabled):
        return []
    if (ev.mask_source if mask_source is None else mask_source) == "rare_events":
        return list(rare_intervals or [])
    tc = require_telecommands(telecommands)
    if period == "train":
        tc = tc[tc["time"] <= train_end]
        return telecommand_intervals(tc["time"], tc["type"], selected_types)
    return telecommand_intervals(tc["time"], tc["type"], selected_types, lo=train_end, hi=test_end)


def _mask(scores: np.ndarray, ts, intervals: list, before: float, after: float) -> np.ndarray:
    if not intervals:
        return np.asarray(scores, dtype=np.float64)
    return apply_known_event_mask(scores, ts, intervals, buffer_before_minutes=before, buffer_after_minutes=after)


def _filtered_f1(scores: np.ndarray, y: np.ndarray, thr: float, min_len: int) -> float:
    """F1 of ``score > thr`` followed by the spike filter (the chain used on the test split)."""
    pred = min_consecutive_filter((scores > thr).astype(int), min_len).astype(bool)
    yb = np.asarray(y) > 0
    tp, fp = int((pred & yb).sum()), int((pred & ~yb).sum())
    return float(2 * tp / (tp + fp + int(yb.sum())))


def fit_frozen_protocol(model: torch.nn.Module, data: ProtocolData, cfg: MissionConfig,
                        mask_params: dict | None = None, device: str = "auto") -> dict:
    """Fit normalizer, mask parameters and threshold from the TRAINING split and return them as a frozen dict.

    Args:
        model: trained model (the one selected on the validation block; no refit afterwards).
        data: splits (the test split is never read here).
        cfg: config (``eval.*`` decides normalizer, buffers, threshold method and fallback).
        mask_params: precomputed mask parameters; ``None`` derives them from the training split with this model.

    Returns:
        JSON-able dict: ``threshold``, ``threshold_source``, ``threshold_method``, ``normalizer`` (state),
        ``alpha_rec``, ``mask_params``, ``mask_buffer_before_minutes``, ``mask_buffer_after_minutes``,
        ``min_consecutive``, ``known_event_mask``, ``n_known_events``, ``train_ratio``, ``test_stride``,
        ``window_size``, ``validation`` (split report + window counts), ``n_train_windows``.

    Raises:
        ValueError: validation block without positive windows and ``eval.allow_train_threshold_fallback`` is False
            (or the fitting part has no positive either).
    """
    w, stride, ev = cfg.model.window_size, cfg.data.test_stride, cfg.eval
    train, sp = data.train, data.split
    use_tc = bool(ev.use_known_event_mask and ev.mask_source == "telecommands")
    intervals = _active_intervals(data, cfg)
    settle: dict | None = None
    if use_tc:   # paper text: buffers = P95 settling after TRAINING telecommands of the types that cause excursions
        settle = (mask_params or {}).get("telecommand")
        if settle is None:
            settle = derive_telecommand_settling(train.values, train.timestamps, train.labels, data.telecommands)
        intervals = known_event_intervals(cfg, rare_intervals=data.intervals, telecommands=data.telecommands,
                                          selected_types=settle["selected_types"],
                                          train_end=to_naive_utc(train.timestamps)[-1], period="train")
    alpha = effective_alpha(model, cfg)

    sc = score_stream(model, train.values, w, stride, ev.batch_size, device)     # whole training split, once
    start, end_index = sc["start"], sc["end_index"]
    y = window_labels(train.labels, start, w)
    fit_m = start + w <= sp.fit.stop                       # windows entirely inside the fitting part
    # windows entirely inside the validation block: BOTH bounds (a window straddling the fit/validation boundary
    # or running past the block end belongs to neither part)
    val_m = (start >= sp.val.start) & (start + w <= sp.val.stop)
    if not fit_m.any():
        raise ValueError("no training window inside the fitting part")
    norm = ScoreNormalizer(ev.normalizer).fit(sc["recon_err"][fit_m])
    scores = dual_evidence(sc["prob"], sc["recon_err"], alpha, norm)

    if mask_params is None:
        if use_tc:   # FP analysis behind the same mask that the test split gets: [t_tc, t_tc + buffer]
            buf = pd.Timedelta(minutes=float(settle["buffer_minutes"]))
            fp_intervals = [(a, a + buf) for a, _ in intervals]
        else:
            fp_intervals = data.intervals if intervals else []
        mask_params = derive_mask_parameters(model, train.values, train.labels, train.timestamps, fp_intervals, cfg,
                                             device=device, normalizer=norm, alpha=alpha, train_scores=sc)
    if use_tc:
        mask_params = telecommand_mask_params(mask_params, settle)
    before, after = mask_buffers(cfg, mask_params)
    ts = to_naive_utc(np.asarray(train.timestamps)[end_index])
    masked = _mask(scores, ts, intervals, before, after)
    in_mask = known_event_mask_flags(ts, intervals, before, after) if intervals else np.zeros(len(ts), dtype=bool)
    min_consec, mc_source = resolve_min_consecutive(ev.min_consecutive, mask_params)
    log.info("spike filter min_consecutive = %d (%s)", min_consec, mc_source)
    if use_tc:
        log.info("known-event mask from telecommands: %d training intervals [t, t + %.0f min] of %d selected types "
                 "cover %.3f %% of the training windows", len(intervals), after, len(settle["selected_types"]),
                 100.0 * float(in_mask.mean()))

    if bool(y[val_m].any()):
        thr, val_f1 = select_threshold_filtered(masked[val_m], y[val_m], ev.threshold_method, min_consec)
        source = "validation"
    elif ev.allow_train_threshold_fallback:
        log.warning("validation block has no positive window: threshold fitted on the FITTING-part scores "
                    "(threshold_source='train_fallback'; optimistic, flagged in the report)")
        thr, val_f1 = select_threshold_filtered(masked[fit_m], y[fit_m], ev.threshold_method, min_consec)
        source = "train_fallback"
    else:
        raise ValueError("The validation block has no positive window, so the F1-optimal threshold is undefined. "
                         "Use a longer training split / larger train.val_fraction, or set "
                         "eval.allow_train_threshold_fallback: true in the config YAML (evaluate.py also accepts "
                         "--allow-train-threshold-fallback) to fit it on the fitting part (flagged).")
    thr_before_floor = float(thr)
    nominal = (y == 0) & ~in_mask       # normal training windows that the mask leaves to the detector
    if not nominal.any():
        nominal = fit_m & (y == 0) & ~in_mask
    thr, floor_info = apply_threshold_floor(thr, ev.threshold_floor, masked[nominal], y[nominal])
    if floor_info["applied"] and source == "validation":
        val_f1 = _filtered_f1(masked[val_m], y[val_m], thr, min_consec)
    log.info("threshold %.4f (%s, %s, F1 %.4f after spike filter) | validation windows %d (%d positive)", thr,
             source, ev.threshold_method, val_f1, int(val_m.sum()), int(y[val_m].sum()))
    # diagnostic of the idealized closed-form threshold (never used for the deployed threshold)
    link = link_report(sc["prob"][val_m], y[val_m], cfg.train.pos_weight, cfg.train.label_smoothing)
    link["classification"] = cfg.train.classification
    if isinstance(mask_params, dict):
        mask_params["mask_buffer_before_minutes"] = before
        mask_params["mask_buffer_after_minutes"] = after
    return {
        "threshold": float(thr), "threshold_before_floor": thr_before_floor, "threshold_floor": floor_info,
        "threshold_link": link, "mask_source": ev.mask_source if ev.use_known_event_mask else "none",
        "mask_coverage_train_windows": float(in_mask.mean()), "threshold_source": source, "threshold_method": ev.threshold_method,
        "normalizer": norm.state_dict(), "alpha_rec": float(alpha), "mask_params": mask_params,
        "mask_buffer_before_minutes": before, "mask_buffer_after_minutes": after,
        "min_consecutive": int(min_consec), "min_consecutive_source": mc_source,
        "threshold_selection_f1": float(val_f1), "known_event_mask": bool(intervals),
        "n_known_events": len(intervals), "train_ratio": float(cfg.data.train_ratio), "test_stride": int(stride),
        "window_size": int(w),
        "validation": {**sp.report, "n_fit_windows": int(fit_m.sum()), "n_val_windows": int(val_m.sum()),
                       "n_val_pos_windows": int(y[val_m].sum()), "n_fit_pos_windows": int(y[fit_m].sum())},
        "n_train_windows": int(len(y)), "data_identity": data_identity(train),
    }


def event_wise_report(enabled: bool, y_true, y_pred, valid, timestamps, event_kwargs: dict | None = None) -> dict:
    """Official ESA-ADB event-wise evaluation, or an explanation why it was not computed."""
    if not enabled:
        log.info("event-wise ESA-ADB score not computed (not requested)")
        return {"computed": False, "reason": "not requested"}
    from atandn.evaluation.esa_adb import evaluate_event_wise

    try:
        res = evaluate_event_wise(np.asarray(y_true)[valid], np.asarray(y_pred)[valid], np.asarray(timestamps)[valid],
                                  **(event_kwargs or {}))
    except ImportError as exc:
        log.warning("official ESA-ADB package not installed; event-wise score NOT computed (%s)", str(exc).split(".")[0])
        return {"computed": False, "reason": "official ESA-ADB package not installed", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - the official entry point is external
        log.warning("official ESA-ADB scorer failed; event-wise score NOT computed: %s", exc)
        return {"computed": False, "reason": f"scorer failed: {exc}", "error": str(exc)}
    return {"computed": True, "result": res}


def evaluate_mission(model: torch.nn.Module, data: ProtocolData, cfg: MissionConfig,
                     mask_params: dict | None = None, *, frozen: dict | None = None, device: str = "auto",
                     esa_adb: bool = False, event_kwargs: dict | None = None, return_arrays: bool = False,
                     report_extra: dict | None = None, report_unmasked: bool = False) -> dict:
    """Evaluate a trained model with the shared protocol; the test split is scored exactly once.

    Args:
        model: trained model.
        data: :class:`ProtocolData` (see :func:`prepare_protocol_data`).
        cfg: config of this model (window size, stride, ``eval.*``). With ``frozen`` it must be consistent with it.
        mask_params: optional precomputed mask parameters (ignored when ``frozen`` is given).
        frozen: result of :func:`fit_frozen_protocol` (e.g. stored in the checkpoint by ``train.py``); ``None``
            fits it now from the training split.
        esa_adb / event_kwargs: also run the official ESA-ADB event-wise scorer.
        return_arrays: add ``"_arrays"`` (scores, predictions, ...) to the result (in-memory only).
        report_extra: extra top-level keys merged into the report (e.g. ``used_legacy_defaults``).
        report_unmasked: also report the metrics WITHOUT Stage 3 (no known-event mask, no spike filter) next to the
            ones with Stage 3, under ``stage3_comparison``; the same frozen threshold is used (it is not re-tuned for
            the unmasked chain, so the unmasked numbers are not an optimised raw-alarm operating point).

    Returns:
        JSON-able report with ``threshold``, ``threshold_source``, ``alpha_rec``, ``normalizer``,
        ``postprocessing``, ``validation``, counts (``n_test_windows``, ``n_test_pos_windows``,
        ``n_test_pos_samples``, ``n_test_events``), ``window_metrics``, ``pointwise_metrics_rule_A``,
        ``esa_adb_event_wise``, ``threshold_floor``, ``threshold_link`` (labelled diagnostic, never used for the
        deployed threshold) and, with ``report_unmasked``, ``stage3_comparison``.
    """
    w, stride, ev, test = cfg.model.window_size, cfg.data.test_stride, cfg.eval, data.test
    if test is None:
        raise ValueError("evaluate_mission needs data.test")
    if frozen is None:
        frozen = fit_frozen_protocol(model, data, cfg, mask_params, device)
    elif frozen["window_size"] != w or frozen["test_stride"] != stride:
        raise ValueError(f"frozen protocol (window {frozen['window_size']}, stride {frozen['test_stride']}) does not "
                         f"match the config (window {w}, stride {stride})")
    ident = frozen.get("data_identity")
    if ident is not None:
        now = data_identity(data.train)
        if now != ident:
            log.warning("the TRAINING data loaded now differs from the data the protocol was frozen on (%s vs %s): "
                        "threshold/normalizer/mask may not belong to this data", {k: v for k, v in now.items()
                                                                                  if k != "sha1"},
                        {k: v for k, v in ident.items() if k != "sha1"})
    mask_source = frozen.get("mask_source", "rare_events")        # checkpoints of the earlier protocol: rare events
    # operational schedule (telecommands): execution times of the training-selected types in the test period; only
    # times and types are used (no telemetry value, no label of the test split)
    sel = (frozen.get("mask_params") or {}).get("telecommand", {}).get("selected_types", [])
    intervals = known_event_intervals(cfg, rare_intervals=data.intervals, telecommands=data.telecommands,
                                      selected_types=sel, train_end=to_naive_utc(data.train.timestamps)[-1],
                                      test_end=to_naive_utc(test.timestamps)[-1], period="test",
                                      mask_source=mask_source, enabled=bool(frozen["known_event_mask"]))
    norm = ScoreNormalizer.from_state_dict(frozen["normalizer"])
    alpha, thr = float(frozen["alpha_rec"]), float(frozen["threshold"])

    # ---- the test split is scored once, with frozen values only -----------------------------------------------
    te = score_stream(model, test.values, w, stride, ev.batch_size, device)
    raw = dual_evidence(te["prob"], te["recon_err"], alpha, norm)
    ts = to_naive_utc(np.asarray(test.timestamps)[te["end_index"]])
    scores = _mask(raw, ts, intervals, frozen["mask_buffer_before_minutes"], frozen["mask_buffer_after_minutes"])
    pred = min_consecutive_filter((scores > thr).astype(int), frozen["min_consecutive"])
    masked_share = float(np.mean(scores != raw)) if len(raw) else 0.0     # windows whose score the mask zeroed

    # ---- test labels are used only from here on, for metrics and descriptive counts ---------------------------
    y_win = window_labels(test.labels, te["start"], w)
    n_events = count_events(test.labels)
    n_pos_samples = int((np.asarray(test.labels) > 0).sum())
    if n_events < MIN_TEST_EVENTS_WARN:
        log.warning("the test split has only %d anomalous event(s) (%d positive samples): metrics are statistically "
                    "fragile (< %d events)", n_events, n_pos_samples, MIN_TEST_EVENTS_WARN)
    win = window_metrics(y_win, pred)
    sample_pred, valid = windows_to_samples(pred, te["end_index"], len(test), stride, w)
    point = pointwise_metrics(test.labels, sample_pred, valid)
    point_pa = point_adjusted_metrics(np.asarray(test.labels)[valid], sample_pred[valid])
    event = event_wise_report(esa_adb, np.asarray(test.labels), sample_pred, valid, test.timestamps, event_kwargs)

    comparison = None
    if report_unmasked:
        pred_raw = (raw > thr).astype(int)                       # same frozen threshold; no mask, no spike filter
        sp_raw, valid_raw = windows_to_samples(pred_raw, te["end_index"], len(test), stride, w)
        point_pa_raw = point_adjusted_metrics(np.asarray(test.labels)[valid_raw], sp_raw[valid_raw])
        comparison = {
            "with_stage3": {"window_metrics": win, "pointwise_metrics_rule_A": point, "point_adjusted_metrics": point_pa},
            "without_stage3": {"window_metrics": window_metrics(y_win, pred_raw),
                               "pointwise_metrics_rule_A": pointwise_metrics(test.labels, sp_raw, valid_raw),
                               "point_adjusted_metrics": point_pa_raw},
            "note": ("'without Stage 3' = no known-event mask and no spike filter, evaluated with the SAME frozen "
                     "threshold; the threshold is not re-tuned for the unmasked chain"),
        }
    mp = frozen.get("mask_params") or {}
    val = frozen["validation"]
    report: dict[str, Any] = {
        "mission": cfg.name,
        "protocol": {
            "validation_block_from_training_split": True, "threshold_source": frozen["threshold_source"],
            "threshold_method": frozen["threshold_method"], "normalizer_source": "training (fitting part)",
            "threshold_floor": (frozen.get("threshold_floor") or {}).get("mode", "none"),
            "mask_buffer_source": "training", "test_labels_used_for_fitting": False,
            "point_adjustment": bool(getattr(ev, "point_adjustment", True)),
            "test_split_scored_once": True, "window_size": w, "test_stride": stride,
            "train_ratio": frozen["train_ratio"],
        },
        "threshold": thr, "threshold_source": frozen["threshold_source"], "alpha_rec": alpha,
        "normalizer": frozen["normalizer"],
        "postprocessing": {
            "known_event_mask": bool(intervals), "n_known_events": len(intervals),
            "mask_source": mask_source if frozen["known_event_mask"] else "none",
            "mask_share_of_test_windows": masked_share,
            "telecommand_selected_types": (mp.get("telecommand") or {}).get("selected_types"),
            "mask_buffer_before_minutes": frozen["mask_buffer_before_minutes"],
            "mask_buffer_after_minutes": frozen["mask_buffer_after_minutes"],
            "min_consecutive": int(frozen["min_consecutive"]),
            "min_consecutive_source": frozen.get("min_consecutive_source", "config"),
            "training_recommended_min_consecutive": mp.get("min_consecutive_windows"),
        },
        "validation": val,
        "val_has_positives": bool(val.get("val_has_positives")),
        "n_train_windows": int(frozen["n_train_windows"]), "n_fit_windows": int(val["n_fit_windows"]),
        "n_val_windows": int(val["n_val_windows"]), "n_val_pos_windows": int(val["n_val_pos_windows"]),
        "n_test_windows": int(len(y_win)), "n_test_pos_windows": int(y_win.sum()),
        "n_test_pos_samples": n_pos_samples, "n_test_events": int(n_events),
        "window_metrics": win, "pointwise_metrics_rule_A": point, "point_adjusted_metrics": point_pa, "esa_adb_event_wise": event,
        "threshold_floor": frozen.get("threshold_floor", {"mode": "none", "applied": False, "floor": None}),
        "threshold_link": frozen.get("threshold_link", {"computed": False, "reason": "not in the frozen protocol"}),
    }
    if comparison is not None:
        report["stage3_comparison"] = comparison
    report.update(report_extra or {})
    if return_arrays:
        report["_arrays"] = {"test_scores": scores, "test_pred": pred, "test_window_labels": y_win,
                             "end_index": te["end_index"], "start": te["start"], "frozen": frozen}
    return report
