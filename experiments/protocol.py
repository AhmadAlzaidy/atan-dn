"""Shared honest train/evaluate protocol used by the ablation and multi-seed runners.

One call of :func:`train_and_evaluate` does, for a single seed:

1. carve the validation block out of the TRAINING split (event-aware, ``atandn.data.splits``) and train on the data
   before it; early stopping / model selection use the threshold-free average precision on the block;
2. hand the selected model to ``atandn.evaluation.pipeline.evaluate_mission``, THE SAME function ``scripts/evaluate.py``
   uses: normalizer (fitting part), mask parameters (derived from the training split for this very model),
   threshold (F1-optimal grid on the validation block), known-event mask, spike filter, test split scored once;
3. flatten the report (window / point / event metrics, counts, threshold, ...);
4. descriptive feature diagnostics (excess kurtosis after the first normalization layer, Fisher criterion of the
   latent features) on a bounded subsample of test windows. They use test labels only to describe class separation
   and feed nothing back.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from atandn.config import MissionConfig
from atandn.data import (MissionData, WindowDataset, iter_window_batches, load_mission_from_config, window_labels,
                         window_starts)
from atandn.evaluation import count_parameters, evaluate_mission, prepare_protocol_data
from atandn.models import ATANDN
from atandn.training import apply_params
from atandn.training.fit import fit_on_split
from atandn.utils.io import resolve_device
from atandn.utils.seed import set_seed

log = logging.getLogger(__name__)

METRIC_KEYS = ("precision", "recall", "fpr", "fnr", "f1", "f0_5")
PAPER_SEEDS = (100, 777, 2026)
_PARAM_KEYS = ("window_size", "overlap", "pos_weight", "lr", "recon_weight", "dropout", "batch_size", "focal_gamma",
               "weight_decay", "recon_score_weight", "num_heads", "conv_f1", "conv_f2", "use_weighted_sampling",
               "use_augmentation")


# ---------------------------------------------------------------------------------------------------------
# configuration / data helpers
# ---------------------------------------------------------------------------------------------------------
def current_params(cfg: MissionConfig) -> dict[str, Any]:
    """Hyperparameters of ``cfg`` in the key space of ``atandn.training.search.suggest_params``."""
    m, t = cfg.model, cfg.train
    return {"window_size": m.window_size, "overlap": t.overlap, "pos_weight": t.pos_weight, "lr": t.lr,
            "recon_weight": t.recon_weight, "dropout": m.dropout, "batch_size": t.batch_size,
            "focal_gamma": t.focal_gamma, "weight_decay": t.weight_decay,
            "recon_score_weight": t.recon_score_weight, "num_heads": m.num_heads, "conv_f1": m.conv_filters[0],
            "conv_f2": m.conv_filters[1], "use_weighted_sampling": t.use_weighted_sampling,
            "use_augmentation": t.use_augmentation}


def load_best_params(path: str | Path, base: MissionConfig) -> MissionConfig:
    """Freeze the hyperparameters stored in a best-params JSON on top of ``base``.

    Accepted layouts: a flat mapping of search-space keys (``{"lr": ..., "dropout": ...}``), or a mapping with
    the parameters under ``"params"``, ``"best_params"`` or ``"best_trial": {"params": ...}`` (an Optuna dump).
    Keys that are absent keep the value of ``base``; unknown keys raise ``ValueError``.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("best_trial"), dict) and "params" in raw["best_trial"]:
            raw = raw["best_trial"]["params"]
        else:
            for key in ("params", "best_params"):
                if isinstance(raw.get(key), dict):
                    raw = raw[key]
                    break
    if not isinstance(raw, dict):
        raise ValueError("best-params JSON must contain a mapping of hyperparameters")
    legacy_keys = {"latent_dim"}
    unknown = set(raw) - set(_PARAM_KEYS) - legacy_keys
    if unknown:
        raise ValueError(f"Unknown hyperparameter keys in {path}: {sorted(unknown)}; expected {list(_PARAM_KEYS)}")
    clean_raw = {k: v for k, v in raw.items() if k in _PARAM_KEYS}
    return apply_params(base, {**current_params(base), **clean_raw})


def load_mission_data(cfg: MissionConfig, data_root: str | Path) -> tuple[MissionData, MissionData]:
    """Load the mission described by ``cfg.data`` and return the chronological ``(train, test)`` split."""
    md = load_mission_from_config(cfg, data_root, parquet_in_root=cfg.data.kind == "esa")
    return md.split(cfg.data.train_ratio)


def build_model(cfg: MissionConfig) -> ATANDN:
    """Instantiate ``ATANDN`` from ``cfg.model``."""
    kw = asdict(cfg.model)
    kw["conv_filters"] = tuple(cfg.model.conv_filters)
    return ATANDN(**kw)


# ---------------------------------------------------------------------------------------------------------
# feature diagnostics
# ---------------------------------------------------------------------------------------------------------
def excess_kurtosis(y: np.ndarray) -> float:
    """Excess kurtosis ``E[(y-mu)^4] / sigma^4 - 3`` (population moments); NaN for a constant sample."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    d = y - y.mean()
    var = float(np.mean(d ** 2))
    if var <= 1e-24:
        return float("nan")
    return float(np.mean(d ** 4) / var ** 2 - 3.0)


def mean_channel_kurtosis(act: np.ndarray) -> float:
    """Mean over channels of the excess kurtosis of activations ``(B, C, T)`` pooled over batch and time."""
    c = act.shape[1]
    flat = np.moveaxis(act, 1, 0).reshape(c, -1)
    vals = [excess_kurtosis(flat[i]) for i in range(c)]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def fisher_criterion(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean over features of ``J = (mean1 - mean0)^2 / (var1 + var0)`` for features ``(N, D)``.

    Class 1 is the anomalous class. NaN if one class is absent; features with zero total variance are skipped.
    """
    f = np.asarray(features, dtype=np.float64)
    pos = np.asarray(labels).astype(bool)
    if not pos.any() or pos.all():
        return float("nan")
    f1, f0 = f[pos], f[~pos]
    den = f1.var(axis=0) + f0.var(axis=0)
    num = (f1.mean(axis=0) - f0.mean(axis=0)) ** 2
    ok = den > 1e-24
    return float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")


@torch.no_grad()
def feature_diagnostics(model: ATANDN, X: np.ndarray, y: np.ndarray, device: str | torch.device = "auto",
                        batch_size: int = 256) -> dict[str, float]:
    """Excess kurtosis after the first encoder norm layer and Fisher criterion of the latent features.

    The norm layer is ``model.encoder_blocks[0][2]`` (``nn.Identity`` for the identity variant, so the value is
    then the kurtosis of the BatchNorm output that feeds it). The latent is the bottleneck output, flattened.
    """
    dev = resolve_device(device)
    model.to(dev).eval()
    norm_out: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    h1 = model.encoder_blocks[0][2].register_forward_hook(lambda m, i, o: norm_out.append(o.detach().cpu().numpy()))
    h2 = model.bottleneck.register_forward_hook(lambda m, i, o: latent.append(o.detach().flatten(1).cpu().numpy()))
    try:
        for i in range(0, len(X), batch_size):
            model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=dev))
    finally:
        h1.remove()
        h2.remove()
    if not norm_out:
        return {"kurtosis_norm": float("nan"), "fisher_latent": float("nan")}
    return {"kurtosis_norm": mean_channel_kurtosis(np.concatenate(norm_out)),
            "fisher_latent": fisher_criterion(np.concatenate(latent), y)}


# ---------------------------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------------------------
def _flatten_numeric(d: dict, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def _diagnostic_windows(test: MissionData, window_size: int, stride: int, max_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """At most ``max_windows`` test windows ``(X, y)`` (bounded memory), preserving anomaly windows."""
    starts = window_starts(len(test), window_size, stride)
    if len(starts) == 0:
        return np.zeros((0, test.values.shape[1], window_size), np.float32), np.zeros(0, np.int64)
    labels = window_labels(test.labels, starts, window_size)
    if len(starts) > max_windows:
        pos_idx = np.where(labels > 0)[0]
        neg_idx = np.where(labels == 0)[0]
        n_pos = len(pos_idx)
        if n_pos >= max_windows:
            sel_pos = pos_idx[np.linspace(0, n_pos - 1, max_windows).astype(np.int64)]
            sel_neg = np.array([], dtype=np.int64)
        else:
            sel_pos = pos_idx
            n_neg_needed = max_windows - n_pos
            if len(neg_idx) > n_neg_needed:
                sel_neg = neg_idx[np.linspace(0, len(neg_idx) - 1, n_neg_needed).astype(np.int64)]
            else:
                sel_neg = neg_idx
        sel = np.sort(np.concatenate([sel_pos, sel_neg]))
        starts = starts[sel]
    if len(starts) == 0:
        return np.zeros((0, test.values.shape[1], window_size), np.float32), np.zeros(0, np.int64)
    X = np.concatenate(list(iter_window_batches(test.values, starts, window_size, 512)))
    return X, window_labels(test.labels, starts, window_size)


def flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    """Flat row (``window_*``, ``point_*``, ``event_*``, counts, threshold, ...) of an ``evaluate_mission`` report."""
    row: dict[str, Any] = {}
    row.update({f"window_{k}": v for k, v in report["window_metrics"].items()})
    row.update({f"point_{k}": v for k, v in report["pointwise_metrics_rule_A"].items()})
    if "point_adjusted_metrics" in report:
        row.update({f"pa_{k}": v for k, v in report["point_adjusted_metrics"].items()})
    ev = report["esa_adb_event_wise"]
    if ev.get("computed"):
        res = ev["result"]
        row.update(_flatten_numeric(res if isinstance(res, dict) else {}, "event_"))
    elif "error" in ev:
        row["event_wise_error"] = ev["error"]
    val, norm, post = report["validation"], report["normalizer"], report["postprocessing"]
    row.update({
        "threshold": report["threshold"], "threshold_source": report["threshold_source"],
        "threshold_method": report["protocol"]["threshold_method"], "alpha_rec": report["alpha_rec"],
        "norm_lo": norm["lo"], "norm_hi": norm["hi"],
        "known_event_mask": post["known_event_mask"], "mask_buffer_before_minutes": post["mask_buffer_before_minutes"],
        "mask_buffer_after_minutes": post["mask_buffer_after_minutes"], "min_consecutive": post["min_consecutive"],
        "val_has_positives": bool(val.get("val_has_positives")), "val_tail_moved": bool(val.get("tail_moved")),
        "n_val_events": val.get("n_val_events"), "n_val_pos_windows": report["n_val_pos_windows"],
        "n_train_windows_scored": report["n_train_windows"], "n_test_windows": report["n_test_windows"],
        "n_test_pos_windows": report["n_test_pos_windows"], "n_test_pos_samples": report["n_test_pos_samples"],
        "n_test_events": report["n_test_events"],
    })
    return row


def train_and_evaluate(cfg: MissionConfig, train: MissionData, test: MissionData, seed: int,
                       device: str = "auto", event_wise: bool = False,
                       event_kwargs: dict[str, Any] | None = None,
                       keep_artifacts: bool = False, report_extra: dict[str, Any] | None = None,
                       diag_max_windows: int = 2000) -> dict[str, Any]:
    """Train one model and evaluate it with the shared protocol (see the module docstring).

    Args:
        cfg: validated mission config (hyperparameters already frozen/overridden).
        train / test: chronological split of the stream (``train.meta["known_event_intervals"]`` feeds the mask).
        seed: seed for ``set_seed`` (also stored in ``cfg.train.seed`` of the run).
        device: ``"auto"`` or a torch device string.
        event_wise: also call the official ESA-ADB scorer (ImportError is caught and recorded).
        event_kwargs: forwarded to ``evaluate_event_wise`` (``module``, ``attribute``, ``scorer``, ...).
        keep_artifacts: if True the result has an ``"_artifacts"`` entry (model, report with arrays).
        report_extra: extra keys merged into the row (e.g. ``used_legacy_defaults``).
        diag_max_windows: cap of test windows used for the feature diagnostics.

    Returns:
        flat dict with ``window_<metric>``, ``point_<metric>``, ``event_*``, ``threshold``, ``threshold_source``,
        ``alpha_rec``, ``norm_lo/hi``, test counts (windows, positive windows/samples, events), ``n_params``,
        ``kurtosis_norm``, ``fisher_latent`` and bookkeeping fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg.train.seed = int(seed)
    cfg.validate()
    t0 = time.time()
    set_seed(int(seed))
    w = cfg.model.window_size

    data = prepare_protocol_data(train, test, cfg)
    model, fit_out = fit_on_split(cfg, train, data.split, device)

    report = evaluate_mission(model, data, cfg, device=device, esa_adb=event_wise, event_kwargs=event_kwargs,
                              return_arrays=keep_artifacts)
    row: dict[str, Any] = {"seed": int(seed), "status": "ok"}
    row.update(flatten_report(report))
    X, y = _diagnostic_windows(test, w, cfg.data.test_stride, diag_max_windows)
    row.update(feature_diagnostics(model, X, y, device))
    row.update({
        "n_params": count_parameters(model), "best_epoch": fit_out["best_epoch"], "epochs_run": fit_out["epochs_run"],
        "best_val_score": fit_out["best_score"], "has_val_positives": fit_out["has_val_positives"],
        "val_metric": fit_out["val_metric"], "seconds": round(time.time() - t0, 2),
    })
    row.update(report_extra or {})
    if keep_artifacts:
        row["_artifacts"] = {"model": model, "report": report}
    return row


# ---------------------------------------------------------------------------------------------------------
# aggregation / IO
# ---------------------------------------------------------------------------------------------------------
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def aggregate(rows: Iterable[dict[str, Any]], group_key: str | None = "variant", ddof: int = 1) -> dict[str, dict[str, dict]]:
    """Mean/std of every numeric column per group (``group_key``; None -> one group named ``"all"``).

    Only rows with ``status == "ok"`` are used. Returns ``{group: {column: {"mean", "std", "n"}}}``; NaN values
    are ignored per column. ``std`` is 0.0 when fewer than ``ddof + 1`` values exist.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status", "ok") != "ok":
            continue
        groups.setdefault(str(r[group_key]) if group_key else "all", []).append(r)
    out: dict[str, dict[str, dict]] = {}
    for g, rs in groups.items():
        cols = sorted({k for r in rs for k, v in r.items() if _is_num(v) and k != "seed"})
        stats: dict[str, dict] = {}
        for c in cols:
            vals = np.array([float(r[c]) for r in rs if c in r and _is_num(r[c])], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            std = float(vals.std(ddof=ddof)) if vals.size > ddof else 0.0
            stats[c] = {"mean": float(vals.mean()), "std": std, "n": int(vals.size)}
        out[g] = stats
    return out


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write rows (union of keys, first-seen order, private ``_`` keys dropped) to a CSV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols and not k.startswith("_"):
                cols.append(k)
    with p.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    return p


def read_rows_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV written by :func:`write_rows_csv`; numeric-looking cells become floats."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in r.items():
                if v == "" or v is None:
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``row`` without keys starting with an underscore (in-memory artifacts)."""
    return {k: v for k, v in row.items() if not k.startswith("_")}
