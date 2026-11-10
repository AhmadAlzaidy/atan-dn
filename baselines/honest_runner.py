"""Helpers of the honest mode of the ``*_custom.py`` baseline runners (``--split-json`` + ``--export-scores``).

Honest mode: the runner trains on the FIT slice of a split file only (the scaler is fitted on it too), then scores
windows of the fit, validation and test slices at their own window ends and writes the raw scores to an ``.npz``.
It computes NO threshold, NO metric and reads NO label (labels are not even passed on; windows get zero labels).
Threshold, normalisation, known-event mask, spike filter and metrics are applied afterwards, identically for all
models, by ``baselines/protocol_adapter.py`` (CLI ``scripts/evaluate_baseline_scores.py``).

The legacy test-label threshold sweep of the runners stays reachable only through ``--legacy-test-label-threshold``
(it prints a warning); without any of the flags a runner refuses to start.

Only ``numpy`` is imported at module level (the runners bring ``torch``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import split_io  # noqa: E402

LEGACY_WARNING = ("WARNING: --legacy-test-label-threshold: the decision threshold is chosen by sweeping the TEST scores "
                  "against the TEST labels (optimistic oracle, not the honest protocol); the reported numbers are not "
                  "comparable with ATAN-DN. Use --split-json/--export-scores and scripts/evaluate_baseline_scores.py "
                  "for the honest protocol.")


def add_honest_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--split-json``, ``--export-scores``, ``--score-stride`` and ``--legacy-test-label-threshold``."""
    parser.add_argument("--split-json", default=None,
                        help="split file from scripts/export_split.py: train on its FIT slice only")
    parser.add_argument("--export-scores", default=None,
                        help="write per-window scores (fit/val/test) with end indices to this .npz; no threshold and "
                             "no test label is used in this mode")
    parser.add_argument("--score-stride", type=int, default=None,
                        help="stride between scored windows (default: the test_stride of the split file, 10)")
    parser.add_argument("--legacy-test-label-threshold", action="store_true",
                        help="LEGACY: sweep the threshold on the test scores with the test labels (optimistic oracle)")


def resolve_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """``"honest"`` or ``"legacy"``; calls ``parser.error`` for inconsistent flag combinations."""
    if args.legacy_test_label_threshold:
        if args.export_scores or args.split_json:
            parser.error("--legacy-test-label-threshold cannot be combined with --split-json/--export-scores")
        print(LEGACY_WARNING)
        return "legacy"
    if not (args.split_json and args.export_scores):
        parser.error("give both --split-json and --export-scores (honest protocol), or --legacy-test-label-threshold "
                     "for the old oracle-threshold behaviour")
    return "honest"


def load_honest_inputs(args: argparse.Namespace, kind: str) -> tuple[np.ndarray, dict, int]:
    """Load the stream and the split, check that they belong together.

    Returns:
        ``(values (N, C) float32, split dict, score stride)``. The labels of the stream are discarded here.
    """
    split = split_io.read_split(args.split_json)
    values, _labels, ts = split_io.load_stream(args.data, kind)
    del _labels                                   # honest mode never looks at a label
    split_io.check_stream_matches_split(split, ts)
    stride = int(args.score_stride or split.get("test_stride", 10))
    if stride < 1:
        raise ValueError("--score-stride must be >= 1")
    return values, split, stride


def scale_with_fit(values: np.ndarray, split: dict) -> np.ndarray:
    """Standardise the whole stream with the mean/std of the FIT slice only (population std, like ``StandardScaler``)."""
    a, b = split_io.part_range(split, "fit")
    fit = values[a:b].astype(np.float64)
    mu, sd = fit.mean(axis=0), fit.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return ((values - mu) / sd).astype(np.float32)


def make_windows(data: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """Stack windows ``data[s : s + window_size]`` for the absolute ``starts`` into ``(n, window_size, C)``."""
    if len(starts) == 0:
        return np.zeros((0, window_size, data.shape[1]), dtype=np.float32)
    view = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)      # (N - w + 1, C, w)
    return np.ascontiguousarray(np.swapaxes(view[starts], 1, 2), dtype=np.float32)


def score_parts(score_fn: Callable[[np.ndarray], np.ndarray], scaled: np.ndarray, split: dict, window_size: int,
                stride: int, parts: tuple[str, ...] = split_io.PARTS,
                chunk: int | None = 4096) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Score the windows of each part that lie entirely inside it, at ``stride``.

    Args:
        score_fn: ``(B, window_size, C)`` float32 windows (scaled with the FIT statistics) -> ``(B,)`` scores. It must
            use only the window itself (causal) and never any label.
        chunk: windows per ``score_fn`` call (``None``: all windows of a part at once).

    Returns:
        ``{part: (scores, end_index)}`` with ABSOLUTE end indices (``start + window_size - 1``).
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for part in parts:
        a, b = split_io.part_range(split, part)
        starts = split_io.window_starts_in(a, b, window_size, stride)
        res = []
        step = chunk or max(len(starts), 1)
        for i in range(0, len(starts), step):
            res.append(np.asarray(score_fn(make_windows(scaled, starts[i:i + step], window_size)),
                                  dtype=np.float64).reshape(-1))
        scores = np.concatenate(res) if res else np.zeros(0)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"part '{part}': the baseline produced {int((~np.isfinite(scores)).sum())} non-finite "
                             "scores")
        out[part] = (scores, starts + window_size - 1)
    return out


def export_scores(args: argparse.Namespace, split: dict, parts: dict, window_size: int, stride: int, model: str,
                  mission: str) -> Path:
    """Write the ``.npz`` requested with ``--export-scores``."""
    return split_io.save_scores(args.export_scores, parts, window_size=window_size, stride=stride, model=model,
                                mission=mission, ts_sha1=split["timestamps"]["sha1"])
