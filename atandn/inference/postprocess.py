"""Post-processing: spike filter and the known-event mask."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from atandn.data.splits import to_naive_utc


def min_consecutive_filter(preds: np.ndarray, min_len: int) -> np.ndarray:
    """Remove runs of positive predictions shorter than ``min_len`` (``min_len <= 1`` is a no-op)."""
    out = np.asarray(preds).astype(int).copy()
    if min_len <= 1 or out.size == 0:
        return out
    padded = np.r_[0, out, 0]
    diff = np.diff(padded)
    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        if e - s < min_len:
            out[s:e] = 0
    return out


def known_event_mask_flags(
    timestamps,
    intervals: Iterable[tuple],
    buffer_before_minutes: float = 0.0,
    buffer_after_minutes: float = 0.0,
) -> np.ndarray:
    """Boolean ``(N,)`` array: True where a timestamp lies inside an interval extended by the buffers.

    The intervals are sorted and merged, then every timestamp is located with ``searchsorted``, so the cost is
    ``O((N + M) log M)`` also for hundreds of thousands of telecommand intervals. Interval ends are inclusive.
    """
    ts = to_naive_utc(timestamps).values
    before, after = np.timedelta64(int(round(buffer_before_minutes * 60e9)), "ns"),         np.timedelta64(int(round(buffer_after_minutes * 60e9)), "ns")
    iv = list(intervals)
    if not iv or len(ts) == 0:
        return np.zeros(len(ts), dtype=bool)
    starts = to_naive_utc([a for a, _ in iv]).values
    ends = to_naive_utc([b for _, b in iv]).values
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order] - before, ends[order] + after
    ends = np.maximum.accumulate(ends)                       # running maximum of the ends: merges overlaps
    idx = np.searchsorted(starts, ts, side="right") - 1      # last interval that starts at or before t
    ok = idx >= 0
    flags = np.zeros(len(ts), dtype=bool)
    flags[ok] = ts[ok] <= ends[idx[ok]]
    return flags


def apply_known_event_mask(
    scores: np.ndarray,
    timestamps,
    intervals: Iterable[tuple],
    buffer_before_minutes: float = 0.0,
    buffer_after_minutes: float = 0.0,
) -> np.ndarray:
    """Set scores to zero inside known planned-event intervals (extended by a buffer).

    ASSUMPTION: the intervals are *planned operations known in advance* (an operational schedule: ESA ``Rare
    Event`` maneuvers, or the telecommand execution times with a settling buffer, see
    ``analysis.telecommand_buffers``), not information derived from the test labels. Buffers must be derived from
    the TRAINING split. The input is not modified.

    Args:
        scores: ``(N,)`` window scores.
        timestamps: ``(N,)`` window timestamps (last sample of each window).
        intervals: iterable of ``(start, end)`` timestamps (``start == end`` for a point event such as a telecommand,
            combined with ``buffer_after_minutes`` this is the paper's ``[t_tc, t_tc + buffer]``).
        buffer_before_minutes / buffer_after_minutes: extension of each interval.
    """
    out = np.asarray(scores, dtype=np.float64).copy()
    out[known_event_mask_flags(timestamps, intervals, buffer_before_minutes, buffer_after_minutes)] = 0.0
    return out
