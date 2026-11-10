"""Chronological splits and the :class:`MissionData` container."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


def chronological_split(n: int, train_ratio: float = 0.5) -> tuple[slice, slice]:
    """Split ``n`` ordered samples into a train prefix and a test suffix (no shuffling).

    Args:
        n: Number of samples.
        train_ratio: Fraction of samples in the training prefix.

    Returns:
        ``(train_slice, test_slice)``; ``train_slice.stop == test_slice.start == int(n * train_ratio)``.
    """
    if n <= 0 or not 0.0 < train_ratio < 1.0:
        raise ValueError("n must be > 0 and train_ratio in (0, 1)")
    cut = int(n * train_ratio)
    return slice(0, cut), slice(cut, n)


def train_val_tail_split(n_train: int, val_fraction: float, window_size: int) -> tuple[slice, slice]:
    """Hold out the chronological tail of the TRAINING split for model selection.

    A gap of ``window_size - 1`` samples is dropped between the two parts so that no window of the
    fitting part overlaps a window of the validation tail.

    Returns:
        ``(fit_slice, val_slice)`` indexing the training split.
    """
    n_val = int(n_train * val_fraction)
    gap = window_size - 1
    if n_val < window_size or n_train - n_val - gap < window_size:
        raise ValueError("Training split too short for the requested validation tail and window size")
    val_start = n_train - n_val
    return slice(0, val_start - gap), slice(val_start, n_train)


def count_events(labels: np.ndarray) -> int:
    """Number of anomalous EVENTS = contiguous runs of positive samples in ``labels``."""
    y = (np.asarray(labels) > 0).astype(np.int8)
    if y.size == 0:
        return 0
    return int(np.sum(np.diff(np.r_[0, y]) == 1))


@dataclass
class ValidationSplit:
    """Result of :func:`event_aware_validation_split`.

    Attributes:
        fit: slice of the TRAINING split used for fitting (data before the validation block minus the gap).
        val: slice of the TRAINING split used for model selection / threshold / early stopping.
        report: JSON-able description (``val_start``, ``val_stop``, ``n_val_events``, ``val_has_positives``,
            ``tail_moved``, ``min_events_satisfied`` ...).
    """

    fit: slice
    val: slice
    report: dict


def _event_runs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) sample index of every contiguous positive run of ``y``."""
    pos = np.r_[0, (np.asarray(y) > 0).astype(np.int8), 0]
    d = np.diff(pos)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def _events_per_block(y: np.ndarray, n_val: int) -> np.ndarray:
    """``out[s]`` = number of events lying ENTIRELY inside the block ``[s, s + n_val)``.

    Events clipped by either block boundary are not counted (a run that is cut at the block start or end is not a
    complete event of the block).
    """
    n = len(y)
    out_len = n - n_val + 1
    rs, re = _event_runs(y)
    diff = np.zeros(out_len + 1, dtype=np.int64)
    if len(rs):
        lo = np.maximum(re - n_val, 0)                    # first block start that still contains the whole run
        hi = np.minimum(rs, out_len - 1)                  # last block start that does
        ok = lo <= hi
        np.add.at(diff, lo[ok], 1)
        np.add.at(diff, hi[ok] + 1, -1)
    return np.cumsum(diff)[:out_len]


def event_aware_validation_split(
    labels: np.ndarray, val_fraction: float, gap: int, min_events: int = 3, min_fit: int = 0,
    min_fit_fraction: float = 0.0, warn: bool = True,
) -> ValidationSplit:
    """Carve a validation block out of the TRAINING split only (the test split is never involved).

    Default: the chronological tail (``int(n * val_fraction)`` samples). The block should hold at least
    ``min_events`` distinct anomalous events (contiguous positive runs) that lie ENTIRELY inside it (events clipped
    by a block boundary are not counted, because a clipped run is only a fragment of an event). The search is
    progressive: for ``m = min_events, min_events - 1, ..., 1`` the tail is accepted if it has ``>= m`` full events,
    otherwise the LATEST block of the same length with ``>= m`` full events is used (its fit part is only the data
    BEFORE it minus ``gap`` samples; ``gap = window_size - 1`` keeps windows of both parts disjoint). Every
    relaxation of ``m`` is logged as a WARNING and the count actually reached is stored in the report
    (``val_events`` = full events of the chosen block, ``min_events_used``). A moved block must leave a fit part of
    at least ``max(min_fit, min_fit_fraction * n)`` samples. If no block holds even one full event the tail is kept
    (WARNING) and the report says whether it contains any positive sample (``val_has_positives``).

    Args:
        labels: ``(n,)`` sample labels of the training split.
        val_fraction: fraction of the training split used for validation.
        gap: samples dropped between the fitting part and the block.
        min_events: wanted number of distinct full events in the block.
        min_fit: minimum length of the fitting part of a moved block, in samples.
        min_fit_fraction: minimum length of the fitting part of a moved block as a fraction of ``n``.
        warn: log the fallback messages as WARNING (False: DEBUG; used to log once per window size in a search).

    Returns:
        :class:`ValidationSplit`.
    """
    _warn = log.warning if warn else log.debug
    y = np.asarray(labels)
    n = len(y)
    n_val = int(n * val_fraction)
    if n_val < 1 or n - n_val - gap < max(min_fit, 1):
        raise ValueError("Training split too short for the requested validation block and window size")
    tail = n - n_val
    counts = _events_per_block(y, n_val)
    lo = gap + max(min_fit, 1, int(np.ceil(min_fit_fraction * n)))   # earliest start of a MOVED block
    start, moved, used = tail, False, 0
    for m in range(max(int(min_events), 1), 0, -1):
        if counts[tail] >= m:
            used = m
            break
        ok = np.where(counts[lo:tail] >= m)[0] if lo < tail else np.empty(0, dtype=np.int64)
        if ok.size:
            start, moved, used = int(lo + ok[-1]), True, m
            _warn("validation tail [%d, %d) has %d full event(s) (< %d); using the latest block [%d, %d) with "
                        "%d full event(s); fitting uses samples [0, %d) only", tail, n, int(counts[tail]), m, start,
                        start + n_val, int(counts[start]), start - gap)
            break
        _warn("no validation block of %d samples with >= %d full event(s) and a fit part >= %d samples "
                    "exists%s", n_val, m, lo - gap, "; relaxing to %d" % (m - 1) if m > 1 else "")
    if used == 0:
        _warn("no validation block holds a full anomalous event; keeping the tail [%d, %d)", tail, n)
    elif used < min_events:
        _warn("validation block holds %d full event(s), fewer than the wanted %d (min_events relaxed)", used,
                    min_events)
    stop = start + n_val
    n_full = int(counts[start])
    has_pos = bool((y[start:stop] > 0).any())
    if not has_pos:
        _warn("validation block [%d, %d) has NO positive sample: selection falls back to the validation loss "
                    "and the threshold is undefined", start, stop)
    report = {"val_start": int(start), "val_stop": int(stop), "fit_stop": int(start - gap), "gap": int(gap),
              "val_events": n_full, "n_val_events": n_full, "n_val_events_incl_clipped": int(count_events(y[start:stop])),
              "min_events": int(min_events), "min_events_used": int(used), "val_has_positives": has_pos,
              "tail_moved": moved, "min_events_satisfied": bool(used >= min_events), "n_train_samples": int(n),
              "min_fit_fraction": float(min_fit_fraction)}
    return ValidationSplit(slice(0, start - gap), slice(start, stop), report)


@dataclass
class MissionData:
    """A multichannel stream with per-sample labels.

    Attributes:
        values: ``(N, C)`` float array.
        labels: ``(N,)`` 0/1 array (1 = anomalous sample).
        timestamps: ``(N,)`` datetime index/array (or any ordered array).
        channel_names: Names of the ``C`` channels.
        meta: Free-form metadata (e.g. ``known_event_intervals`` for ESA missions).
    """

    values: np.ndarray
    labels: np.ndarray
    timestamps: Any
    channel_names: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def slice(self, sl: slice) -> "MissionData":
        """Return the sub-stream ``sl`` (metadata is shared)."""
        return MissionData(self.values[sl], self.labels[sl], self.timestamps[sl], self.channel_names, self.meta)

    def split(self, train_ratio: float = 0.5) -> tuple["MissionData", "MissionData"]:
        """Chronological ``(train, test)`` split."""
        tr, te = chronological_split(len(self), train_ratio)
        return self.slice(tr), self.slice(te)


def to_naive_utc(ts: Any) -> pd.DatetimeIndex:
    """Convert timestamps to a timezone-naive UTC ``DatetimeIndex``."""
    idx = pd.DatetimeIndex(pd.to_datetime(ts, **_parse_kwargs(ts)))
    return idx.tz_convert(None) if idx.tz is not None else idx


def _parse_kwargs(ts: Any) -> dict:
    """Explicit parsing options for string timestamps (avoids the 'Could not infer format' warning).

    ``format="mixed"`` parses every element on its own (the previous implicit behaviour); non-string input and
    pandas < 2 are left untouched. Naive strings stay naive (``utc`` is not forced), so results are unchanged.
    """
    if int(pd.__version__.split(".")[0]) < 2:
        return {}
    arr = np.asarray(ts, dtype=object).ravel() if not isinstance(ts, (pd.Series, pd.Index)) else ts
    if len(arr) and isinstance(arr[0] if not hasattr(arr, "iloc") else arr.iloc[0], str):
        return {"format": "mixed"}
    return {}
