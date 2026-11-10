"""Loaders of the OPS-SAT ``segments.csv``: the legacy interleaved stream and the multichannel pivot."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from atandn.data.splits import MissionData, _parse_kwargs, to_naive_utc

log = logging.getLogger(__name__)


def _row_labels(df: pd.DataFrame) -> np.ndarray:
    """0/1 label per row from the ``anomaly`` column (strings are mapped with ``== "anomaly"``)."""
    anomaly = df["anomaly"]
    if anomaly.dtype == object:
        return (anomaly.astype(str).str.lower() == "anomaly").to_numpy(dtype=np.int64)
    return anomaly.to_numpy(dtype=np.int64)


def load_opssat(csv_path: Path) -> MissionData:
    """Load OPS-SAT segments into one single-feature stream.

    As in the legacy code, the nine channels of ``segments.csv`` are merged into one stream by sorting all
    rows on ``timestamp`` (stable sort; ties broken by channel name, then file order, so the stream is
    deterministic; the legacy used an unstable quicksort, so tie order differs from the legacy run). The number of
    duplicate timestamps is logged. Window content therefore mixes physically different channels. The feature is ``value`` (C = 1) and
    the label is the ``anomaly`` column (strings are mapped with ``== "anomaly"``). The ``train`` column of the
    file is ignored: the split is a chronological ratio (see ``chronological_split``).
    """
    df = pd.read_csv(csv_path)
    for col in ("timestamp", "value", "anomaly"):
        if col not in df.columns:
            raise ValueError(f"segments.csv is missing column '{col}'")
    df["timestamp"] = pd.to_datetime(df["timestamp"], **_parse_kwargs(df["timestamp"]))
    n_dup = int(df["timestamp"].duplicated().sum())
    log.info("OPS-SAT: %d rows, %d duplicate timestamps (%.1f%%); rows with equal timestamps are ordered by "
             "channel name (stable sort), so the interleaved stream is deterministic.",
             len(df), n_dup, 100.0 * n_dup / max(len(df), 1))
    # Deterministic order: timestamp, then channel name, then original file order (stable sort).
    sort_cols = ["timestamp"] + (["channel"] if "channel" in df.columns else [])
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    labels = pd.Series(_row_labels(df))
    channels = sorted(df["channel"].unique().tolist()) if "channel" in df.columns else []
    return MissionData(
        values=df["value"].to_numpy(dtype=np.float64)[:, None],
        labels=labels.to_numpy(),
        timestamps=to_naive_utc(df["timestamp"]),
        channel_names=["value"],
        meta={"mission": "ops_sat", "source_channels": channels, "known_event_intervals": []},
    )


def load_opssat_multichannel(csv_path: Path, label_mode: str = "any") -> MissionData:
    """Load OPS-SAT as ONE multichannel stream: the nine channels pivoted on the unique timestamps (C = 9).

    ``segments.csv`` has one row per ``(channel, timestamp)``; the channels are sampled at different, partly
    non-overlapping times (a timestamp holds between 1 and 7 channels), and every row has its own ``anomaly`` flag.
    Construction (deterministic, independent of the row order of the file):

    * rows are pivoted on the unique timestamps, columns are the channel names in sorted order;
    * values: zero-order hold (forward fill) over time for the channels missing at a timestamp, and the first
      observed value of a channel is back-filled over the leading rows before its first observation (so no NaN
      remains; this uses a value from the future only for the leading rows of a channel);
    * label per timestamp, computed from the OBSERVED rows only (never from filled values, which would carry a label
      far beyond its segment): ``label_mode="any"`` (default) = 1 if any channel observed at that timestamp is
      anomalous; ``"majority"`` = 1 if at least half of the observed channels are anomalous;
    * a repeated ``(channel, timestamp)`` pair is averaged (value) and max-ed (label).

    The statistics of the split (anomaly percentage per part) differ from the interleaved stream; they are logged and
    stored in ``meta["label_rule"]``. The ``train`` column is ignored (chronological split on unique timestamps).

    Args:
        csv_path: ``segments.csv`` with columns ``channel, timestamp, value, anomaly``.
        label_mode: ``"any"`` or ``"majority"``.

    Returns:
        MissionData with ``values`` of shape ``(n_unique_timestamps, n_channels)``.
    """
    if label_mode not in {"any", "majority"}:
        raise ValueError("label_mode must be 'any' or 'majority'")
    df = pd.read_csv(csv_path)
    for col in ("channel", "timestamp", "value", "anomaly"):
        if col not in df.columns:
            raise ValueError(f"segments.csv is missing column '{col}'")
    df["timestamp"] = to_naive_utc(df["timestamp"])
    df["y"] = _row_labels(df)
    grp = df.groupby(["timestamp", "channel"], sort=True)
    agg = grp.agg(value=("value", "mean"), y=("y", "max")).reset_index()
    vals = agg.pivot(index="timestamp", columns="channel", values="value").sort_index().sort_index(axis=1)
    lab = agg.pivot(index="timestamp", columns="channel", values="y").reindex(index=vals.index, columns=vals.columns)
    n_obs = lab.notna().sum(axis=1).to_numpy()
    n_pos = lab.fillna(0).sum(axis=1).to_numpy()
    y = (n_pos > 0) if label_mode == "any" else (2 * n_pos >= n_obs)
    filled = vals.ffill().bfill()
    out = MissionData(
        values=filled.to_numpy(dtype=np.float64),
        labels=y.astype(np.int64),
        timestamps=pd.DatetimeIndex(vals.index),
        channel_names=[str(c) for c in vals.columns],
        meta={"mission": "ops_sat", "mode": "multichannel", "source_channels": [str(c) for c in vals.columns],
              "known_event_intervals": [], "n_rows": int(len(df)),
              "label_rule": f"{label_mode} observed channel anomalous at the timestamp (no label forward fill)",
              "value_rule": "zero-order hold; leading rows back-filled with the first observation of the channel"},
    )
    cut = int(len(out) * 0.75)
    log.info("OPS-SAT multichannel: %d rows -> %d unique timestamps x %d channels; anomalous %.2f %% overall "
             "(%.2f %% in the first 75%%, %.2f %% in the last 25%%) with label rule '%s'",
             len(df), len(out), vals.shape[1], 100.0 * float(out.labels.mean()),
             100.0 * float(out.labels[:cut].mean()) if cut else float("nan"),
             100.0 * float(out.labels[cut:].mean()) if cut < len(out) else float("nan"), label_mode)
    return out
