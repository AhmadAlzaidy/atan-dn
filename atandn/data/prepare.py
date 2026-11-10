"""Raw ESA-ADB mission folder -> parquet (port of cell 0 of ``phd-v1.ipynb``, generalised to any channel range)."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MISSION_PRESETS = {
    "m1": {"channel_start": 41, "channel_end": 46, "target_sampling_rate": 0.033, "out_name": "all_telemetry_41_46.parquet"},
    "m2": {"channel_start": 18, "channel_end": 28, "target_sampling_rate": 0.056, "out_name": "all_telemetry_18_28.parquet"},
}


def _naive(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tzinfo is not None else t


def prepare_esa_mission(
    base_dir: Path,
    out_path: Path,
    channel_start: int,
    channel_end: int,
    target_sampling_rate: float,
    labels_csv: str = "labels.csv",
    anomaly_types_csv: str = "anomaly_types.csv",
    channels_csv: str = "channels.csv",
    channel_subdir: str = "channels",
) -> Path:
    """Build the single parquet of one ESA-ADB mission.

    Rows ``channel_start-1 : channel_end`` of ``channels.csv`` select the channels (1-based, inclusive). Each
    channel is a pickled DataFrame without extension in ``<base_dir>/<channel_subdir>``; its first column is
    used (float64). Channels are outer-joined on their timestamps. The sample label is 1 only inside
    ``[StartTime, EndTime]`` of a ``labels.csv`` row on a selected channel whose Category (from
    ``anomaly_types.csv``) is ``"Anomaly"``; ``"Rare Event"`` rows stay label 0. The frame is then resampled to
    ``1/target_sampling_rate`` seconds with ``ffill().bfill()`` and written as parquet (float64).

    Note: pickle is used because the dataset ships pickled frames; only load data you trust.

    Returns:
        ``out_path``.
    """
    base_dir, out_path = Path(base_dir), Path(out_path)
    labels_df = pd.read_csv(base_dir / labels_csv)
    types_df = pd.read_csv(base_dir / anomaly_types_csv)
    channels_df = pd.read_csv(base_dir / channels_csv)
    labels_df = labels_df.merge(types_df[["ID", "Category"]], on="ID", how="left")

    sel = channels_df.iloc[channel_start - 1: channel_end]["Channel"].astype(str).tolist()
    log.info("Selected %d channels: %s", len(sel), sel)

    series = {}
    for ch in sel:
        path = base_dir / channel_subdir / ch
        if not path.is_file():
            raise FileNotFoundError(f"Channel file not found: {path}")
        with open(path, "rb") as f:
            df = pickle.load(f)
        s = df.iloc[:, 0].astype("float64")
        s.name = f"ch{ch}"
        series[ch] = s
    telemetry = pd.concat(series.values(), axis=1)
    if telemetry.index.tz is not None:
        telemetry.index = telemetry.index.tz_convert(None)

    mask = np.zeros(len(telemetry), dtype=bool)
    idx = telemetry.index
    for row in labels_df.itertuples(index=False):
        if str(row.Channel) in sel and row.Category == "Anomaly":
            mask |= (idx >= _naive(row.StartTime)) & (idx <= _naive(row.EndTime))
    telemetry["label"] = np.where(mask, 1, 0).astype("float64")

    interval = pd.Timedelta(seconds=1 / target_sampling_rate)
    telemetry = telemetry.resample(interval).ffill().bfill()
    cols = [c for c in telemetry.columns if c != "label"]
    telemetry[cols] = telemetry[cols].astype("float64")
    log.info("Prepared frame %s, NaNs: %d", telemetry.shape, int(telemetry.isnull().sum().sum()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_parquet(out_path)
    return out_path
