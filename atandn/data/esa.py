"""Loading of prepared ESA-ADB mission parquet files and the known-event intervals."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atandn.data.splits import MissionData, to_naive_utc

MISSION_FILES = {"m1": "all_telemetry_41_46.parquet", "m2": "all_telemetry_18_28.parquet"}
MISSION_CHANNELS = {"m1": (41, 46), "m2": (18, 28)}


def load_rare_event_intervals(
    labels_csv: Path, anomaly_types_csv: Path, channels: list[str]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ``(start, end)`` (naive UTC) of labels.csv rows with Category ``'Rare Event'`` on ``channels``.

    These are the *planned/known* events used by ``apply_known_event_mask`` (an operational-schedule assumption).
    """
    labels = pd.read_csv(labels_csv)
    types = pd.read_csv(anomaly_types_csv)
    merged = labels.merge(types[["ID", "Category"]], on="ID", how="left")
    rare = merged[merged["Channel"].astype(str).isin(channels) & (merged["Category"] == "Rare Event")]
    out = []
    for _, row in rare.iterrows():
        s, e = to_naive_utc([row["StartTime"], row["EndTime"]])
        out.append((s, e))
    return out


def load_esa_mission(root: Path, mission: str, channels: list[str] | None = None,
                     telecommand_root: Path | None = None, telecommand_priorities: list | None = None) -> MissionData:
    """Load a prepared ESA-ADB mission (output of ``prepare_esa_mission``).

    Args:
        root: Folder holding the parquet (and, optionally, labels.csv / anomaly_types.csv).
        mission: ``"m1"`` (channels 41-46) or ``"m2"`` (channels 18-28).
        channels: Optional subset of feature columns; default all columns except ``label``.
        telecommand_root: raw mission folder with ``telecommands.csv``; if given, ``meta["telecommands"]`` holds the
            execution table of :func:`atandn.data.telecommands.load_telecommands`.
        telecommand_priorities: priorities kept in that table (None = all).

    Returns:
        MissionData whose ``meta["known_event_intervals"]`` holds the rare-event intervals when the
        csv files are present (empty list otherwise).
    """
    if mission not in MISSION_FILES:
        raise ValueError(f"Unknown ESA mission '{mission}'")
    root = Path(root)
    df = pd.read_parquet(root / MISSION_FILES[mission])
    feats = [c for c in df.columns if c != "label"] if channels is None else list(channels)
    lo, hi = MISSION_CHANNELS[mission]
    intervals: list = []
    lab, typ = root / "labels.csv", root / "anomaly_types.csv"
    if lab.exists() and typ.exists():
        intervals = load_rare_event_intervals(lab, typ, [f"channel_{i}" for i in range(lo, hi + 1)])
    meta: dict = {"mission": mission, "known_event_intervals": intervals}
    if telecommand_root is not None:
        from atandn.data.telecommands import load_telecommands

        meta["telecommands"] = load_telecommands(telecommand_root, mission, telecommand_priorities)
    return MissionData(
        values=df[feats].to_numpy(dtype=np.float64),
        labels=df["label"].to_numpy().astype(np.int64),
        timestamps=to_naive_utc(df.index),
        channel_names=feats,
        meta=meta,
    )
