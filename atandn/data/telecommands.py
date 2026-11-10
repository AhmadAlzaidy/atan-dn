"""Telecommand execution times of the ESA-ADB missions (used by the Stage-2 settling buffers).

Folder layout (checked on the public ESA-ADB download, read-only)::

    <mission folder>/telecommands.csv          columns: Telecommand, Priority   (M1: 698 rows, M2: 123 rows)
    <mission folder>/telecommands/
        M1: preprocessed_telecommand_<N>.pkl   pickled DataFrame, UTC tz-aware DatetimeIndex, one column
                                               ``telecommand_<N>`` (uint8, value 1 = executed);
            telecommand_<N>/telecommand_<N>    the same table with a naive index (raw copy), plus ``.zip`` archives
        M2: telecommand_<N>/telecommand_<N>    pickled DataFrame, naive DatetimeIndex named ``datetime``, column
                                               ``telecommand_<N>`` (object: 1 or 1.0)

Each file is one telecommand TYPE; every index entry is one execution of it (M1: 1.59 million executions of 681
types, M2: 1.92 million executions of 123 types, many with identical timestamps in M2). Timestamps are UTC; tz-aware
indices are converted and made naive, naive indices are taken as UTC. The execution times are an operational record
that is known when the command is executed (causal at test time); no label is involved.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from atandn.data.splits import to_naive_utc

log = logging.getLogger(__name__)

_MISSION_FOLDERS = {"m1": "ESA-Mission1", "m2": "ESA-Mission2"}


def _mission_dir(root: Path, mission: str) -> Path:
    """Folder that holds ``telecommands.csv`` (``root`` itself, or its ``ESA-MissionN`` child)."""
    root = Path(root)
    if (root / "telecommands.csv").is_file():
        return root
    child = root / _MISSION_FOLDERS.get(mission, "")
    if (child / "telecommands.csv").is_file():
        return child
    raise FileNotFoundError(f"telecommands.csv not found in {root} (or in {root / _MISSION_FOLDERS.get(mission, '<mission>')}); "
                            "pass the raw ESA-ADB mission folder")


def _find_pickle(tc_dir: Path, name: str) -> Path | None:
    for cand in (tc_dir / f"preprocessed_{name}.pkl", tc_dir / name / name, tc_dir / name):
        if cand.is_file():
            return cand
    return None


def load_telecommands(root: str | Path, mission: str, priorities: list | None = None) -> pd.DataFrame:
    """All telecommand executions of a mission as a table sorted by time.

    Args:
        root: raw mission folder (the one with ``telecommands.csv``), or its parent.
        mission: ``"m1"`` or ``"m2"`` (only used to locate the folder).
        priorities: keep only telecommand types whose ``Priority`` in ``telecommands.csv`` is in this list
            (``None`` = all).

    Returns:
        DataFrame with columns ``time`` (naive UTC ``datetime64[ns]``), ``type`` (``"telecommand_<N>"``) and
        ``priority`` (int), sorted by ``(time, type)``; exact duplicate rows are dropped. Types whose file is missing
        are skipped with a WARNING.
    """
    mdir = _mission_dir(Path(root), mission)
    meta = pd.read_csv(mdir / "telecommands.csv")
    if priorities is not None:
        keep = {int(p) for p in priorities}
        meta = meta[meta["Priority"].astype(int).isin(keep)]
    tc_dir = mdir / "telecommands"
    frames, missing = [], 0
    for name, prio in zip(meta["Telecommand"].astype(str), meta["Priority"].astype(int)):
        path = _find_pickle(tc_dir, name)
        if path is None:
            missing += 1
            continue
        d = pd.read_pickle(path)
        vals = pd.to_numeric(d.iloc[:, 0], errors="coerce").to_numpy()
        idx = to_naive_utc(d.index)
        ok = np.isfinite(vals) & (vals != 0)            # a row is an execution unless its value is 0/NaN
        if ok.any():
            frames.append(pd.DataFrame({"time": idx[ok], "type": name, "priority": int(prio)}))
    if missing:
        log.warning("%d telecommand file(s) listed in telecommands.csv were not found under %s", missing, tc_dir)
    if not frames:
        return pd.DataFrame({"time": pd.DatetimeIndex([]), "type": pd.Series(dtype=object),
                             "priority": pd.Series(dtype=int)})
    out = pd.concat(frames, ignore_index=True)
    n0 = len(out)
    out = out.drop_duplicates().sort_values(["time", "type"], kind="stable").reset_index(drop=True)
    log.info("%s: %d telecommand executions of %d types (%d exact duplicate rows dropped), %s .. %s",
             mission, len(out), out["type"].nunique(), n0 - len(out), out["time"].iloc[0], out["time"].iloc[-1])
    return out


def load_telecommand_times(root: str | Path, mission: str, priorities: list | None = None) -> pd.DatetimeIndex:
    """Sorted, naive-UTC execution times of the telecommands (duplicates kept; see :func:`load_telecommands`)."""
    return pd.DatetimeIndex(load_telecommands(root, mission, priorities)["time"])
