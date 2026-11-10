"""Prepare the data of one mission.

ESA-ADB (m1, m2): raw mission folder (labels.csv, anomaly_types.csv, channels.csv, channels/) -> one parquet
file at <data-root>/<data.path>.
OPS-SAT: no conversion is needed; the script only checks segments.csv (columns, size, label balance).

Exit codes: 0 ok, 1 runtime failure (e.g. missing files), 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import ScriptError, add_common_args, load_run_config, resolve_paths, run_cli

log = logging.getLogger("atandn.scripts.prepare_data")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``prepare_data.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="raw ESA-ADB mission folder (default: paths.raw_dir of the config); ignored for OPS-SAT")
    p.add_argument("--force", action="store_true", help="rebuild the parquet even if it already exists")
    return p


def run(args: argparse.Namespace) -> None:
    """Build the ESA parquet or validate the OPS-SAT csv."""
    cfg, _, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, cfg, paths_cfg)
    target = rp.data_root / cfg.data.path
    if cfg.data.kind == "opssat":
        from atandn.data import load_mission_from_config

        if not target.is_file():
            raise ScriptError(f"OPS-SAT file not found: {target}. Place segments.csv there or use --data-root.")
        data = load_mission_from_config(cfg, rp.data_root)     # data.opssat_mode: interleaved | multichannel
        log.info("OPS-SAT OK (%s): %d samples x %d channels, anomalous samples %.3f %%, source channels: %s",
                 cfg.data.opssat_mode, len(data), data.values.shape[1], 100.0 * float(data.labels.mean()),
                 data.meta.get("source_channels"))
        return
    if target.is_file() and not args.force:
        log.info("parquet already exists (%s); use --force to rebuild", target)
        return
    if rp.raw_dir is None:
        raise ScriptError("raw ESA-ADB folder required: pass --raw-dir or set paths.raw_dir in the config")
    if not rp.raw_dir.is_dir():
        raise ScriptError(f"raw folder not found: {rp.raw_dir}")
    from atandn.data import prepare_esa_mission

    d = cfg.data
    out = prepare_esa_mission(rp.raw_dir, target, d.channel_start, d.channel_end, d.target_sampling_rate,
                              labels_csv=d.labels_csv, anomaly_types_csv=d.anomaly_types_csv)
    log.info("wrote %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
