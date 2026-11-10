"""Export the fit / validation / test index ranges of a mission so that every baseline sees the SAME split as ATAN-DN.

The ranges come from the code ATAN-DN itself uses (``prepare_protocol_data`` -> ``event_aware_validation_split`` on the
TRAINING split; the test split is ``train_ratio`` .. end). The JSON (schema ``atandn-split/1``, see
``baselines/split_io.py``) holds absolute half-open sample ranges, the ATAN-DN window length and stride that fixed the
validation gap, the split report and the identity hash of the sample timestamps, which the baseline runners verify.

The validation block depends on ``model.window_size`` only through its gap and minimum fitting length; pass the
window of the final ATAN-DN model with ``--window-size`` (or ``--best-params best_params.json``) so that both use the
same block. Only training-split labels are read (for the event-aware block); no test label is used.

Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from _common import (REPO_ROOT, ScriptError, add_common_args, load_mission, load_run_config, prepare_cfg_for_data,
                     resolve_paths, run_cli)

log = logging.getLogger("atandn.scripts.export_split")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``export_split.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--out", required=True, type=Path, help="output split JSON")
    p.add_argument("--window-size", type=int, default=None, help="ATAN-DN window length (default: the config's)")
    p.add_argument("--best-params", type=Path, default=None,
                   help="best_params.json of the ATAN-DN run: its window_size is used (overridden by --window-size)")
    return p


def build_split(cfg, train, test, n_total: int, timestamps) -> dict:
    """Build the split dictionary from the loaded splits (``train``/``test`` :class:`MissionData`)."""
    sys.path.insert(0, str(REPO_ROOT / "baselines"))
    import split_io
    from atandn.evaluation.pipeline import prepare_protocol_data

    data = prepare_protocol_data(train, test, cfg)
    sp = data.split
    n_train = len(train)
    return {
        "schema": split_io.SPLIT_SCHEMA, "mission": cfg.name, "n_samples": int(n_total),
        "train_ratio": float(cfg.data.train_ratio), "train_stop": int(n_train),
        "fit": {"start": int(sp.fit.start), "stop": int(sp.fit.stop)},
        "val": {"start": int(sp.val.start), "stop": int(sp.val.stop)},
        "test": {"start": int(n_train), "stop": int(n_total)},
        "window_size": int(cfg.model.window_size), "test_stride": int(cfg.data.test_stride),
        "validation_report": sp.report,
        "timestamps": split_io.timestamps_identity(timestamps),
        "note": "fit/val index the full stream; fit + val + gap lie inside the training split; the test range is never "
                "used for fitting. Baselines keep their own window length and are scored at window ends.",
    }


def run(args: argparse.Namespace) -> None:
    """Load the mission, compute the split and write the JSON."""
    import json

    cfg, _, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, cfg, paths_cfg)
    if args.best_params is not None:
        if not args.best_params.is_file():
            raise ScriptError(f"best-params file not found: {args.best_params}", 2)
        raw = json.loads(args.best_params.read_text(encoding="utf-8"))
        win = (raw.get("params") or raw).get("window_size")
        if win is not None:
            cfg.model.window_size = int(win)
    if args.window_size is not None:
        cfg.model.window_size = int(args.window_size)
    mission = load_mission(cfg, rp.data_root)
    prepare_cfg_for_data(cfg, mission.values.shape[1])
    train, test = mission.split(cfg.data.train_ratio)
    split = build_split(cfg, train, test, len(mission), mission.timestamps)
    sys.path.insert(0, str(REPO_ROOT / "baselines"))
    import split_io

    out = split_io.write_split(args.out, split)
    split_io.validate_split(split)
    log.info("wrote %s | fit [%d, %d) val [%d, %d) test [%d, %d) | window %d stride %d", out, split["fit"]["start"],
             split["fit"]["stop"], split["val"]["start"], split["val"]["stop"], split["test"]["start"],
             split["test"]["stop"], split["window_size"], split["test_stride"])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
