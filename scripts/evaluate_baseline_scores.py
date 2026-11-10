"""Evaluate the exported window scores of a baseline with the SAME honest protocol as ATAN-DN.

Input: ``--scores`` (``.npz`` from a baseline runner started with ``--split-json``/``--export-scores``) and the split
JSON of ``scripts/export_split.py``. The threshold is fitted on the VALIDATION windows (grid F1 with the spike filter
inside the search), the scores are normalised with the fit-slice scores, the known-event mask (``eval.mask_source``: telecommands or rare events, built by the same functions as
``evaluate_mission``; ESA with ``data.telecommand_dir`` set), its buffers and the spike
filter length are the TRAINING-derived ones (``mask_params.json`` of the ATAN-DN run on the same mission, default
``<out-dir>/mask_params.json`` if it exists), and the test labels are used only for the final metrics. With
``--no-stage3`` the baseline is reported without mask and spike filter. The report has the keys of ``report.json``
(``window_metrics``, ``pointwise_metrics_rule_A``, counts), so ``scripts/make_tables.py`` reads it like an ATAN-DN report.
See ``baselines/protocol_adapter.py`` for the exact rules (``--grid native|atandn``).

Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from _common import (REPO_ROOT, ScriptError, add_common_args, load_mission, load_run_config, resolve_paths, run_cli)

log = logging.getLogger("atandn.scripts.evaluate_baseline_scores")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``evaluate_baseline_scores.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--scores", required=True, type=Path, help="score .npz written by a baseline runner")
    p.add_argument("--split-json", required=True, type=Path, help="split JSON of scripts/export_split.py")
    p.add_argument("--mask-params", type=Path, default=None,
                   help="mask_params.json of an ATAN-DN run on this mission (default: <out-dir>/mask_params.json)")
    p.add_argument("--no-stage3", action="store_true", help="report the baseline without known-event mask and spike filter")
    p.add_argument("--grid", default="native", choices=["native", "atandn"],
                   help="native: the baseline's own windows (default); atandn: causal mapping to the ATAN-DN window grid")
    p.add_argument("--allow-train-threshold-fallback", action="store_true",
                   help="no positive validation window: fit the threshold on the fit windows (flagged)")
    p.add_argument("--report", type=Path, default=None, help="output JSON (default: <scores stem>_report.json)")
    return p


def run(args: argparse.Namespace) -> None:
    """Evaluate the score file and write the report."""
    sys.path.insert(0, str(REPO_ROOT / "baselines"))
    import protocol_adapter
    from atandn.utils.io import save_json

    cfg, _, paths_cfg = load_run_config(args.config)
    if args.allow_train_threshold_fallback:
        cfg.eval.allow_train_threshold_fallback = True
    rp = resolve_paths(args, cfg, paths_cfg)
    for f in (args.scores, args.split_json):
        if not Path(f).is_file():
            raise ScriptError(f"file not found: {f}", 2)
    mp_path = args.mask_params or rp.mask_params
    mask_params = None
    if Path(mp_path).is_file():
        mask_params = json.loads(Path(mp_path).read_text(encoding="utf-8"))
        log.info("training-derived mask parameters from %s", mp_path)
    elif args.mask_params is not None:
        raise ScriptError(f"mask parameters file not found: {mp_path}", 2)
    else:
        log.warning("no mask_params.json at %s: after-buffer 0 and spike filter 1 unless the config fixes them", mp_path)
    mission = load_mission(cfg, rp.data_root)
    report = protocol_adapter.evaluate_baseline_scores(
        args.scores, cfg, labels=mission.labels, timestamps=mission.timestamps, split=args.split_json,
        intervals=list(mission.meta.get("known_event_intervals", [])),
        telecommands=mission.meta.get("telecommands"), values=mission.values, mask_params=mask_params,
        stage3=not args.no_stage3, grid=args.grid,
        report_extra={"scores_file": Path(args.scores).name, "mask_params_file": str(Path(mp_path).name) if mask_params else None})
    out = save_json(report, args.report or Path(args.scores).with_name(Path(args.scores).stem + "_report.json"))
    log.info("%s | window F1 %.4f | point-wise F1 %.4f | wrote %s", report["baseline"]["model"],
             report["window_metrics"]["f1"], report["pointwise_metrics_rule_A"]["f1"], out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
