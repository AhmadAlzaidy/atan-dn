"""Dispatcher for the baseline runners.

Usage (honest protocol; see baselines/README.md):
    python scripts/export_split.py --config configs/m1.yaml --data-root D --out split.json
    python baselines/run_baseline.py --model {tranad,usad,anomaly_transformer,memto}
        --mission {m1,m2,ops_sat} --data PATH [--out DIR] --split-json split.json --export-scores scores.npz
    python scripts/evaluate_baseline_scores.py --config configs/m1.yaml --data-root D --scores scores.npz --split-json split.json

Legacy (optimistic oracle threshold on the TEST labels, kept only for reproducing the old numbers):
    python baselines/run_baseline.py --model ... --mission ... --data PATH --legacy-test-label-threshold

The matching custom runner is started in a subprocess; the protocol flags are passed through unchanged. Without
``--split-json``/``--export-scores`` or ``--legacy-test-label-threshold`` the runner refuses to start.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("run_baseline")

# (model, mission) -> (runner path relative to baselines/, name of the data argument)
RUNNERS: dict[tuple[str, str], tuple[str, str]] = {
    ("tranad", "m1"): ("tranad/m1_tranad_custom.py", "--data"),
    ("tranad", "m2"): ("tranad/m2_tranad_custom.py", "--data"),
    ("tranad", "ops_sat"): ("tranad/ops_tranad_custom.py", "--data"),
    ("usad", "m1"): ("usad/m1_usad_custom.py", "--data"),
    ("usad", "m2"): ("usad/m2_usad_custom.py", "--data"),
    ("usad", "ops_sat"): ("usad/ops_usad_custom.py", "--data"),
    # Anomaly Transformer: run_custom.py is generic for the parquet missions (m1, m2).
    ("anomaly_transformer", "m1"): ("anomaly_transformer/run_custom.py", "--data_path"),
    ("anomaly_transformer", "m2"): ("anomaly_transformer/run_custom.py", "--data_path"),
    ("anomaly_transformer", "ops_sat"): ("anomaly_transformer/ops_at_custom.py", "--data"),
    ("memto", "m1"): ("memto/m1_memto_custom.py", "--data"),
    ("memto", "m2"): ("memto/m2_memto_custom.py", "--data"),
    ("memto", "ops_sat"): ("memto/ops_memto_custom.py", "--data"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    p = argparse.ArgumentParser(description="Run a baseline on one mission.")
    p.add_argument("--model", required=True,
                   choices=["tranad", "usad", "anomaly_transformer", "memto"])
    p.add_argument("--mission", required=True, choices=["m1", "m2", "ops_sat"])
    p.add_argument("--data", required=True, type=Path,
                   help="mission data file (parquet for m1/m2, csv for ops_sat)")
    p.add_argument("--out", type=Path, default=Path("baseline_out"),
                   help="output directory for checkpoints (default: baseline_out)")
    p.add_argument("--split-json", type=Path, default=None,
                   help="split file of scripts/export_split.py: the runner trains on its FIT slice only")
    p.add_argument("--export-scores", type=Path, default=None,
                   help="write per-window scores (fit/val/test, with end indices) to this .npz; no threshold, no test labels")
    p.add_argument("--score-stride", type=int, default=None,
                   help="stride of the scored windows (default: test_stride of the split file)")
    p.add_argument("--legacy-test-label-threshold", action="store_true",
                   help="LEGACY: threshold swept on the test scores with the test labels (optimistic oracle)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Validate the arguments and run the runner; return its exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    rel, data_flag = RUNNERS[(args.model, args.mission)]
    runner = HERE / rel
    if not runner.is_file():
        LOG.error("runner not found: %s", runner)
        return 2
    data = args.data.resolve()
    if not data.is_file():
        LOG.error("data file not found: %s", data)
        return 2
    cmd = [sys.executable, str(runner), data_flag, str(data), "--out", str(args.out.resolve())]
    if args.split_json is not None:
        cmd += ["--split-json", str(args.split_json.resolve())]
    if args.export_scores is not None:
        cmd += ["--export-scores", str(args.export_scores.resolve())]
    if args.score_stride is not None:
        cmd += ["--score-stride", str(args.score_stride)]
    if args.legacy_test_label_threshold:
        LOG.warning("legacy mode: the threshold is swept on the TEST labels (optimistic oracle)")
        cmd.append("--legacy-test-label-threshold")
    LOG.info("running %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(runner.parent))


if __name__ == "__main__":
    sys.exit(main())
