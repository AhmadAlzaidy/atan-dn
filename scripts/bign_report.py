"""BIGN saturation report of a trained checkpoint (paper Prop. bounded): Gamma/beta statistics per BIGN layer.

With --data-root the empirical max |y_i| on the TEST windows of the checkpoint mission is compared with the bound
gamma_ii^(-1/2) (descriptive; nothing is fitted on the test split). Writes a JSON list, one entry per BIGN layer.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import run_cli
from _reports import add_report_args, load_model_and_test_windows

log = logging.getLogger("atandn.scripts.bign_report")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``bign_report.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_report_args(p)
    p.add_argument("--out", type=Path, default=Path("bign_report.json"), help="output JSON (default: bign_report.json)")
    return p


def run(args: argparse.Namespace) -> None:
    """Compute and write the report."""
    from atandn.evaluation.bign_report import bign_stats
    from atandn.utils.io import save_json

    model, _cfg, X = load_model_and_test_windows(args)
    out = save_json(bign_stats(model, X), args.out)
    log.info("wrote %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
