"""Attention-norm report of a trained checkpoint: per-head spectral norms of W_Q, W_K and the logit bound Lambda
(Eq. Lambda of the paper). Needs only the weights; --data-root is accepted for symmetry and ignored."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import run_cli
from _reports import add_report_args, load_model_and_test_windows

log = logging.getLogger("atandn.scripts.attention_report")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``attention_report.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_report_args(p)
    p.add_argument("--out", type=Path, default=Path("attention_report.json"), help="output JSON")
    return p


def run(args: argparse.Namespace) -> None:
    """Compute and write the report."""
    from atandn.evaluation.attention_report import attention_norms
    from atandn.utils.io import save_json

    args.data_root = None   # weights only
    model, _cfg, _ = load_model_and_test_windows(args)
    out = save_json(attention_norms(model), args.out)
    log.info("wrote %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
