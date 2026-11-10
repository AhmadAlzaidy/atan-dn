"""Measure the round-trip accuracy of the BIGN inverse (the source of every number in docs/BIGN_INVERSE.md).

For every case a fresh ``BIGN1D(C)`` is built (seeded), an input ``x ~ N(0, scale^2)`` of shape (B, C, T) is drawn
and ``ExactInverseBIGN1D.round_trip_error(x, bign)`` is evaluated; the table lists the mean (over seeds) relative
error and clamp rate. Conditions printed in the header: init mode, seeds, input distribution, shape, dtype.

Example::

    python scripts/inverse_diagnostics.py
    python scripts/inverse_diagnostics.py --channels 1 6 --scales 1 10 --seeds 4 --init uniform

Exit code 0 on success, 2 on invalid arguments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import ScriptError, run_cli

log = logging.getLogger("atandn.scripts.inverse_diagnostics")

DEFAULT_CHANNELS = (1, 6, 64, 192)
DEFAULT_SCALES = (1.0, 10.0, 1000.0)
DEFAULT_METHODS = (("solve", 4), ("neumann", 4), ("neumann", 50))


def measure(channels: int, scale: float, init: str = "diag_dominant", method: str = "solve", neumann_steps: int = 4,
            seeds: Sequence[int] = tuple(range(8)), batch: int = 8, length: int = 64) -> dict:
    """Mean/max over ``seeds`` of the relative round-trip error and the clamp rate of one case.

    Args:
        channels: C of the BIGN layer.
        scale: standard deviation of the Gaussian input (``x ~ N(0, scale^2)``).
        init: ``gamma_init_mode`` of the layer ("diag_dominant" | "uniform").
        method / neumann_steps: inverse settings.
        seeds: ``torch.manual_seed`` values (parameters and inputs are drawn after seeding).
        batch / length: input shape is (batch, channels, length), float32.

    Returns:
        ``{"rel_error": mean, "rel_error_max": max, "clamp_rate": mean}``.
    """
    import torch

    from atandn.layers.bign import BIGN1D, ExactInverseBIGN1D

    errs, clamps = [], []
    for seed in seeds:
        torch.manual_seed(int(seed))
        bign = BIGN1D(channels, gamma_init_mode=init)
        x = torch.randn(batch, channels, length, dtype=torch.float32) * float(scale)
        r = ExactInverseBIGN1D.round_trip_error(x, bign, method=method, neumann_steps=neumann_steps)
        errs.append(r["rel_error"])
        clamps.append(r["clamp_rate"])
    return {"rel_error": sum(errs) / len(errs), "rel_error_max": max(errs), "clamp_rate": sum(clamps) / len(clamps)}


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``inverse_diagnostics.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS), help="values of C")
    p.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES), help="input standard deviations")
    p.add_argument("--init", nargs="+", default=["diag_dominant", "uniform"], choices=["diag_dominant", "uniform"],
                   help="gamma_init_mode values to test (default: both)")
    p.add_argument("--seeds", type=int, default=8, help="number of seeds 0..N-1 (default: 8)")
    p.add_argument("--batch", type=int, default=8, help="batch size B (default: 8)")
    p.add_argument("--length", type=int, default=64, help="time steps T (default: 64)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run(args: argparse.Namespace) -> None:
    """Print the measurement table."""
    if args.seeds < 1 or min(args.channels) < 1:
        raise ScriptError("--seeds and --channels must be >= 1", 2)
    seeds = tuple(range(args.seeds))
    print(f"conditions: float32, x ~ N(0, scale^2) of shape ({args.batch}, C, {args.length}), seeds {seeds[0]}..{seeds[-1]} "
          f"(mean over seeds), BIGN1D defaults (beta_min 1e-6, gamma_diag 0.5, gamma_off 0.01/(C-1)), eps 1e-3")
    print(f"{'init':<14}{'C':>5}{'scale':>8}  {'method':<10}{'rel_error':>11}{'max':>10}{'clamp_rate':>12}")
    for init in args.init:
        for c in args.channels:
            for scale in args.scales:
                for method, k in DEFAULT_METHODS:
                    r = measure(c, scale, init, method, k, seeds, args.batch, args.length)
                    name = method if method == "solve" else f"neumann{k}"
                    print(f"{init:<14}{c:>5}{scale:>8g}  {name:<10}{r['rel_error']:>11.3g}{r['rel_error_max']:>10.3g}"
                          f"{r['clamp_rate']:>12.3g}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
