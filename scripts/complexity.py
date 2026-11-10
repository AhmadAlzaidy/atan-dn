"""Model complexity: parameters, analytic FLOPs per window (with/without decoder, per inverse method), latency.

The model is built from the config (or restored from --checkpoint). FLOPs follow the analytic formula of
atandn.evaluation.flops_per_window; latency is the measured wall-clock time of one forward pass (batch 1).
Results are logged and, with --out, written as JSON; --project adds the OBC latency projection. Exit codes: 0 ok, 1 runtime failure, 2 invalid input.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import add_common_args, load_run_config, run_cli

log = logging.getLogger("atandn.scripts.complexity")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``complexity.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, with_device=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="optional model.pt; default: build from the config")
    p.add_argument("--repeats", type=int, default=200, help="timed forward passes for the latency (default: 200)")
    p.add_argument("--warmup", type=int, default=20, help="untimed warm-up passes (default: 20)")
    p.add_argument("--out", type=Path, default=None, help="write the results as JSON to this file")
    p.add_argument("--project", type=Path, default=None,
                   help="hardware YAML (configs/obc_hardware.yaml): add a theoretical latency projection per platform "
                        "(flops / (efficiency * peak GFLOPS)); stored under 'projection' in the JSON")
    p.add_argument("--project-flops-key", default="decoder_inverse_solve",
                   choices=["without_decoder", "decoder_no_inverse_layer", "decoder_inverse_solve",
                            "decoder_inverse_neumann"], help="which FLOPs count is projected (default: %(default)s)")
    return p


def run(args: argparse.Namespace) -> None:
    """Compute and report the complexity numbers."""
    from atandn.evaluation import count_parameters, flops_per_window, measure_latency
    from atandn.models import ATANDN
    from atandn.training import load_checkpoint
    from atandn.utils.io import save_json

    cfg, _, _ = load_run_config(args.config)
    if args.checkpoint is not None:
        model, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    else:
        model = ATANDN(**{**cfg.model.__dict__, "conv_filters": tuple(cfg.model.conv_filters)})
    m = cfg.model
    kw = dict(in_channels=m.in_channels, window_size=m.window_size, conv_filters=tuple(m.conv_filters),
              neumann_steps=m.neumann_steps)
    no_dec = ATANDN(**{**m.__dict__, "conv_filters": tuple(m.conv_filters), "use_reconstruction": False})
    res = {
        "mission": cfg.name, "model_config": cfg.to_dict()["model"],
        "parameters": {"with_decoder": count_parameters(model), "without_decoder": count_parameters(no_dec)},
        "flops_per_window": {
            "without_decoder": flops_per_window(include_decoder=False, inverse_method=None, **kw),
            "decoder_no_inverse_layer": flops_per_window(include_decoder=True, inverse_method=None, **kw),
            "decoder_inverse_solve": flops_per_window(include_decoder=True, inverse_method="solve", **kw),
            "decoder_inverse_neumann": flops_per_window(include_decoder=True, inverse_method="neumann", **kw),
        },
        "latency_with_decoder": measure_latency(model, (1, m.in_channels, m.window_size), args.device,
                                                repeats=args.repeats, warmup=args.warmup),
        "latency_without_decoder": measure_latency(no_dec, (1, m.in_channels, m.window_size), args.device,
                                                   repeats=args.repeats, warmup=args.warmup),
    }
    log.info("parameters: with decoder %d | without decoder %d", res["parameters"]["with_decoder"],
             res["parameters"]["without_decoder"])
    for name, f in res["flops_per_window"].items():
        log.info("FLOPs per window [%s]: total %.3e", name, f["total"])
    log.info("latency (batch 1, %s): with decoder %.3f ms, without decoder %.3f ms (mean)",
             res["latency_with_decoder"]["device"], res["latency_with_decoder"]["mean_ms"],
             res["latency_without_decoder"]["mean_ms"])
    if args.project is not None:
        from atandn.evaluation.projection import load_hardware, project_latency

        hw, period, frac = load_hardware(args.project)
        flops = res["flops_per_window"][args.project_flops_key]["total"]
        rows = project_latency(flops, hw, period, frac)
        res["projection"] = {"flops": flops, "flops_key": args.project_flops_key, "decision_period_s": period,
                             "realtime_fraction": frac, "hardware_file": args.project.name, "platforms": rows,
                             "note": "compute-bound estimate; hardware values from the YAML (see its source column)"}
        for r in rows:
            log.info("projection %-26s %10.4f ms %12.2f fps  %s", r["platform"], r["latency_ms"], r["fps"], r["viability"])
    if args.out is not None:
        log.info("wrote %s", save_json(res, args.out))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
