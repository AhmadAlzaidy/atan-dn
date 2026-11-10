"""Train ATAN-DN with the tuned hyperparameters and freeze the evaluation protocol from the TRAINING split.

Two splits only (train/test). A validation block is carved from the TRAINING split (event-aware: the chronological
tail, or the latest block with >= train.val_min_events anomalous events if the tail has fewer); the model is fitted
on the data before it. Early stopping and checkpoint selection use the threshold-free average precision of the
dual-evidence score on that block. The selected model is final (no refit); the normalizer, the training-derived mask
parameters and the F1-optimal threshold of the validation block are frozen by the shared pipeline
(atandn.evaluation.pipeline) and stored in model.pt together with train_ratio, test_stride and window_size.
Refuses to run without best-params unless --allow-legacy-defaults is given.
Writes model.pt, mask_params.json and train_summary.json to --out-dir.

Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import (ScriptError, add_common_args, load_mission, load_run_config, params_from_cfg,
                     prepare_cfg_for_data, resolve_paths, run_cli)

log = logging.getLogger("atandn.scripts.train")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``train.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, with_device=True)
    p.add_argument("--params", "--best-params", dest="params", type=Path, default=None,
                   help="best_params.json from tune.py (default: <out-dir>/best_params.json)")
    p.add_argument("--allow-legacy-defaults", action="store_true",
                   help="train with predefined mission defaults when no best-params file exists")
    p.add_argument("--seed", type=int, default=None, help="training seed (default: train.seed)")
    p.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs")
    return p


def run(args: argparse.Namespace) -> None:
    """Train, freeze the protocol (normalizer, mask, threshold) from the training split, save everything."""
    from atandn.analysis import save_mask_parameters
    from atandn.evaluation.pipeline import fit_frozen_protocol, prepare_protocol_data
    from atandn.training import apply_params, check_legacy_defaults, save_checkpoint
    from atandn.training.fit import fit_on_split
    from atandn.utils.io import load_json, save_json
    from atandn.utils.seed import set_seed

    cfg, _, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, cfg, paths_cfg)
    params_path = args.params or rp.best_params
    legacy = check_legacy_defaults(params_path.is_file(), args.allow_legacy_defaults, "train.py")
    if params_path.is_file():
        params = {**params_from_cfg(cfg), **load_json(params_path)["params"]}
        cfg = apply_params(cfg, params)
        log.info("hyperparameters from %s", params_path)
    else:
        params = params_from_cfg(cfg)
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.max_epochs is not None:
        cfg.train.max_epochs = args.max_epochs

    set_seed(cfg.train.seed)
    mission = load_mission(cfg, rp.data_root)
    prepare_cfg_for_data(cfg, mission.values.shape[1])
    train, _test_unused = mission.split(cfg.data.train_ratio)   # test part dropped: training uses the head only
    del _test_unused, mission

    w = cfg.model.window_size
    data = prepare_protocol_data(train, None, cfg)       # event-aware validation block carved from the train split
    vs = data.split
    model, result = fit_on_split(cfg, train, vs, args.device)
    summary = {k: v for k, v in result.items() if k != "history"}
    log.info("training done: %s", summary)

    # No refit on the full training split: the model selected on the validation block is the final model, and the
    # threshold fitted on that block belongs to this very model.
    extra = {"best_params": params, "fit_summary": summary, "val_split": vs.report, "used_legacy_defaults": legacy,
             "train_ratio": cfg.data.train_ratio, "test_stride": cfg.data.test_stride, "window_size": w}
    frozen, frozen_error = None, None
    try:
        frozen = fit_frozen_protocol(model, data, cfg, device=args.device)
    except ValueError as exc:   # e.g. no positive in the validation block; evaluate.py may refit with a fallback
        frozen_error = str(exc)
        log.error("could not freeze the protocol: %s", exc)
    if frozen is not None:
        save_mask_parameters(frozen["mask_params"], rp.mask_params)
        extra.update({"frozen": frozen, "mask_params": frozen["mask_params"], "threshold": frozen["threshold"],
                      "threshold_source": frozen["threshold_source"], "normalizer": frozen["normalizer"],
                      "alpha_rec": frozen["alpha_rec"]})
    else:
        extra["frozen_error"] = frozen_error
    save_checkpoint(rp.checkpoint, model, cfg, extra)
    save_json({"summary": summary, "history": result["history"], "mask_params_file": rp.mask_params.name,
               "seed": cfg.train.seed, "val_split": vs.report, "used_legacy_defaults": legacy,
               "frozen_error": frozen_error}, rp.out_dir / "train_summary.json")
    log.info("wrote %s", rp.checkpoint)
    if frozen_error is not None:
        raise ScriptError(f"model saved to {rp.checkpoint} but the protocol could not be frozen: {frozen_error}", 2)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
