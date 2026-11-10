"""Optuna TPE hyperparameter search (MedianPruner) on the TRAINING split only.

The objective is the model-selection score on a event-aware validation block of the training split (see
atandn.training.search.build_objective). The test split is cut off before the search starts and never used.
Writes <out-dir>/best_params.json (with a ``provenance`` block: duration_s, n_pruned, library versions) and
<out-dir>/trials.csv (``study.trials_dataframe()``: number, value, state, duration, params_*; input of
scripts/sensitivity.py).

Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import logging
import platform
import time
from typing import Sequence

from _common import (ScriptError, add_common_args, load_mission, load_run_config, params_from_cfg,
                     prepare_cfg_for_data, resolve_paths, run_cli)

log = logging.getLogger("atandn.scripts.tune")


def library_versions() -> dict:
    """Versions of python and of the libraries that influence a search (missing ones are reported as None)."""
    import importlib

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "optuna", "numpy", "pandas", "sklearn", "atandn"):
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except ImportError:
            out[name] = None
    return out


def write_trials(study, path) -> "Path":
    """Write every trial of ``study`` (params, value, state, duration, user attributes) to ``path`` as CSV."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(p, index=False)
    return p


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``tune.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, with_device=True)
    p.add_argument("--n-trials", type=int, default=None, help="number of TPE trials (default: search.n_trials, 75)")
    p.add_argument("--seed", type=int, default=None, help="TPE sampler seed (default: search.seed)")
    p.add_argument("--max-epochs", type=int, default=None, help="cap the epochs of every trial (default: train.max_epochs)")
    p.add_argument("--timeout", type=float, default=None, help="stop the search after this many seconds")
    return p


def run(args: argparse.Namespace) -> None:
    """Run the search and write ``best_params.json``."""
    from atandn.training import build_objective, run_tpe_search
    from atandn.utils.io import save_json
    from atandn.utils.seed import set_seed

    cfg, search, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, cfg, paths_cfg)
    n_trials = args.n_trials if args.n_trials is not None else search.n_trials
    seed = args.seed if args.seed is not None else search.seed
    max_epochs = args.max_epochs if args.max_epochs is not None else search.max_epochs
    if n_trials < 1:
        raise ScriptError("--n-trials must be >= 1", 2)

    set_seed(seed)
    mission = load_mission(cfg, rp.data_root)
    prepare_cfg_for_data(cfg, mission.values.shape[1])
    train, _test_unused = mission.split(cfg.data.train_ratio)   # the test part is dropped here and never used
    del _test_unused, mission
    log.info("tuning on the TRAINING split only: %d samples, %d channels, %d trials",
             len(train), cfg.model.in_channels, n_trials)

    t_start = time.time()
    objective = build_objective(cfg, train.values, train.labels, device=args.device, fixed=search.fixed or None,
                                max_epochs=max_epochs)
    study = run_tpe_search(objective, n_trials, seed, n_startup_trials=search.n_startup_trials, timeout=args.timeout)
    duration_s = time.time() - t_start
    trials_path = write_trials(study, rp.out_dir / "trials.csv")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    log.info("wrote %s (%d trials, %d pruned, %.0f s)", trials_path, len(study.trials), n_pruned, duration_s)
    complete = [t for t in study.trials if t.value is not None]
    if not complete:
        raise ScriptError("no trial completed (all pruned or failed); check the data and the search settings")
    params = {**params_from_cfg(cfg), **search.fixed, **study.best_params}
    out = save_json({
        "mission": cfg.name, "params": params, "best_value": float(study.best_value),
        "best_trial": study.best_trial.number, "n_trials": len(study.trials), "n_completed": len(complete),
        "seed": seed, "selected_on": "event-aware validation block of the training split",
        "provenance": {"duration_s": round(duration_s, 1), "n_pruned": n_pruned, "trials_csv": trials_path.name,
                       "n_startup_trials": search.n_startup_trials, "max_epochs": max_epochs,
                       "versions": library_versions()},
    }, rp.best_params)
    log.info("best value %.4f (trial %d); wrote %s", study.best_value, study.best_trial.number, out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
