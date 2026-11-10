"""Hyperparameter search: Optuna TPE with a MedianPruner (search space of the legacy objective).

The objective maximises the model-selection score on the event-aware validation block of the TRAINING split (threshold-free average precision of the dual-evidence
score; validation loss when the block has no positive).
The test split is never touched here.
"""
from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np

from atandn.config import MissionConfig

log = logging.getLogger(__name__)


def suggest_params(trial, fixed: dict | None = None) -> dict:
    """Legacy search space (``m1_new.py::train_enhanced``; identical in m2_new.py and ops_new.py).

    ``latent_dim`` of the legacy space is dropped (the contract ``ATANDN`` has no such argument). Entries of
    ``fixed`` replace the corresponding suggestion (e.g. ``{"window_size": 300}`` to keep the window fixed).
    """
    fixed = fixed or {}

    def pick(name, fn):
        return fixed[name] if name in fixed else fn()

    p = {
        "window_size": pick("window_size", lambda: trial.suggest_int("window_size", 200, 400, step=50)),
        "overlap": pick("overlap", lambda: trial.suggest_float("overlap", 0.75, 0.92)),
        "pos_weight": pick("pos_weight", lambda: trial.suggest_float("pos_weight", 3.0, 25.0)),
        "lr": pick("lr", lambda: trial.suggest_float("lr", 1e-5, 1e-3, log=True)),
        "recon_weight": pick("recon_weight", lambda: trial.suggest_float("recon_weight", 0.1, 0.4)),
        "dropout": pick("dropout", lambda: trial.suggest_float("dropout", 0.15, 0.45)),
        "batch_size": pick("batch_size", lambda: trial.suggest_categorical("batch_size", [32, 48, 64])),
        "focal_gamma": pick("focal_gamma", lambda: trial.suggest_float("focal_gamma", 1.0, 3.0)),
        "weight_decay": pick("weight_decay", lambda: trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)),
        "recon_score_weight": pick("recon_score_weight", lambda: trial.suggest_float("recon_score_weight", 0.0, 0.5)),
        "num_heads": pick("num_heads", lambda: trial.suggest_categorical("num_heads", [2, 4, 8])),
        "conv_f1": pick("conv_f1", lambda: trial.suggest_categorical("conv_f1", [64, 96, 128])),
        "conv_f2": pick("conv_f2", lambda: trial.suggest_categorical("conv_f2", [128, 192, 256])),
        "use_weighted_sampling": pick("use_weighted_sampling", lambda: trial.suggest_categorical("use_weighted_sampling", [True, False])),
        "use_augmentation": pick("use_augmentation", lambda: trial.suggest_categorical("use_augmentation", [True, False])),
    }
    return p


def apply_params(base: MissionConfig, p: dict) -> MissionConfig:
    """Return a copy of ``base`` with the suggested hyperparameters applied (and validated)."""
    cfg = copy.deepcopy(base)
    cfg.model.window_size = int(p["window_size"])
    cfg.model.conv_filters = (int(p["conv_f1"]), int(p["conv_f2"]))
    cfg.model.num_heads = int(p["num_heads"])
    cfg.model.dropout = float(p["dropout"])
    t = cfg.train
    for k in ("overlap", "pos_weight", "lr", "recon_weight", "focal_gamma", "weight_decay", "recon_score_weight"):
        setattr(t, k, float(p[k]))
    t.batch_size = int(p["batch_size"])
    t.use_weighted_sampling = bool(p["use_weighted_sampling"])
    t.use_augmentation = bool(p["use_augmentation"])
    cfg.validate()
    return cfg


def build_objective(
    base_cfg: MissionConfig,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    model_factory: Callable | None = None,
    device: str = "auto",
    fixed: dict | None = None,
    max_epochs: int | None = None,
) -> Callable:
    """Create an Optuna objective that trains one configuration and returns its validation-block score.

    Args:
        base_cfg: Mission config providing the non-searched fields.
        train_values / train_labels: the TRAINING split only, ``(N, C)`` and ``(N,)``.
        model_factory: ``factory(model_cfg) -> nn.Module``; default builds ``atandn.models.ATANDN``.
        fixed: Search-space entries to hold fixed (see :func:`suggest_params`).
        max_epochs: Optional override of ``TrainConfig.max_epochs`` (smoke tests).
    """
    import optuna

    from atandn.data.splits import event_aware_validation_split
    from atandn.data.windows import WindowDataset
    from atandn.training.trainer import Trainer, build_loaders
    from atandn.utils.seed import set_seed

    def default_factory(mc):
        from atandn.models import ATANDN
        return ATANDN(**{**mc.__dict__, "conv_filters": tuple(mc.conv_filters)})

    factory = model_factory or default_factory

    warned_windows: set = set()      # the split depends on the window size only: warn once per distinct size

    def objective(trial) -> float:
        p = suggest_params(trial, fixed)
        if p["conv_f1"] % p["num_heads"] or p["conv_f2"] % p["num_heads"]:
            raise optuna.TrialPruned("conv filters must be divisible by num_heads")
        cfg = apply_params(base_cfg, p)
        if max_epochs is not None:
            cfg.train.max_epochs = max_epochs
        set_seed(cfg.train.seed + trial.number)
        w = cfg.model.window_size
        vs = event_aware_validation_split(train_labels, cfg.train.val_fraction, w - 1, cfg.train.val_min_events,
                                          min_fit=w, min_fit_fraction=cfg.train.val_min_fit_fraction,
                                          warn=w not in warned_windows)
        warned_windows.add(w)
        fit_sl, val_sl = vs.fit, vs.val
        fit_ds = WindowDataset(train_values[fit_sl], train_labels[fit_sl], w, cfg.train.train_stride(w), augment=cfg.train.use_augmentation)
        val_ds = WindowDataset(train_values[val_sl], train_labels[val_sl], w, cfg.train.train_stride(w))
        fit_loader, val_loader = build_loaders(fit_ds, val_ds, cfg.train)
        trainer = Trainer(factory(cfg.model), cfg.train, device)

        def cb(epoch: int, score: float, _hist: dict) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        out = trainer.fit(fit_loader, val_loader, epoch_callback=cb)
        trial.set_user_attr("best_epoch", out["best_epoch"])
        trial.set_user_attr("has_val_positives", out["has_val_positives"])
        trial.set_user_attr("val_metric", out["val_metric"])
        return out["best_score"]

    return objective


def run_tpe_search(
    objective: Callable,
    n_trials: int,
    seed: int,
    study_name: str | None = None,
    storage: str | None = None,
    n_startup_trials: int = 10,
    pruner_startup_trials: int = 5,
    pruner_warmup_steps: int = 10,
    timeout: float | None = None,
):
    """Run Optuna TPE (maximise) with a MedianPruner (legacy settings: 10 startup trials, pruner 5/10).

    Args:
        objective: ``objective(trial) -> float`` (see :func:`build_objective`).

    Returns:
        The finished ``optuna.Study``.
    """
    import optuna

    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage, load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials, n_warmup_steps=pruner_warmup_steps),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    if study.best_trial is not None:
        log.info("best value %.4f params %s", study.best_value, study.best_params)
    return study


class LegacyDefaultsError(ValueError):
    """Raised when training would use default configurations without an explicit flag or best-params file."""


def check_legacy_defaults(has_best_params: bool, allow_legacy_defaults: bool, where: str = "this run") -> bool:
    """Ensure parameters are explicitly verified or loaded from best-params.

    Args:
        has_best_params: True if a best-params file (from ``tune.py``) is used.
        allow_legacy_defaults: the ``--allow-legacy-defaults`` flag.
        where: name of the caller for the messages.

    Returns:
        ``used_legacy_defaults`` (True only when no best-params file is used and the flag is given).

    Raises:
        LegacyDefaultsError: if there is neither a best-params file nor the explicit flag.
    """
    if has_best_params:
        return False
    if not allow_legacy_defaults:
        raise LegacyDefaultsError(
            f"{where}: no best-params file given. Run scripts/tune.py and pass --best-params, or pass "
            "--allow-legacy-defaults to proceed with predefined mission defaults.")
    log.info("%s: using predefined mission configuration defaults.", where)
    return True
