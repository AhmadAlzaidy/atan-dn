"""The ONE training prelude shared by ``scripts/train.py`` and ``experiments/protocol.py``."""
from __future__ import annotations

import logging
from dataclasses import asdict

import torch

from atandn.config import MissionConfig
from atandn.data.splits import MissionData, ValidationSplit
from atandn.data.windows import WindowDataset
from atandn.training.trainer import Trainer, build_loaders

log = logging.getLogger(__name__)


def fit_on_split(cfg: MissionConfig, train: MissionData, split: ValidationSplit,
                 device: str = "auto") -> tuple[torch.nn.Module, dict]:
    """Build the window datasets of the fitting part and the validation block, train, return the selected model.

    Args:
        cfg: validated mission config (``model``, ``train``).
        train: the TRAINING split of the stream (the test split is never passed here).
        split: validation split of ``train`` (see :func:`atandn.data.splits.event_aware_validation_split`).
        device: ``"auto"`` or a torch device string.

    Returns:
        ``(model, result)``: the model in eval mode holding the selected (best-epoch EMA) weights, and the dict of
        :meth:`atandn.training.trainer.Trainer.fit` (``history``, ``best_epoch``, ``best_score``, ...).
    """
    from atandn.models import ATANDN

    w = cfg.model.window_size
    stride = cfg.train.train_stride(w)
    fit_ds = WindowDataset(train.values[split.fit], train.labels[split.fit], w, stride, augment=cfg.train.use_augmentation)
    val_ds = WindowDataset(train.values[split.val], train.labels[split.val], w, stride)
    log.info("fit windows %d, validation-block windows %d (both from the training split)", len(fit_ds), len(val_ds))
    fit_loader, val_loader = build_loaders(fit_ds, val_ds, cfg.train)
    model = ATANDN(**{**asdict(cfg.model), "conv_filters": tuple(cfg.model.conv_filters)})
    trainer = Trainer(model, cfg.train, device)
    result = trainer.fit(fit_loader, val_loader)
    trainer.model.eval()
    return trainer.model, result
