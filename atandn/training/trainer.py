"""Trainer: AdamW, warm-up, cosine warm restarts, gradient clipping, EMA, early stopping.

Model selection and early stopping use ONLY the validation loader, which must be built from the validation block
of the TRAINING split (``atandn.data.splits.event_aware_validation_split``), never the test split. The selection
score is threshold-free (average precision of the dual-evidence score; validation loss if the block has no positive).
"""
from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from atandn.config import TrainConfig
from atandn.inference.scoring import average_precision
from atandn.losses import rr_wfl_loss
from atandn.training.ema import EMA
from atandn.utils.io import resolve_device

log = logging.getLogger(__name__)


def _clamp_rates(model: torch.nn.Module) -> list[torch.Tensor]:
    """Lazily stored clamp-rate tensors of every ExactInverseBIGN1D (no host sync here)."""
    from atandn.layers.bign import ExactInverseBIGN1D
    out = []
    for m in model.modules():
        if isinstance(m, ExactInverseBIGN1D) and m.last_clamp_rate_tensor is not None:
            out.append(m.last_clamp_rate_tensor.detach())
    return out


def resolve_amp(amp: object, device: torch.device) -> tuple[bool, torch.dtype | None]:
    """Resolve ``TrainConfig.amp`` to ``(enabled, autocast dtype)``.

    ``"auto"``: on only for CUDA (float16 + GradScaler). ``True`` on CUDA: float16. ``True`` on CPU: bfloat16
    autocast (no scaler needed; a message is logged, speed is not expected to improve). ``False``: off.
    """
    if amp == "auto":
        amp = device.type == "cuda"
    if not amp:
        return False, None
    if device.type == "cuda":
        return True, torch.float16
    log.warning("amp=True on %s: using bfloat16 autocast (no GradScaler); BIGN layers still run in float32", device.type)
    return True, torch.bfloat16


def build_loaders(fit_ds, val_ds, cfg: TrainConfig) -> tuple[DataLoader, DataLoader | None]:
    """Create the fit loader (optionally class-balanced via a weighted sampler) and the validation loader."""
    if cfg.use_weighted_sampling and sum(fit_ds.window_labels) > 0:
        w = fit_ds.sampler_weights()
        sampler = WeightedRandomSampler(w, len(w), replacement=True)
        fit = DataLoader(fit_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        fit = DataLoader(fit_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) if val_ds is not None else None
    return fit, val


class Trainer:
    """Trains an ``ATANDN``-style model whose forward returns ``(recon | None, logits)``."""

    def __init__(self, model: torch.nn.Module, train_cfg: TrainConfig, device: str = "auto"):
        self.cfg = train_cfg
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=train_cfg.cosine_t0, T_mult=train_cfg.cosine_t_mult, eta_min=train_cfg.lr * train_cfg.eta_min_factor)
        self.ema = EMA(self.model, train_cfg.ema_decay)
        self._epoch_clamp: list[float] = []
        self._warned_no_pos = False
        self.amp, self.amp_dtype = resolve_amp(train_cfg.amp, self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.amp_dtype == torch.float16)

    def _autocast(self):
        """Autocast context of the forward pass (a no-op context when amp is off)."""
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.float32, enabled=self.amp)

    def _loss(self, recon, x, logits, y):
        c = self.cfg
        if self.amp:   # the loss is always evaluated in float32
            recon = None if recon is None else recon.float()
            logits = logits.float()
        return rr_wfl_loss(recon, x, logits, y, recon_weight=c.recon_weight, pos_weight=c.pos_weight,
                           focal_gamma=c.focal_gamma, label_smoothing=c.label_smoothing, classification=c.classification)[0]

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        rates: list[list[torch.Tensor]] = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                recon, logits = self.model(x)
            loss = self._loss(recon, x, logits, y)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)   # clip the true gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            self.ema.update()
            rates.append(_clamp_rates(self.model))
            total += float(loss.item())
            n += 1
        # one host sync per layer per epoch instead of per step
        self._epoch_clamp = ([float(torch.stack(list(col)).mean().item()) for col in zip(*rates)]
                             if rates and rates[0] else [])
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, bool]:
        """Selection score on the validation block with EMA weights.

        Threshold-free: average precision of the dual-evidence score ``(1-a)*prob + a*err_norm`` (``a`` =
        ``recon_score_weight``, forced to 0 for a model without reconstruction head; ``err_norm`` uses the 1st/99th
        percentile of the block's own reconstruction errors, no labels involved). If the block contains no
        positive the score is minus the validation loss (logged once). Returns ``(score, has_positives)``.
        """
        probs, errs, ys, losses = [], [], [], []
        has_recon = True
        with self.ema.average_weights():
            self.model.eval()
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                with self._autocast():
                    recon, logits = self.model(x)
                if self.amp:
                    recon = None if recon is None else recon.float()
                    logits = logits.float()
                has_recon = has_recon and recon is not None
                losses.append(float(self._loss(recon, x, logits, y).item()))
                probs.append(torch.sigmoid(logits.reshape(len(x), -1)[:, 0]).cpu().numpy())
                errs.append(np.zeros(len(x)) if recon is None else ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
                ys.append(y.cpu().numpy())
        p, e, y = np.concatenate(probs), np.concatenate(errs), np.concatenate(ys)
        if not (y > 0).any():
            if not self._warned_no_pos:
                log.warning("validation block has no positive window: model selection uses the validation loss")
                self._warned_no_pos = True
            return -float(np.mean(losses)), False
        a = self.cfg.recon_score_weight if has_recon else 0.0
        lo, hi = np.percentile(e, [1.0, 99.0])
        e_norm = np.clip((e - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(e)
        score = (1 - a) * p + a * e_norm
        metric = getattr(self.cfg, "val_metric", "f1")
        if metric == "f1":
            from ..inference.threshold import grid_f1_search
            f1_opt, _ = grid_f1_search(y, score)
            return float(f1_opt), True
        return average_precision(score, y), True

    def _set_lr(self, epoch: int) -> None:
        c = self.cfg
        if epoch < c.warmup_epochs:
            for g in self.optimizer.param_groups:
                g["lr"] = c.lr * (epoch + 1) / c.warmup_epochs
        elif epoch == c.warmup_epochs:
            for g in self.optimizer.param_groups:
                g["lr"] = c.lr

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
            epoch_callback: Callable[[int, float, dict], None] | None = None) -> dict:
        """Train; keep the best EMA weights according to the validation score and load them into the model.

        Args:
            train_loader: fitting windows.
            val_loader: validation block of the TRAINING split (if None, early stopping uses the
                training loss and the last weights are kept).
            epoch_callback: ``callback(epoch, val_score, history)`` called after every epoch; it may raise
                (e.g. ``optuna.TrialPruned``) to stop training.

        Returns:
            dict with ``amp`` (effective), ``amp_dtype``, ``best_score``, ``best_epoch``, ``epochs_run``, ``stopped_early``, ``has_val_positives``, ``val_metric``, ``history``.
        """
        c = self.cfg
        best, best_epoch, bad, best_state = -np.inf, -1, 0, None
        history: list[dict] = []
        has_pos = False
        for epoch in range(c.max_epochs):
            self._set_lr(epoch)
            loss = self._train_epoch(train_loader)
            if epoch >= c.warmup_epochs:
                self.scheduler.step()
            if val_loader is not None:
                score, has_pos = self._validate(val_loader)
            else:
                score = -loss
            cl = self._epoch_clamp
            mean_clamp = float(np.mean(cl)) if cl else None
            history.append({"epoch": epoch, "train_loss": loss, "val_score": score, "lr": self.optimizer.param_groups[0]["lr"],
                            "clamp_rate": mean_clamp, "clamp_rates": cl})
            if mean_clamp is not None:
                log.info("epoch %d | mean ExactInverseBIGN1D clamp rate %.4f (per layer %s)", epoch + 1, mean_clamp,
                         ["%.4f" % r for r in cl])
                if c.clamp_warn_epochs > 0 and epoch + 1 >= c.clamp_warn_epochs and mean_clamp > c.clamp_warn_threshold:
                    log.warning("decoder inverse clamp rate %.3f > %.2f after %d epochs: the exact inverse is mostly the clamp",
                                mean_clamp, c.clamp_warn_threshold, epoch + 1)
            log.info("epoch %d | loss %.4f | val %.4f | lr %.2e", epoch + 1, loss, score, history[-1]["lr"])
            if score > best + c.min_delta:
                best, best_epoch, bad = score, epoch, 0
                with self.ema.average_weights():
                    best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad += 1
            if epoch_callback is not None:
                epoch_callback(epoch, score, {"history": history})
            if bad >= c.patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif history:
            # validation score never improved on -inf (e.g. NaN in every epoch): fall back to the last EMA weights
            log.warning("validation score was NaN (or never finite) in all %d epochs: no best epoch exists; "
                        "returning the LAST EMA weights. Model selection did not take place.", len(history))
            with self.ema.average_weights():
                last_state = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(last_state)
        return {"amp": self.amp, "amp_dtype": None if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", ""),
                "best_score": float(best), "best_epoch": best_epoch, "epochs_run": len(history),
                "stopped_early": len(history) < c.max_epochs, "has_val_positives": has_pos,
                "val_metric": ((getattr(c, "val_metric", "f1") if has_pos else "val_loss") if val_loader is not None else "train_loss"),
                "history": history}
