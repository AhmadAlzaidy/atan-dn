"""Exponential moving average of model parameters."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


class EMA:
    """EMA of trainable parameters (``shadow = d * shadow + (1 - d) * param``), as in the legacy code.

    Buffers (e.g. BatchNorm running statistics) are not averaged; they are used as they are in the model.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model, self.decay = model, decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self) -> None:
        """Update the shadow weights from the current model parameters."""
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        """Swap the shadow weights into the model (call :meth:`restore` afterwards)."""
        self._backup = {}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.detach().clone()
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self) -> None:
        """Put the original (non-averaged) weights back."""
        for n, p in self.model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = {}

    @contextmanager
    def average_weights(self) -> Iterator[None]:
        """Context manager: evaluate with the EMA weights, restore on exit."""
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()
