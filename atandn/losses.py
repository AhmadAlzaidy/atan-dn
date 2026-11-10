"""Loss functions: focal loss and the reconstruction-regularized weighted focal loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal loss on logits: ``alpha * (1 - p_t)^gamma * BCE`` with optional pos_weight.

    Standalone alpha-balanced variant for external use. It is NOT used by :func:`rr_wfl_loss`, which keeps the
    legacy ``combined_loss`` form without alpha (``(1 - exp(-l))^gamma * l``); the two differ by the factor alpha.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Mean focal loss; logits and targets have the same shape."""
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=logits.new_tensor([self.pos_weight]), reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def rr_wfl_loss(recon: Tensor | None, target_x: Tensor, logits: Tensor, target_y: Tensor,
                recon_weight: float = 0.1, pos_weight: float = 5.0, focal_gamma: float = 2.0,
                label_smoothing: float = 0.05, classification: str = "focal"
                ) -> tuple[Tensor, dict[str, Tensor]]:
    """Reconstruction-regularized weighted focal loss.

    ``loss = recon_weight * MSE + (1 - recon_weight) * clf`` where, for
    ``classification="focal"``, ``clf = mean[(1 - exp(-l))^gamma * l]`` with ``l`` the
    pos-weighted BCE on smoothed targets ``y*(1-s) + 0.5*s`` (legacy ``combined_loss``).
    ``"bce"`` is the unweighted BCE mean, ``"weighted_bce"`` the pos-weighted mean.
    If ``recon`` is None, the loss is ``clf`` alone (no reconstruction term).

    Args:
        logits: (B,) or (B, 1). target_y: (B,) in {0, 1}.

    Returns:
        (loss, components) with detached ``mse`` (absent if recon is None) and ``clf``.
    """
    logits = logits.reshape(-1)
    y = target_y.reshape(-1).to(logits.dtype)
    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + 0.5 * label_smoothing
    if classification == "focal":
        bce = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=logits.new_tensor([pos_weight]), reduction="none")
        clf = ((1 - torch.exp(-bce)) ** focal_gamma * bce).mean()
    elif classification == "bce":
        clf = F.binary_cross_entropy_with_logits(logits, y)
    elif classification == "weighted_bce":
        clf = F.binary_cross_entropy_with_logits(logits, y, pos_weight=logits.new_tensor([pos_weight]))
    else:
        raise ValueError(f"classification must be focal|bce|weighted_bce, got {classification!r}")
    comps: dict[str, Tensor] = {"clf": clf.detach()}
    if recon is None:
        return clf, comps
    mse = F.mse_loss(recon, target_x)
    comps["mse"] = mse.detach()
    return recon_weight * mse + (1 - recon_weight) * clf, comps
