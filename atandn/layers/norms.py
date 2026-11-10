"""Normalization layers with a common (B, C, T) -> (B, C, T) interface (swap ablation)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .bign import BIGN1D

NORM_KINDS = ("bign", "identity", "layernorm", "instancenorm", "groupnorm", "batchnorm", "rmsnorm")


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis at each time step (same axis as BIGN)."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, C, T)."""
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ChannelRMSNorm(nn.Module):
    """RMSNorm over channels at each time step with a learned per-channel scale."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, C, T)."""
        rms = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * rms * self.weight.view(1, -1, 1)


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def build_norm(kind: str, channels: int, **kw) -> nn.Module:
    """Build a normalization layer.

    Args:
        kind: one of ``NORM_KINDS``. ``instancenorm`` normalizes over time per channel.
        channels: number of channels C.
        **kw: forwarded to BIGN1D for ``bign`` (beta_min, gamma_init).

    Returns:
        Module mapping (B, C, T) to (B, C, T).
    """
    if kind == "bign":
        return BIGN1D(channels, **kw)
    if kind == "identity":
        return nn.Identity()
    if kind == "layernorm":
        return ChannelLayerNorm(channels)
    if kind == "instancenorm":
        return nn.InstanceNorm1d(channels, affine=True)
    if kind == "groupnorm":
        return nn.GroupNorm(_groups(channels), channels)
    if kind == "batchnorm":
        return nn.BatchNorm1d(channels)
    if kind == "rmsnorm":
        return ChannelRMSNorm(channels)
    raise ValueError(f"unknown norm kind {kind!r}; expected one of {NORM_KINDS}")
