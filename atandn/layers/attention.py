"""Positional encoding, multi-head attention and squeeze-excitation for (B, C, T) tensors."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding1D(nn.Module):
    """Sinusoidal positional encoding added to a (B, C, T) tensor (C = model dim)."""

    def __init__(self, d_model: int, max_len: int = 500, persistent: bool = True) -> None:
        super().__init__()
        d_even = d_model + (d_model % 2)
        pe = torch.zeros(max_len, d_even)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_even, 2).float() * (-math.log(10000.0) / d_even))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe[:, :d_model].t().unsqueeze(0), persistent=persistent)  # (1, C, max_len); no parameters

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, C, T); requires T <= max_len."""
        return x + self.pe[:, : x.size(1), : x.size(2)]


class MultiHeadAttention1D(nn.Module):
    """Self-attention over time for (B, C, T); channels are the embedding dimension.

    If ``pe_max_len`` is given, a sinusoidal positional encoding with ``d_model = num_channels`` is added to the
    input of the attention (``X_att = X + PE``, paper Eq. pe-cases). It has no parameters and its buffer is not
    stored in the state dict (it is a deterministic function of the sizes).
    """

    def __init__(self, num_channels: int, num_heads: int = 4, dropout: float = 0.2,
                 pe_max_len: int | None = None) -> None:
        super().__init__()
        self.pos_enc = (PositionalEncoding1D(num_channels, pe_max_len, persistent=False)
                        if pe_max_len is not None else None)
        self.mha = nn.MultiheadAttention(num_channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, C, T)."""
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        h = x.transpose(1, 2)
        out, _ = self.mha(h, h, h, need_weights=False)
        return out.transpose(1, 2)


class SqueezeExcitation1D(nn.Module):
    """Channel recalibration: global average over time, bottleneck MLP, sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T) -> (B, C, T)."""
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=2)))))
        return x * gate.unsqueeze(2)
