"""ATAN-DN: attention autoencoder with BIGN normalization and a classifier head."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.attention import MultiHeadAttention1D, PositionalEncoding1D, SqueezeExcitation1D
from ..layers.bign import BIGN1D, ExactInverseBIGN1D, MirrorBIGN1D
from ..layers.norms import build_norm


class ATANDN(nn.Module):
    """Reconstruction-regularized classifier for multivariate telemetry windows.

    Shapes (T = window_size, C1, C2 = conv_filters, T1 = ceil(T/2), T2 = ceil(T1/2)):
    input (B, C_in, T) -> encoder block 1 (B, C1, T1) -> block 2 (B, C2, T2)
    -> bottleneck attention + LayerNorm([C2, T2]) = latent (B, C2, T2)
    -> decoder (B, C1, T1) -> (B, C_in, T) [interpolated to T if needed], and
    classifier on flatten(latent) (B, C2*T2) -> logits (B,).

    Encoder block: Conv1d(k5,s2,p2) -> BatchNorm -> norm -> MHA -> SE -> ReLU -> Dropout.
    Decoder block: ConvTranspose1d(k5,s2,p2,op1) -> BatchNorm -> inverse -> ReLU -> Dropout;
    the last block is linear (no ReLU, no dropout) so negative standardized targets
    are reachable. ``inverse`` in {"exact", "mirror", "none"}; with "exact" and
    ``norm == "bign"`` the first decoder layer is tied to the first encoder BIGN layer
    (shared tensors, counted once by ``parameters()``); the last has its own parameters.

    Args:
        gamma_init_mode / gamma_diag / gamma_off: Gamma init of all BIGN-type layers
            ("diag_dominant" default, "uniform" = legacy), see ``bign._init_params``.
        inverse_method: "solve" (default, documented; exact where the clamp does not fire, see
            ``docs/BIGN_INVERSE.md``) or "neumann" (K = ``neumann_steps`` fixed-point steps: an APPROXIMATION,
            relative round-trip error 0.2-0.5 at unit scale for K = 4).
        norm: kind passed to ``build_norm`` after each encoder BatchNorm.
        pe_mode: "attention" (default, paper) adds the sinusoidal PE to the input of every MultiHeadAttention1D
            (both encoder blocks and the bottleneck) with d_model = that block's width; "input" (legacy) adds it once
            to the raw window with d_model = C_in. Neither has parameters; with ``use_attention=False`` the
            "attention" mode has no PE at all.
        use_attention / use_se / use_reconstruction: ablation switches. Without
            reconstruction the decoder is not built and ``forward`` returns None.
    """

    def __init__(self, in_channels: int, window_size: int = 300,
                 conv_filters: tuple[int, int] = (64, 128), num_heads: int = 4,
                 dropout: float = 0.2, norm: str = "bign", inverse: str = "exact",
                 inverse_method: str = "solve", neumann_steps: int = 4,
                 use_attention: bool = True, use_reconstruction: bool = True,
                 use_se: bool = True, beta_min: float = 1e-6,
                 gamma_init_mode: str = "diag_dominant", gamma_diag: float = 0.5,
                 gamma_off: float | None = None, pe_mode: str = "attention") -> None:
        super().__init__()
        if inverse not in ("exact", "mirror", "none"):
            raise ValueError(f"inverse must be exact|mirror|none, got {inverse!r}")
        if pe_mode not in ("attention", "input"):
            raise ValueError(f"pe_mode must be 'attention' or 'input', got {pe_mode!r}")
        for i, c in enumerate(conv_filters):
            if use_se and c % 8 != 0:
                raise ValueError(f"conv_filters[{i}]={c} must be divisible by 8 for Squeeze-and-Excitation (reduction ratio r=8)")
            if use_attention and c % num_heads != 0:
                raise ValueError(f"conv_filters[{i}]={c} must be divisible by num_heads={num_heads}")
        f1, f2 = conv_filters
        self.pe_mode = pe_mode
        # 'input' (legacy): PE on the raw window, d_model = C_in. 'attention' (paper): PE inside every MHA, d_model = C_l.
        self.pos_enc: nn.Module = (PositionalEncoding1D(in_channels, window_size) if pe_mode == "input"
                                   else nn.Identity())
        mha_pe = window_size if pe_mode == "attention" else None

        ginit = dict(gamma_init_mode=gamma_init_mode, gamma_diag=gamma_diag, gamma_off=gamma_off)
        norm_kw = {"beta_min": beta_min, **ginit} if norm == "bign" else {}
        blocks: list[nn.Module] = []
        c_prev = in_channels
        for c_out in conv_filters:
            layers: list[nn.Module] = [nn.Conv1d(c_prev, c_out, 5, 2, 2), nn.BatchNorm1d(c_out),
                                       build_norm(norm, c_out, **norm_kw)]
            if use_attention:
                layers.append(MultiHeadAttention1D(c_out, num_heads, dropout, pe_max_len=mha_pe))
            if use_se:
                layers.append(SqueezeExcitation1D(c_out))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            blocks.append(nn.Sequential(*layers))
            c_prev = c_out
        self.encoder_blocks = nn.ModuleList(blocks)

        t = window_size
        for _ in conv_filters:
            t = (t + 4 - 5) // 2 + 1
        self.bottleneck_len = t
        bott: list[nn.Module] = []
        if use_attention:
            bott.append(MultiHeadAttention1D(f2, num_heads, dropout, pe_max_len=mha_pe))
        bott.append(nn.LayerNorm([f2, t]))
        self.bottleneck = nn.Sequential(*bott)

        self.decoder: nn.Sequential | None = None
        if use_reconstruction:
            first = self.encoder_blocks[0][2]
            tied = first if (inverse == "exact" and isinstance(first, BIGN1D)) else None
            dec: list[nn.Module] = []
            outs = [f1, in_channels]
            c_prev = f2
            for i, c_out in enumerate(outs):
                dec += [nn.ConvTranspose1d(c_prev, c_out, 5, 2, 2, output_padding=1), nn.BatchNorm1d(c_out)]
                if inverse == "exact":
                    dec.append(ExactInverseBIGN1D(c_out, tied_to=tied if i == 0 else None,
                                                  method=inverse_method, neumann_steps=neumann_steps,
                                                  beta_min=beta_min, **ginit))
                elif inverse == "mirror":
                    dec.append(MirrorBIGN1D(c_out, beta_min=beta_min, **ginit))
                if i < len(outs) - 1:
                    dec += [nn.ReLU(), nn.Dropout(dropout)]
                c_prev = c_out
            self.decoder = nn.Sequential(*dec)

        self.classifier = nn.Sequential(
            nn.Linear(f2 * t, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor | None, Tensor]:
        """x: (B, C_in, T) window-z-normalized -> (recon (B, C_in, T) | None, logits (B,))."""
        h = self.pos_enc(x)
        for block in self.encoder_blocks:
            h = block(h)
        latent = self.bottleneck(h)
        recon = None
        if self.decoder is not None:
            recon = self.decoder(latent)
            if recon.size(2) != x.size(2):
                recon = F.interpolate(recon, size=x.size(2), mode="linear", align_corners=False)
        logits = self.classifier(latent.flatten(1)).squeeze(-1)
        return recon, logits

    def bign_layers(self) -> list[BIGN1D]:
        """All BIGN1D modules (encoder norm layers when ``norm == "bign"``)."""
        return [m for m in self.modules() if isinstance(m, BIGN1D)]
