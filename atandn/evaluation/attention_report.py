"""Attention-logit bound report (Proposition attn of the paper) computed from the weights of a model."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from atandn.layers.attention import MultiHeadAttention1D
from atandn.layers.bign import BIGN1D

log = logging.getLogger(__name__)


def _spec(m: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(m.double(), ord=2))


@torch.no_grad()
def attention_norms(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Spectral norms of the per-head query/key projections and the logit bound Lambda of every attention block.

    ``nn.MultiheadAttention.in_proj_weight`` (3C x C) is sliced into ``W_Q = in_proj_weight[:C]`` and
    ``W_K = in_proj_weight[C:2C]``, ``in_proj_bias`` into ``b_Q``, ``b_K``, and each is split into ``H`` head blocks of
    ``d_k = C/H`` rows. Spectral norm: ``torch.linalg.matrix_norm(ord=2)``; vector norm: Euclidean.

    Lambda is Eq. ``eq:Lambda`` of ``s4_architecture.tex`` (lines 135-137, Prop. ``prop:attn``), per head h::

        Lambda_h = (||W_Q,h|| R + ||b_Q,h||) (||W_K,h|| R + ||b_K,h||) / sqrt(d_k),   R^2 = sum_i 1 / gamma_ii

    (``R`` of Prop. ``prop:bounded``(a), taken from the BIGN layer directly before the attention in the same encoder
    block; note the paper's formula uses ``R``, not ``R^2``); ``Lambda = max_h Lambda_h``. This is the paper's
    statement for a head fed directly by BIGN. A block that does not directly follow a BIGN layer (the bottleneck, or
    ``norm != "bign"``) has ``R = Lambda = None``.
    With ``pe_mode="attention"`` the attention input is ``y + PE``, so ``||input|| <= R + max_t ||PE_t||``; then
    ``Lambda_with_pe`` uses that radius (the paper's Lambda is not a valid bound there) and ``pe_norm_max`` is given.

    Returns:
        One dict per attention block: ``layer``, ``C``, ``H``, ``d_k``, ``spec_WQ`` / ``spec_WK`` (max over heads),
        ``spec_WQ_heads`` / ``spec_WK_heads``, ``prod`` (max over heads of ``||W_Q,h|| ||W_K,h||``), ``bQ`` / ``bK``
        (max head bias norm), ``R``, ``Lambda``, ``pe_norm_max``, ``Lambda_with_pe``.
    """
    out: list[dict[str, Any]] = []
    for pname, parent in model.named_modules():
        if not isinstance(parent, nn.Sequential):
            continue
        kids = list(parent.named_children())
        for i, (cname, mod) in enumerate(kids):
            if not isinstance(mod, MultiHeadAttention1D):
                continue
            mha = mod.mha
            c, h = mha.embed_dim, mha.num_heads
            dk = c // h
            w, b = mha.in_proj_weight.detach(), mha.in_proj_bias.detach()
            wq, wk = w[:c].reshape(h, dk, c), w[c:2 * c].reshape(h, dk, c)
            bq, bk = b[:c].reshape(h, dk), b[c:2 * c].reshape(h, dk)
            sq = [_spec(wq[j]) for j in range(h)]
            sk = [_spec(wk[j]) for j in range(h)]
            nq = [float(torch.linalg.vector_norm(bq[j].double())) for j in range(h)]
            nk = [float(torch.linalg.vector_norm(bk[j].double())) for j in range(h)]
            prev = kids[i - 1][1] if i > 0 else None
            r = None
            if isinstance(prev, BIGN1D):
                r = float((1.0 / torch.diagonal(prev.gamma.detach().double())).sum().sqrt())
            pe_norm = None
            if mod.pos_enc is not None:
                pe_norm = float(torch.linalg.vector_norm(mod.pos_enc.pe.double()[0], dim=0).max())

            def lam(radius: float) -> float:
                return max((sq[j] * radius + nq[j]) * (sk[j] * radius + nk[j]) / dk ** 0.5 for j in range(h))

            out.append({
                "layer": f"{pname}.{cname}" if pname else cname, "C": int(c), "H": int(h), "d_k": int(dk),
                "spec_WQ": max(sq), "spec_WK": max(sk), "spec_WQ_heads": sq, "spec_WK_heads": sk,
                "prod": max(a * b_ for a, b_ in zip(sq, sk)), "bQ": max(nq), "bK": max(nk),
                "R": r, "Lambda": None if r is None else lam(r),
                "pe_norm_max": pe_norm,
                "Lambda_with_pe": None if (r is None or pe_norm is None) else lam(r + pe_norm),
            })
    return out
