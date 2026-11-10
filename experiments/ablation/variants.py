"""Registry of ablation variants, expressed as overrides of ``ModelConfig`` / ``TrainConfig`` / ``EvalConfig``.

Every variant is a small set of overrides on top of the SAME mission config with the SAME frozen
hyperparameters; no variant gets its own search.

Legacy note ("Without BIGN"): in the legacy ablation (``ablation/ablation.py``) the variant called
"Without BIGN" replaced the BIGN layers by an identity while everything else stayed: the per-window
z-normalization of the input, the ``BatchNorm1d`` after every convolution and the ``LayerNorm`` in the
bottleneck remain. This is exactly ``norm_identity`` here (``ModelConfig.norm = "identity"``; the decoder
inverse layers are kept, as in the full model, only the encoder normalization is swapped). The other
``norm_*`` variants swap BIGN for a standard layer at the SAME position (after each encoder BatchNorm).

Groups: ``norm`` (normalization swap), ``components`` (attention, SE, reconstruction), ``loss`` (classification
loss), ``decoder`` (decoder inverse) and ``operational`` (Stage 3: ``no_stage3`` = raw alarms, no known-event mask,
no spike filter; the threshold is selected for that raw chain on the validation block, i.e. it is re-tuned). ``full`` is the unmodified configuration and belongs to every study.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from atandn.config import MissionConfig


@dataclass(frozen=True)
class Variant:
    """A named set of config overrides.

    Attributes:
        name: registry key.
        group: ``"norm"``, ``"components"``, ``"loss"``, ``"decoder"``, ``"operational"`` or ``"base"``.
        description: one-line description used in tables.
        model / train / eval: attribute overrides of the respective config section.
    """

    name: str
    group: str
    description: str
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)


def _v(name: str, group: str, desc: str, model: dict | None = None, train: dict | None = None,
       eval: dict | None = None) -> Variant:  # noqa: A002 - mirrors the config section name
    return Variant(name, group, desc, model or {}, train or {}, eval or {})


_ALL = [
    _v("full", "base", "Full ATAN-DN (BIGN, attention, SE, reconstruction, exact inverse, focal loss)"),
    # (a) normalization swap at the same position
    _v("norm_bign", "norm", "BIGN (reference, = full)", model={"norm": "bign"}),
    _v("norm_identity", "norm", "Without BIGN (identity; z-score, BatchNorm1d, LayerNorm remain)", model={"norm": "identity"}),
    _v("standard_norm", "norm", "Standard Normalization (BatchNorm + LayerNorm)", model={"norm": "identity"}),
    _v("norm_layernorm", "norm", "LayerNorm over channels per time step", model={"norm": "layernorm"}),
    _v("norm_instancenorm", "norm", "InstanceNorm over time per channel", model={"norm": "instancenorm"}),
    _v("norm_groupnorm", "norm", "GroupNorm", model={"norm": "groupnorm"}),
    _v("norm_batchnorm", "norm", "BatchNorm1d", model={"norm": "batchnorm"}),
    _v("norm_rmsnorm", "norm", "RMSNorm over channels per time step", model={"norm": "rmsnorm"}),
    # (b) components
    _v("no_attention", "components", "Without multi-head attention", model={"use_attention": False}),
    _v("multi_head_attention", "components", "Multi-Head Attention (Full Model)", model={"use_attention": True}),
    _v("conv_feedforward", "components", "Convolutional Feed-Forward (No Attention)", model={"use_attention": False}),
    _v("no_se", "components", "Without squeeze-and-excitation", model={"use_se": False}),
    _v("classifier_only", "components", "Classifier only (no decoder, no reconstruction term or score)",
       model={"use_reconstruction": False}, train={"recon_score_weight": 0.0}, eval={"alpha_rec": 0.0}),
    _v("dual_head", "components", "Dual-Head (Classifier + Decoder)", model={"use_reconstruction": True}),
    _v("no_recon_score", "components", "Trained with reconstruction, scored with the classifier only (alpha_rec = 0)",
       eval={"alpha_rec": 0.0}),
    # (c) classification loss
    _v("loss_focal", "loss", "Focal loss (reference, = full)", train={"classification": "focal"}),
    _v("loss_bce", "loss", "Plain binary cross-entropy", train={"classification": "bce"}),
    _v("loss_weighted_bce", "loss", "Positive-weighted binary cross-entropy", train={"classification": "weighted_bce"}),
    # (d) decoder inverse
    _v("inv_exact_solve", "decoder", "Exact inverse, linear solve (reference, = full)",
       model={"inverse": "exact", "inverse_method": "solve"}),
    _v("inv_exact_neumann", "decoder", "Exact inverse, truncated Neumann series",
       model={"inverse": "exact", "inverse_method": "neumann"}),
    _v("inv_mirror", "decoder", "Mirror (legacy) decoder layer", model={"inverse": "mirror"}),
    _v("inv_none", "decoder", "No inverse layer (plain linear decoder)", model={"inverse": "none"}),
    # (e) operational (Stage 3)
    _v("no_stage3", "operational", "Raw alarms: no known-event mask, no spike filter",
       eval={"use_known_event_mask": False, "min_consecutive": 1}),
    _v("raw_alarms", "operational", "Raw Alarms (Without Stage 3 Masking)",
       eval={"use_known_event_mask": False, "min_consecutive": 1}),
    _v("stage3_masking", "operational", "Stage 3 Operational Post-Processing (Full Model)",
       eval={"use_known_event_mask": True}),
]

VARIANTS: dict[str, Variant] = {v.name: v for v in _ALL}

STUDIES: dict[str, tuple[str, ...]] = {
    "norm": tuple(v.name for v in _ALL if v.group == "norm"),
    "components": ("full", "no_attention", "no_se", "classifier_only", "no_recon_score"),
    "loss": tuple(v.name for v in _ALL if v.group == "loss"),
    "decoder": tuple(v.name for v in _ALL if v.group == "decoder"),
    "operational": ("full", "no_stage3", "raw_alarms", "stage3_masking"),
    "table6": ("full", "standard_norm", "conv_feedforward", "classifier_only", "raw_alarms"),
}


def get_variant(name: str) -> Variant:
    """Return the registered variant or raise ``KeyError`` listing the valid names."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"unknown variant {name!r}; available: {sorted(VARIANTS)}") from None


def apply_variant(cfg: MissionConfig, variant: Variant | str) -> MissionConfig:
    """Return a validated deep copy of ``cfg`` with the variant's overrides applied."""
    var = get_variant(variant) if isinstance(variant, str) else variant
    out = copy.deepcopy(cfg)
    for section, overrides in (("model", var.model), ("train", var.train), ("eval", var.eval)):
        target = getattr(out, section)
        for key, value in overrides.items():
            if not hasattr(target, key):
                raise AttributeError(f"{section} config has no field {key!r} (variant {var.name})")
            setattr(target, key, value)
    out.validate()
    return out
