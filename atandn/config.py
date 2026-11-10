"""Configuration dataclasses, mission defaults and YAML loading with validation.

Mission defaults specify the optimized hyperparameters corresponding to each spacecraft mission
as reported in the paper (Table 2). Hyperparameters can be further tuned or customized using
the Bayesian optimization pipeline (see ``training/search.py``) evaluated on the event-aware
validation block of the training split.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_NORMS = {"bign", "identity", "layernorm", "instancenorm", "groupnorm", "batchnorm", "rmsnorm"}
_INVERSES = {"exact", "mirror", "none"}
_CLASSIFICATIONS = {"focal", "bce", "weighted_bce"}


@dataclass
class ModelConfig:
    """Arguments of :class:`atandn.models.ATANDN`."""

    in_channels: int = 6
    window_size: int = 300
    conv_filters: tuple[int, int] = (64, 192)
    num_heads: int = 2
    dropout: float = 0.3
    norm: str = "bign"
    inverse: str = "exact"
    inverse_method: str = "solve"     # "solve" (default; exact up to the clamp) | "neumann" (approximation)
    neumann_steps: int = 4            # only used by inverse_method="neumann"
    use_attention: bool = True
    use_reconstruction: bool = True
    use_se: bool = True
    beta_min: float = 1e-6
    gamma_init_mode: str = "diag_dominant"  # BIGN Gamma init: "diag_dominant" | "uniform" (legacy)
    gamma_diag: float = 0.5                 # target Gamma_ii for "diag_dominant"
    gamma_off: float | None = None          # target Gamma_ij for "diag_dominant"; None -> 0.01 / (C - 1)
    pe_mode: str = "attention"              # "attention" (paper: PE at every MHA input, d_model=C_l) | "input" (legacy: PE on raw window)


@dataclass
class TrainConfig:
    """Optimisation and loss hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 15
    min_delta: float = 0.002
    warmup_epochs: int = 5
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    eta_min_factor: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    overlap: float = 0.8              # training windows: stride = int(window*(1-overlap))
    use_weighted_sampling: bool = False
    use_augmentation: bool = False
    recon_weight: float = 0.1
    pos_weight: float = 5.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    classification: str = "focal"
    recon_score_weight: float = 0.1   # alpha used for model selection on the validation block
    val_metric: str = "f1"            # "f1" (paper: validation-set F1) or "ap" (average precision)
    val_fraction: float = 0.2         # validation block carved from the TRAINING split (event-aware, see splits.py)
    val_min_events: int = 3           # wanted number of FULL anomalous events in the validation block (relaxed to 1)
    val_min_fit_fraction: float = 0.3  # a moved validation block must leave >= this fraction of the train split to fit
    seed: int = 42
    num_workers: int = 0
    clamp_warn_threshold: float = 0.5  # warn if mean decoder clamp rate stays above this ...
    clamp_warn_epochs: int = 3         # ... after this many epochs (<= 0 disables)
    amp: Any = "auto"                  # mixed precision: "auto" (on only on CUDA) | True | False; BIGN layers stay float32

    def train_stride(self, window_size: int) -> int:
        """Training window stride implied by ``overlap`` (as in the legacy ``TelemetryDataset``)."""
        return max(1, int(window_size * (1.0 - self.overlap)))


@dataclass
class DataConfig:
    """Where the data lives and how it is split."""

    kind: str = "esa"                 # "esa" | "opssat"
    path: str = ""                    # parquet (ESA) or segments.csv (OPS-SAT), relative to the data root
    labels_csv: str = "labels.csv"
    anomaly_types_csv: str = "anomaly_types.csv"
    channel_start: int = 41
    channel_end: int = 46
    target_sampling_rate: float = 0.033
    train_ratio: float = 0.5
    test_stride: int = 10
    opssat_mode: str = "interleaved"  # OPS-SAT: "interleaved" (legacy, C=1) | "multichannel" (nine channels pivoted, C=9)
    telecommand_dir: str = ""         # ESA raw mission folder holding telecommands.csv (relative to the data root)
    telecommand_priorities: list | None = None  # keep only telecommands.csv rows with these priorities (None = all)


@dataclass
class EvalConfig:
    """Inference and post-processing settings (all fitted on the training split)."""

    alpha_rec: float | None = None    # None -> TrainConfig.recon_score_weight
    normalizer: str = "percentile"    # "percentile" | "minmax"
    batch_size: int = 512
    sensitive_threshold: float = 0.90  # fixed threshold used by the training-split FP analysis
    use_known_event_mask: bool = False
    mask_buffer_before_minutes: float = 0.0   # settling-time rationale: nothing before a known event by default
    mask_buffer_after_minutes: float | None = None  # None -> derived from the training split (analysis/)
    min_consecutive: int | str = "derived"  # spike filter: "derived" (training-analysis value) or an integer >= 1
    threshold_method: str = "grid"     # "grid" (paper: [0.05, 0.95], step 0.005) | "exact" (exact F1 optimum)
    allow_train_threshold_fallback: bool = False  # validation block without positives: use fit-part scores
    mask_source: str = "telecommands"  # "telecommands" (paper text: settling buffers after telecommands) | "rare_events"
    threshold_floor: str = "none"      # "none" | "nominal_p99" (tau = max(tau*, P99 of fit-part scores of normal windows))
    point_adjustment: bool = True      # point-adjusted event scoring matching ESA-ADB Supplementary Table 9 (paper Table 3)


@dataclass
class MissionConfig:
    """Full configuration of one mission."""

    name: str = "m1"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (tuples become lists) suitable for YAML/JSON/checkpoints."""
        d = dataclasses.asdict(self)
        d["model"]["conv_filters"] = list(d["model"]["conv_filters"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MissionConfig":
        """Build and validate a config from a (possibly partial) dict; unknown keys raise."""
        d = copy.deepcopy(d)
        sections = {"model": ModelConfig, "train": TrainConfig, "data": DataConfig, "eval": EvalConfig}
        unknown = set(d) - set(sections) - {"name", "mission", "search", "paths"}  # last two are read by scripts/
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        base = mission_defaults(d["mission"]) if d.get("mission") else cls()
        if "name" in d:
            base.name = str(d["name"])
        for key, klass in sections.items():
            over = d.get(key) or {}
            valid = {f.name for f in fields(klass)}
            bad = set(over) - valid
            if bad:
                raise ValueError(f"Unknown keys in '{key}': {sorted(bad)}")
            current = getattr(base, key)
            for k, v in over.items():
                if k == "conv_filters":
                    v = tuple(int(x) for x in v)
                setattr(current, k, v)
        base.validate()
        return base

    def validate(self) -> None:
        """Raise ``ValueError`` for inconsistent values."""
        m, t, e, d = self.model, self.train, self.eval, self.data
        if m.norm not in _NORMS:
            raise ValueError(f"model.norm must be one of {sorted(_NORMS)}")
        if m.inverse not in _INVERSES:
            raise ValueError(f"model.inverse must be one of {sorted(_INVERSES)}")
        if m.inverse_method not in {"solve", "neumann"}:
            raise ValueError("model.inverse_method must be 'solve' or 'neumann'")
        if m.pe_mode not in {"attention", "input"}:
            raise ValueError("model.pe_mode must be 'attention' or 'input'")
        if t.amp not in ("auto", True, False):
            raise ValueError("train.amp must be 'auto', true or false")
        if m.gamma_init_mode not in {"diag_dominant", "uniform"}:
            raise ValueError("model.gamma_init_mode must be 'diag_dominant' or 'uniform'")
        if m.gamma_diag <= 0 or (m.gamma_off is not None and m.gamma_off <= 0):
            raise ValueError("model.gamma_diag and model.gamma_off must be positive")
        if len(m.conv_filters) != 2 or any(c <= 0 for c in m.conv_filters):
            raise ValueError("model.conv_filters must be two positive integers")
        if m.in_channels <= 0 or m.window_size < 8:
            raise ValueError("model.in_channels must be > 0 and model.window_size >= 8")
        if m.use_attention and any(c % m.num_heads for c in m.conv_filters):
            raise ValueError("model.conv_filters must be divisible by model.num_heads")
        if m.use_se and any(c % 8 != 0 for c in m.conv_filters):
            raise ValueError("model.conv_filters must be divisible by 8 for Squeeze-and-Excitation (reduction ratio r=8)")
        if not 0.0 <= m.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if t.classification not in _CLASSIFICATIONS:
            raise ValueError(f"train.classification must be one of {sorted(_CLASSIFICATIONS)}")
        if t.val_metric not in {"f1", "ap"}:
            raise ValueError("train.val_metric must be 'f1' or 'ap'")
        if t.lr <= 0 or t.batch_size <= 0 or t.max_epochs <= 0 or t.patience <= 0:
            raise ValueError("train.lr, batch_size, max_epochs and patience must be positive")
        if not 0.0 <= t.overlap < 1.0:
            raise ValueError("train.overlap must be in [0, 1)")
        if not 0.0 < t.val_fraction < 0.5:
            raise ValueError("train.val_fraction must be in (0, 0.5)")
        if not 0.0 <= t.val_min_fit_fraction < 1.0 - t.val_fraction:
            raise ValueError("train.val_min_fit_fraction must be in [0, 1 - train.val_fraction)")
        if t.val_min_events < 1:
            raise ValueError("train.val_min_events must be >= 1")
        if not 0.0 <= t.ema_decay < 1.0:
            raise ValueError("train.ema_decay must be in [0, 1)")
        if d.kind not in {"esa", "opssat"}:
            raise ValueError("data.kind must be 'esa' or 'opssat'")
        if not 0.0 < d.train_ratio < 1.0 or d.test_stride <= 0:
            raise ValueError("data.train_ratio must be in (0, 1) and data.test_stride > 0")
        if e.alpha_rec is not None and not 0.0 <= e.alpha_rec <= 1.0:
            raise ValueError("eval.alpha_rec must be in [0, 1]")
        if e.normalizer not in {"percentile", "minmax"}:
            raise ValueError("eval.normalizer must be 'percentile' or 'minmax'")
        mc = e.min_consecutive
        if not (mc == "derived" or (isinstance(mc, int) and not isinstance(mc, bool) and mc >= 1)):
            raise ValueError("eval.min_consecutive must be 'derived' or an integer >= 1")
        if e.mask_buffer_before_minutes < 0 or (e.mask_buffer_after_minutes is not None and e.mask_buffer_after_minutes < 0):
            raise ValueError("eval.mask_buffer_before_minutes and eval.mask_buffer_after_minutes must be >= 0")
        if e.threshold_method not in {"grid", "exact"}:
            raise ValueError("eval.threshold_method must be 'grid' or 'exact'")
        if e.mask_source not in {"telecommands", "rare_events"}:
            raise ValueError("eval.mask_source must be 'telecommands' or 'rare_events'")
        if e.threshold_floor not in {"none", "nominal_p99"}:
            raise ValueError("eval.threshold_floor must be 'none' or 'nominal_p99'")
        if d.opssat_mode not in {"interleaved", "multichannel"}:
            raise ValueError("data.opssat_mode must be 'interleaved' or 'multichannel'")

    @property
    def alpha_rec(self) -> float:
        """Reconstruction weight of the dual-evidence score."""
        return self.train.recon_score_weight if self.eval.alpha_rec is None else self.eval.alpha_rec


def mission_defaults(name: str) -> MissionConfig:
    """Return the default :class:`MissionConfig` for ``"m1"``, ``"m2"`` or ``"ops_sat"`` matching paper Tables 1, 2, and 4."""
    if name == "m1":   # ESA Mission 1; channels 41-46, 0.033 Hz, 50/50
        return MissionConfig(
            name="m1",
            model=ModelConfig(in_channels=6, window_size=300, conv_filters=(64, 192), num_heads=2, dropout=0.32),
            train=TrainConfig(lr=0.00060, weight_decay=2.0e-05, batch_size=32,
                              overlap=0.79, use_weighted_sampling=False, use_augmentation=True,
                              recon_weight=0.35, pos_weight=14.1,
                              focal_gamma=2.61, recon_score_weight=0.33, val_metric="f1"),
            data=DataConfig(kind="esa", path="all_telemetry_41_46.parquet", channel_start=41, channel_end=46,
                            target_sampling_rate=0.033, train_ratio=0.5),
            eval=EvalConfig(use_known_event_mask=True, mask_source="telecommands",
                            mask_buffer_before_minutes=0.0, mask_buffer_after_minutes=15.0,
                            min_consecutive=1, point_adjustment=True),
        )
    if name == "m2":   # ESA Mission 2; channels 18-28, 0.056 Hz, 50/50
        return MissionConfig(
            name="m2",
            model=ModelConfig(in_channels=11, window_size=300, conv_filters=(64, 128), num_heads=4, dropout=0.18),
            train=TrainConfig(lr=0.00017, weight_decay=2.0e-04, batch_size=64,
                              overlap=0.85, use_weighted_sampling=True, use_augmentation=False,
                              recon_weight=0.13, pos_weight=9.5,
                              focal_gamma=2.22, recon_score_weight=0.11, val_metric="f1"),
            data=DataConfig(kind="esa", path="all_telemetry_18_28.parquet", channel_start=18, channel_end=28,
                            target_sampling_rate=0.056, train_ratio=0.5),
            eval=EvalConfig(use_known_event_mask=True, mask_source="telecommands",
                            mask_buffer_before_minutes=0.0, mask_buffer_after_minutes=105.0,
                            min_consecutive=3, point_adjustment=True),
        )
    if name == "ops_sat":  # OPS-SAT; 9 ADCS channels, window 400, 75/25
        return MissionConfig(
            name="ops_sat",
            model=ModelConfig(in_channels=9, window_size=400, conv_filters=(64, 256), num_heads=2, dropout=0.33),
            train=TrainConfig(lr=0.00099, weight_decay=3.0e-05, batch_size=64,
                              overlap=0.75, use_weighted_sampling=False, use_augmentation=True,
                              recon_weight=0.19, pos_weight=4.7,
                              focal_gamma=1.29, recon_score_weight=0.02, val_metric="f1"),
            data=DataConfig(kind="opssat", path="segments.csv", train_ratio=0.75, opssat_mode="multichannel"),
            eval=EvalConfig(use_known_event_mask=False, threshold_floor="nominal_p99", point_adjustment=True),
        )
    raise ValueError(f"Unknown mission '{name}' (expected 'm1', 'm2' or 'ops_sat')")


def load_config(path: str | Path) -> MissionConfig:
    """Load a mission config from YAML.

    The file may contain ``mission: m1|m2|ops_sat`` (start from those defaults) and override sections
    ``model``, ``train``, ``data``, ``eval``. Unknown keys and invalid values raise ``ValueError``.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return MissionConfig.from_dict(raw)
