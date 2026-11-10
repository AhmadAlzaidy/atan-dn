"""Positional-encoding modes (A28), mixed precision (A59) and the bottleneck length (A24)."""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from atandn.config import MissionConfig, TrainConfig
from atandn.layers.attention import MultiHeadAttention1D, PositionalEncoding1D
from atandn.layers.bign import BIGN1D, ExactInverseBIGN1D
from atandn.models.atandn import ATANDN
from atandn.training import load_checkpoint, save_checkpoint
from atandn.training.trainer import Trainer, resolve_amp


def _model(pe_mode: str, **kw) -> ATANDN:
    torch.manual_seed(0)
    return ATANDN(in_channels=3, window_size=40, conv_filters=(8, 16), num_heads=2, pe_mode=pe_mode, **kw)


@pytest.mark.parametrize("mode", ["attention", "input"])
def test_pe_modes_shapes_and_params(mode):
    m = _model(mode)
    recon, logits = m.eval()(torch.randn(2, 3, 40))
    assert recon.shape == (2, 3, 40) and logits.shape == (2,)
    ref = _model("input")
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in ref.parameters())  # PE has no parameters
    pes = [x for x in m.modules() if isinstance(x, PositionalEncoding1D)]
    assert all(len(list(x.parameters())) == 0 for x in pes)
    if mode == "attention":
        assert len(pes) == 3   # two encoder blocks + bottleneck
        assert sorted(x.pe.shape[1] for x in pes) == [8, 16, 16]   # d_model = block width
    else:
        assert len(pes) == 1 and pes[0].pe.shape[1] == 3          # d_model = C_in


def test_pe_attention_changes_output_vs_input():
    a, b = _model("attention"), _model("input")
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k in b.state_dict()}, strict=False)
    x = torch.randn(2, 3, 40)
    assert not torch.allclose(a.eval()(x)[1], b.eval()(x)[1])


def test_pe_mode_invalid():
    with pytest.raises(ValueError):
        _model("bogus")


@pytest.mark.parametrize("mode", ["attention", "input"])
def test_checkpoint_round_trip(tmp_path, mode):
    cfg = MissionConfig()
    cfg.model.in_channels, cfg.model.window_size, cfg.model.conv_filters = 3, 40, (8, 16)
    cfg.model.num_heads, cfg.model.pe_mode = 2, mode
    m = ATANDN(**{**cfg.model.__dict__, "conv_filters": (8, 16)}).eval()
    p = save_checkpoint(tmp_path / "m.pt", m, cfg, {})
    m2, cfg2, _ = load_checkpoint(p, "cpu")
    assert cfg2.model.pe_mode == mode and m2.pe_mode == mode
    x = torch.randn(2, 3, 40)
    assert torch.equal(m(x)[1], m2(x)[1])


def test_legacy_checkpoint_without_pe_mode_loads_as_input(tmp_path):
    cfg = MissionConfig()
    cfg.model.in_channels, cfg.model.window_size, cfg.model.conv_filters, cfg.model.num_heads = 3, 40, (8, 16), 2
    cfg.model.pe_mode = "input"
    m = ATANDN(**{**cfg.model.__dict__, "conv_filters": (8, 16)}).eval()
    p = save_checkpoint(tmp_path / "m.pt", m, cfg, {})
    blob = torch.load(p, weights_only=True)
    del blob["cfg"]["model"]["pe_mode"]
    torch.save(blob, p)
    m2, cfg2, _ = load_checkpoint(p, "cpu")
    assert cfg2.model.pe_mode == "input"


@pytest.mark.parametrize("t", [200, 250, 300, 350, 400])
def test_bottleneck_length_is_ceil_t_over_4_and_decoder_interpolated(t):
    m = ATANDN(in_channels=2, window_size=t, conv_filters=(8, 16), num_heads=2).eval()
    assert m.bottleneck_len == math.ceil(t / 4)
    recon, _ = m(torch.randn(1, 2, t))
    assert recon.shape[-1] == t


# ---- mixed precision --------------------------------------------------------------------------------------------

def test_resolve_amp():
    cpu = torch.device("cpu")
    assert resolve_amp("auto", cpu) == (False, None)
    assert resolve_amp(False, cpu) == (False, None)
    assert resolve_amp(True, cpu) == (True, torch.bfloat16)
    assert resolve_amp("auto", torch.device("cuda")) == (True, torch.float16)


def test_config_amp_validation():
    cfg = MissionConfig()
    cfg.train.amp = "maybe"
    with pytest.raises(ValueError):
        cfg.validate()
    assert TrainConfig().amp == "auto"


def test_bign_runs_float32_under_autocast():
    b = BIGN1D(4)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = b(x)
        inv = ExactInverseBIGN1D(tied_to=b)(y)
    assert y.dtype == torch.float32 and inv.dtype == torch.float32
    assert torch.isfinite(y).all() and torch.isfinite(inv).all()


def test_bign_float32_unchanged_by_autocast_guard():
    torch.manual_seed(0)
    b = BIGN1D(4)
    x = torch.randn(2, 4, 16)
    ref = x * torch.rsqrt(b.beta.view(1, -1, 1) + torch.matmul(b.gamma, x * x))
    assert torch.equal(b(x), ref)


def _loader():
    torch.manual_seed(1)
    x, y = torch.randn(16, 3, 40), (torch.rand(16) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _train(amp) -> tuple[ATANDN, dict]:
    m = _model("attention")
    cfg = TrainConfig(max_epochs=2, warmup_epochs=1, patience=5, amp=amp)
    res = Trainer(m, cfg, device="cpu").fit(_loader(), _loader())
    return m, res


def test_amp_false_bit_identical_to_default_auto_on_cpu():
    torch.manual_seed(5)
    m1, r1 = _train(False)
    torch.manual_seed(5)
    m2, r2 = _train("auto")
    assert r1["amp"] is False and r2["amp"] is False
    for (k, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    assert [h["train_loss"] for h in r1["history"]] == [h["train_loss"] for h in r2["history"]]


def test_amp_true_cpu_finite_loss():
    _, res = _train(True)
    assert res["amp"] is True and res["amp_dtype"] == "bfloat16"
    assert all(torch.isfinite(torch.tensor(h["train_loss"])) for h in res["history"])
