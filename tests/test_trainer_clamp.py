"""Trainer reports the decoder-inverse clamp rate in its history."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from atandn.config import TrainConfig
from atandn.models.atandn import ATANDN
from atandn.training.trainer import Trainer


def test_history_contains_clamp_rate():
    torch.manual_seed(0)
    model = ATANDN(in_channels=4, window_size=32, conv_filters=(8, 8), num_heads=2)
    x, y = torch.randn(16, 4, 32), (torch.rand(16) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    cfg = TrainConfig(max_epochs=2, warmup_epochs=1, patience=5)
    res = Trainer(model, cfg, device="cpu").fit(loader, loader)
    h = res["history"]
    assert len(h) == 2
    for e in h:
        assert 0.0 <= e["clamp_rate"] <= 1.0
        assert len(e["clamp_rates"]) == 2  # two ExactInverseBIGN1D in the decoder
