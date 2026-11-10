"""Tests for the loss functions (regression against the legacy combined_loss)."""
import pytest
import torch
import torch.nn.functional as F

from atandn.losses import FocalLoss, rr_wfl_loss


def _legacy_combined_loss(recon, target_x, logits, target_y, recon_w=0.1, pos_weight=5.0,
                          focal_gamma=2.0, label_smoothing=0.0):
    recon_loss = F.mse_loss(recon, target_x)
    y = target_y * (1 - label_smoothing) + 0.5 * label_smoothing if label_smoothing > 0 else target_y
    bce = F.binary_cross_entropy_with_logits(
        logits, y.unsqueeze(1), pos_weight=torch.tensor([pos_weight]), reduction="none")
    pt = torch.exp(-bce)
    clf = ((1 - pt) ** focal_gamma * bce).mean()
    return recon_w * recon_loss + (1 - recon_w) * clf


@pytest.mark.parametrize("smooth", [0.0, 0.05])
def test_focal_path_matches_legacy(smooth):
    torch.manual_seed(0)
    recon, x = torch.randn(8, 3, 20), torch.randn(8, 3, 20)
    logits = torch.randn(8)
    y = (torch.rand(8) > 0.6).float()
    new, comps = rr_wfl_loss(recon, x, logits, y, recon_weight=0.2, pos_weight=7.0,
                             focal_gamma=1.5, label_smoothing=smooth)
    old = _legacy_combined_loss(recon, x, logits.unsqueeze(1), y, 0.2, 7.0, 1.5, smooth)
    assert torch.allclose(new, old, atol=1e-7)
    assert set(comps) == {"clf", "mse"}


@pytest.mark.parametrize("mode", ["focal", "bce", "weighted_bce"])
def test_modes_finite_with_gradients(mode):
    logits = torch.randn(8, requires_grad=True)
    y = (torch.rand(8) > 0.5).float()
    loss, _ = rr_wfl_loss(torch.randn(8, 2, 5), torch.randn(8, 2, 5), logits, y,
                          classification=mode, label_smoothing=0.05)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_recon_none_has_no_mse_term():
    logits, y = torch.randn(6), (torch.rand(6) > 0.5).float()
    loss, comps = rr_wfl_loss(None, torch.zeros(1), logits, y)
    assert "mse" not in comps and torch.allclose(loss, comps["clf"])


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        rr_wfl_loss(None, torch.zeros(1), torch.zeros(2), torch.zeros(2), classification="x")


def test_focal_loss_module():
    f = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=3.0)
    logits, y = torch.randn(5, 1), torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]])
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([3.0]), reduction="none")
    expected = (0.25 * (1 - torch.exp(-bce)) ** 2 * bce).mean()
    assert torch.allclose(f(logits, y), expected)
