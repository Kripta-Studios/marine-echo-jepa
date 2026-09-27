"""Forward EMA acoustic predictor contracts."""

from __future__ import annotations

import torch

from marine_echo.models.aeon_forward_ssl import AeonForwardSSL


def test_forward_pretrain_has_trainable_variance_and_detached_teacher() -> None:
    torch.manual_seed(7)
    model = AeonForwardSSL(width=16, layers=2, variance_floor=0.1, variance_weight=0.04)
    past = torch.randn(8, 24, 4, requires_grad=True)
    past_mask = torch.ones_like(past, dtype=torch.bool)
    future = torch.randn(8, 6, 4, requires_grad=True)
    future_mask = torch.ones_like(future, dtype=torch.bool)
    loss = model.pretrain_loss(past, past_mask, future, future_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert past.grad is not None and past.grad.abs().sum() > 0
    assert future.grad is not None and future.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    forecast = model(past.detach(), past_mask)
    assert forecast.shape == (8, 3, 5)
    assert torch.all(forecast.diff(dim=-1) >= 0)


def test_forward_target_view_only_places_six_future_products() -> None:
    model = AeonForwardSSL(width=16, layers=2)
    future = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    mask = torch.ones_like(future, dtype=torch.bool)
    target, target_mask = model.target_view(future, mask)
    assert target.shape == (2, 24, 4)
    assert not target_mask[:, :18].any()
    assert torch.equal(target[:, 18:], future)
    assert torch.equal(target_mask[:, 18:], mask)

def test_variance_floor_uses_unbiased_batch_std_and_fixed_epsilon() -> None:
    model = AeonForwardSSL(width=4, layers=2, variance_floor=0.1, variance_weight=0.04)
    collapsed = torch.zeros(8, 4, requires_grad=True)
    penalty = model.variance_floor_penalty(collapsed)
    assert torch.isclose(penalty, torch.tensor(0.09))
    penalty.backward()
    assert collapsed.grad is not None
