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

def test_vicreg_penalty_has_fixed_floor_covariance_and_two_online_gradients() -> None:
    model = AeonForwardSSL(width=4, layers=2)
    collapsed = torch.zeros(8, 4)
    assert torch.isclose(model.variance_floor_penalty(collapsed), torch.tensor(0.99))
    assert model.covariance_penalty(collapsed) == 0
    perfectly_correlated = torch.arange(8, dtype=torch.float32).reshape(-1, 1).repeat(1, 4)
    assert model.covariance_penalty(perfectly_correlated) > 0
    decorrelated = torch.eye(4)
    assert model.covariance_penalty(decorrelated) > 0  # Centering still induces covariance.

    torch.manual_seed(9)
    past = torch.randn(8, 24, 4, requires_grad=True)
    future = torch.randn(8, 6, 4, requires_grad=True)
    loss = model.pretrain_loss(
        past, torch.ones_like(past, dtype=torch.bool),
        future, torch.ones_like(future, dtype=torch.bool),
    )
    loss.backward()
    assert past.grad is not None and past.grad.abs().sum() > 0
    assert future.grad is not None and future.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    two_dimensional = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    assert torch.isclose(model.covariance_penalty(two_dimensional), torch.tensor(4.0))

def test_forward_loss_matches_reviewed_two_view_mean_formula() -> None:
    from torch.nn import functional as F

    torch.manual_seed(11)
    model = AeonForwardSSL(width=8, layers=2)
    past = torch.randn(8, 24, 4)
    past_mask = torch.ones_like(past, dtype=torch.bool)
    future = torch.randn(8, 6, 4)
    future_mask = torch.ones_like(future, dtype=torch.bool)
    target_values, target_mask = model.target_view(future, future_mask)
    context = model.encoder(past, past_mask)
    predicted = model.predictor(context)
    teacher = model.teacher(target_values, target_mask).detach()
    online_future = model.encoder(target_values, target_mask)
    expected = (
        25 * F.smooth_l1_loss(predicted, teacher)
        + 25 * (model.variance_floor_penalty(context)
                + model.variance_floor_penalty(online_future)) / 2
        + (model.covariance_penalty(context)
           + model.covariance_penalty(online_future)) / 2
    )
    assert torch.allclose(model.pretrain_loss(past, past_mask, future, future_mask), expected)
    changed_future = future.clone()
    changed_future[:, :, 0] += 2
    assert not torch.isclose(model.pretrain_loss(past, past_mask, changed_future, future_mask), expected)

    past.requires_grad_()
    future.requires_grad_()
    context_regularizer = (model.variance_floor_penalty(model.encoder(past, past_mask))
                           + model.covariance_penalty(model.encoder(past, past_mask)))
    context_regularizer.backward()
    assert past.grad is not None and past.grad.abs().sum() > 0
    future_view, future_view_mask = model.target_view(future, future_mask)
    online = model.encoder(future_view, future_view_mask)
    future_regularizer = model.variance_floor_penalty(online) + model.covariance_penalty(online)
    future_regularizer.backward()
    assert future.grad is not None and future.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
