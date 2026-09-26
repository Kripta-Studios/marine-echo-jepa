"""Teacher versus shared target gradient semantics."""

from __future__ import annotations

import torch

from marine_echo.models.compact import ModelConfig, TemporalJEPA


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    context = torch.randn(3, 96, 4, 64)
    future = torch.randn(3, 3, 4, 4, 64)
    return (
        context,
        torch.ones_like(context, dtype=torch.bool),
        future,
        torch.ones_like(future, dtype=torch.bool),
    )


def test_ema_teacher_is_frozen_and_shared_target_receives_gradient() -> None:
    args = _inputs()
    ema = TemporalJEPA(
        ModelConfig(width=32, layers=1, heads=4), mode="ema", sigreg_weight=0.0
    )
    ema_out = ema.objective(*args)
    ema_out.loss.backward()
    assert ema_out.target.requires_grad is False
    assert all(parameter.grad is None for parameter in ema.target_encoder.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in ema.encoder.parameters()
    )

    shared = TemporalJEPA(
        ModelConfig(width=32, layers=1, heads=4),
        mode="shared_sigreg",
        sigreg_weight=0.04,
    )
    shared_out = shared.objective(*args)
    shared_out.target.retain_grad()
    shared_out.loss.backward()
    assert shared_out.target.requires_grad is True
    assert shared_out.target.grad is not None
    assert shared_out.target.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in shared.encoder.parameters()
    )


def test_jepa_rejects_future_as_context_shape() -> None:
    context, mask, future, future_mask = _inputs()
    model = TemporalJEPA(ModelConfig(width=32, layers=1, heads=4), mode="ema")
    with torch.no_grad():
        output = model.objective(context, mask, future, future_mask)
    assert output.predicted.shape == (3, 3, 8, 32)
    assert output.target.shape == output.predicted.shape
