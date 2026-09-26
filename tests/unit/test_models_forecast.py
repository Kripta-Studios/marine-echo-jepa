"""Forecast shape, ordering and abstention obligations."""

from __future__ import annotations

import torch

from marine_echo.models.compact import DirectForecaster, ModelConfig


def test_direct_forecast_shapes_quantiles_and_missingness() -> None:
    torch.manual_seed(7)
    model = DirectForecaster(ModelConfig(width=32, layers=1, heads=4))
    context = torch.randn(2, 96, 4, 64)
    valid = torch.ones_like(context, dtype=torch.bool)
    valid[1] = False
    result = model(context, valid)
    assert result.quantiles.shape == (2, 3, 5)
    assert result.profile.shape == (2, 3, 4, 64)
    assert result.eligible.tolist() == [True, False]
    assert torch.isfinite(result.quantiles).all()
    assert torch.isfinite(result.profile).all()
    assert torch.all(result.quantiles[..., 1:] >= result.quantiles[..., :-1])


def test_parameter_budget_default() -> None:
    model = DirectForecaster(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) < 1_500_000
