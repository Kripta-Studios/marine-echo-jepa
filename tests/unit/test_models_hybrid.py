"""Frozen-latent downstream heads use only permitted past context."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marine_echo.models.compact import ModelConfig, TemporalJEPA
from marine_echo.models.hybrid import (
    FrozenLatentRidge,
    RawLatentTreeQuantiles,
    frozen_context_features,
)


def test_frozen_latents_and_hybrid_heads_are_train_only() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    encoder = TemporalJEPA(ModelConfig(width=16, layers=1, heads=4), mode="ema")
    values = np.random.default_rng(7).normal(-60, 1, size=(20, 96, 4, 64))
    mask = np.ones_like(values, dtype=bool)
    targets = np.random.default_rng(13).normal(-70, 2, size=(20, 3))
    latents = frozen_context_features(encoder, values[:2], mask[:2])
    assert latents.shape == (2, 4 * 16)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    for head in (FrozenLatentRidge(encoder), RawLatentTreeQuantiles(encoder, max_iter=4)):
        with pytest.raises(ValueError):
            head.fit(values, mask, targets, partition="validation")
        head.fit(values, mask, targets, partition="train")
        prediction = head.predict(values[:2], mask[:2])
        assert prediction.quantiles.shape == (2, 3, 5)
        assert np.isfinite(prediction.quantiles).all()
        assert np.all(np.diff(prediction.quantiles, axis=-1) >= 0)
