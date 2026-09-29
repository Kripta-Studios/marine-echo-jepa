"""Context-only reusable inference preserves the native forecast contract."""

from __future__ import annotations

import io
from dataclasses import asdict

import numpy as np
import pytest
import torch

from marine_echo.inference.native_acoustic import NativeAcousticPredictor, validate_context
from marine_echo.training.native_ssl import Config, Scalers, cpu_state, initialize_model


def context():
    meta = np.zeros((2, 4, 10), np.float32)
    meta[:, :, 0] = np.asarray([38000, 125000, 200000, 455000]) / 455000
    meta[:, :, 1:3] = 1
    meta[:, :, 4] = 230 / 250
    query = np.tile(meta[:, :1], (1, 3, 1))
    query[:, :, -1] = [1, 3, 6]
    return {
        "x": np.ones((2, 96, 4), np.float32) * -70,
        "observed": np.ones((2, 96, 4), bool),
        "metadata": meta,
        "query": query,
    }


def predictor():
    config = Config(width=8, latent=4, blocks=1, heads=2)
    model = initialize_model(7, width=8, latent=4, blocks=1, heads=2)
    scaler = Scalers(
        np.full(4, -70, np.float32),
        np.ones(4, np.float32),
        np.full(3, -70, np.float32),
        np.ones(3, np.float32),
    )
    buffer = io.BytesIO()
    torch.save(
        {
            "kind": "native_ssl_weights_only_inference_v1",
            "config": asdict(config),
            "model": cpu_state(model),
            "scalers": scaler.to_dict(),
        },
        buffer,
    )
    buffer.seek(0)
    return NativeAcousticPredictor(buffer)


def test_missing_primary_prefix_is_outside_declared_issuance_contract():
    data = context()
    data["observed"][0, 0, 0] = False
    with pytest.raises(ValueError, match="primary"):
        validate_context(data, 96)


def test_native230_cannot_be_relabelled_native200():
    data = context()
    data["query"][:, :, 4] = 200 / 250
    with pytest.raises(ValueError, match="native geometry"):
        validate_context(data, 96)


def test_forecasts_and_representations_ignore_assessment_and_missing_fill_values():
    class ContextOnly(dict):
        def __getitem__(self, key):
            if key in ("y", "future", "future_observed", "y_observed"):
                raise AssertionError("Assessment target was accessed.")
            return super().__getitem__(key)

    data = ContextOnly(context())
    data["observed"][0, :, 3] = False
    model = predictor()
    before = model.encode_and_forecast(data)
    data["x"][0, :, 3] = 1e20
    after = model.encode_and_forecast(data)
    np.testing.assert_array_equal(before["embeddings"], after["embeddings"])
    np.testing.assert_array_equal(before["quantile_forecasts_db"], after["quantile_forecasts_db"])
    assert before["embeddings"].shape == (2, 4)
    assert before["quantile_forecasts_db"].shape == (2, 3, 5)
    assert (np.diff(before["quantile_forecasts_db"], axis=-1) >= 0).all()
