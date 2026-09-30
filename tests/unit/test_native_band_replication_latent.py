"""SYNTHETIC_CORRECTNESS_ONLY in-memory codec/objective and typed kind checks."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
from replication_test_support import artifact, codec, synthetic_inputs

from marine_echo.inference.native_band_acoustic import (
    NativeBandAcousticPredictor as OriginalForecast,
)
from marine_echo.inference.native_band_replication_acoustic import NativeBandAcousticPredictor
from marine_echo.inference.native_band_replication_encoder import NativeBandReplicationEncoder
from marine_echo.inference.native_band_replication_latent import NativeLatentPredictor


@pytest.mark.parametrize("seed", [7, 13, 23])
@pytest.mark.parametrize("method", ["shared_ssl", "permuted_ssl"])
def test_actual_forward_encoder_query_head_safe_relocation_rng_and_frozen_replay(seed, method):
    saved, model = artifact(seed=seed, method=method)
    saved["bindings"] = {"Z:/UNAVAILABLE/SYNTHETIC_CORRECTNESS_ONLY/ancestor.py": "a" * 64}
    x, mask, meta, query = synthetic_inputs(n=5)
    rng = torch.get_rng_state().clone()
    latent = NativeLatentPredictor(codec(saved))
    selected = {k: v for k, v in saved.items() if k != "model"}
    selected["kind"] = "native_band_replication_ssl_selected_encoder_v2"
    selected["encoder"] = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    encoder = NativeBandReplicationEncoder(codec(selected))
    forecast = NativeBandAcousticPredictor(codec(saved))
    assert torch.equal(rng, torch.get_rng_state())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    values = latent._scalers.channels(x, mask)
    with torch.no_grad():
        features = model.encoder.encode(
            torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(meta)
        )
        expected = model.predictor(features, torch.from_numpy(query)).numpy()
    actual = latent.predict_latents(x, mask, meta, query)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        encoder.encode(x, mask, meta), features.numpy(), atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        latent.encode(x, mask, meta), forecast.encode(x, mask, meta), atol=2e-6, rtol=2e-6
    )
    x[~mask] = np.nan
    np.testing.assert_array_equal(actual, latent.predict_latents(x, mask, meta, query))
    assert meta[0, 0, 4] == np.float32(230 / 250)
    assert query[0, 0, 4] == np.float32(230 / 250)
    assert actual.shape == (5, 3, 8) and actual.dtype == np.float32
    assert all(torch.equal(v, latent._model.state_dict()[k]) for k, v in state.items())
    assert all(not p.requires_grad and p.grad is None for p in latent._model.parameters())
    assert latent.objective_metadata["objective"] == (
        "shared_future_block" if method == "shared_ssl" else "permuted_pairing_control"
    )
    opaque = latent.source_bindings
    opaque.clear()
    assert latent.source_bindings == saved["bindings"]
    with pytest.raises(TypeError):
        latent.predict_latents(x, mask, meta, query, future=np.zeros((5, 3, 4, 4)))


@pytest.mark.parametrize("method", ["masked_ssl", "random_frozen", "direct"])
def test_untrained_forward_heads_never_claim_latent_objective(method):
    saved, _ = artifact(method=method)
    if method == "random_frozen":
        saved["config"]["pretrain_updates"] = 0
        saved["selected_pretrain_step"] = 0
    predictor = NativeLatentPredictor(codec(saved))
    x, mask, metadata, query = synthetic_inputs()
    with pytest.raises(ValueError, match="untrained"):
        predictor.predict_latents(x, mask, metadata, query)
    assert predictor.encode(x, mask, metadata).shape == (3, 8)


@pytest.mark.parametrize(
    "change",
    [
        "v1",
        "resume",
        "supervised",
        "config_missing",
        "config_extra",
        "config_nonfinite",
        "seed",
        "tensor_missing",
        "tensor_extra",
        "tensor_dtype",
        "tensor_nonfinite",
        "scaler",
        "unsafe",
        "downstream",
    ],
)
def test_exact_kind_configuration_and_safe_state_guards(change):
    saved, _ = artifact()
    saved = copy.deepcopy(saved)
    if change in ("v1", "resume", "supervised"):
        saved["kind"] = {
            "v1": "native_band_ssl_weights_only_inference_v1",
            "resume": "native_band_replication_ssl_resume_v2",
            "supervised": "native_band_replication_downstream_supervised_encoder_v2",
        }[change]
    elif change == "config_missing":
        del saved["config"]["cf_lr"]
    elif change == "config_extra":
        saved["config"]["variant"] = "another"
    elif change == "config_nonfinite":
        saved["config"]["cf_lr"] = float("nan")
    elif change == "seed":
        saved["config"]["seed"] = 17
    elif change == "tensor_missing":
        saved["model"].pop(next(iter(saved["model"])))
    elif change == "tensor_extra":
        saved["model"]["extra"] = torch.ones(1)
    elif change == "tensor_dtype":
        key = next(iter(saved["model"]))
        saved["model"][key] = saved["model"][key].double()
    elif change == "tensor_nonfinite":
        saved["model"][next(iter(saved["model"]))].flatten()[0] = float("inf")
    elif change == "scaler":
        saved["scalers"]["channel_std"][0] = 0
    elif change == "unsafe":
        saved["optimizer"] = {}
    else:
        saved["supervised_ancestry"] = {}
    with pytest.raises((ValueError, TypeError)):
        NativeLatentPredictor(codec(saved))


def test_v1_and_v2_encoders_keep_actual_kinds_and_v1_forecast_refuses_v2():
    selected, _ = artifact(seed=7, selected=True)
    original = copy.deepcopy(selected)
    original["kind"] = "native_band_ssl_selected_encoder_v1"
    a = NativeBandReplicationEncoder(codec(original))
    b = NativeBandReplicationEncoder(codec(selected))
    assert a.artifact_version == 1 and b.artifact_version == 2
    assert a.artifact_kind == original["kind"] and b.artifact_kind == selected["kind"]
    x, mask, metadata, _ = synthetic_inputs()
    np.testing.assert_array_equal(a.encode(x, mask, metadata), b.encode(x, mask, metadata))
    full, _ = artifact(seed=7)
    with pytest.raises(ValueError):
        OriginalForecast(codec(full))


def test_safe_load_float32_no_defaultdtype_or_rng_change(monkeypatch):
    saved, _ = artifact()
    calls = []
    actual_load = torch.load

    def checked(*args, **kwargs):
        calls.append(kwargs.copy())
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        rng = torch.get_rng_state().clone()
        predictor = NativeLatentPredictor(codec(saved))
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.get_default_dtype() == torch.float64
        assert all(p.dtype == torch.float32 for p in predictor._model.parameters())
        assert calls == [{"weights_only": True, "map_location": "cpu"}]
    finally:
        torch.set_default_dtype(old)
