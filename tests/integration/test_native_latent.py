"""SYNTHETIC_CORRECTNESS_ONLY codec, relocation and no-side-effect replay."""

import importlib.util
import io
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_latent_integration_support",
    Path(__file__).resolve().parents[2] / "evidence/ssl-latent-builder-v1/test_support.py",
)
support = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(support)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "method,band", [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]
)
def test_safe_codec_relocation_no_rng_cache_fit_or_provenance_reads(monkeypatch, method, band):
    artifact, _ = support.fixture(method, band)
    api = support.load_api()
    stream = support.codec(artifact)
    arrays = support.inputs()
    expected = api.NativeLatentPredictor(support.codec(artifact)).encode(*arrays[:3])
    real_load = torch.load
    calls = []

    def load_spy(weights, **kwargs):
        assert weights is stream
        assert kwargs == {"weights_only": True, "map_location": "cpu"}
        calls.append(kwargs)
        return real_load(weights, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("No RNG reset/cache/optimizer/fit/provenance execution is allowed.")

    monkeypatch.setattr(torch, "load", load_spy)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch, "set_default_dtype", forbidden)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", forbidden)
    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim, "SGD", forbidden)
    monkeypatch.setattr(api.Config, "validate", forbidden)
    monkeypatch.setattr(api.BandConfig, "validate", forbidden)
    monkeypatch.setattr(api.legacy.Scalers, "fit", forbidden)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    adapter = api.NativeLatentPredictor(stream)
    np.testing.assert_array_equal(adapter.encode(*arrays[:3]), expected)
    if method == "cf_jepa":
        assert np.isfinite(adapter.predict_cf_zones(*arrays[:3])).all()
    else:
        assert np.isfinite(adapter.predict_latents(*arrays)).all()
    assert torch.equal(torch.get_rng_state(), torch_rng)
    after = np.random.get_state()
    assert after[0] == numpy_rng[0] and after[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after[1], numpy_rng[1])
    assert random.getstate() == python_rng
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    assert adapter.source_bindings == artifact["bindings"]


@pytest.mark.parametrize(
    "method,band", [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]
)
def test_float32_replay_independent_of_float64_default_and_no_global_change(method, band):
    artifact, _ = support.fixture(method, band)
    api = support.load_api()
    arrays = support.inputs()
    original = api.NativeLatentPredictor(support.codec(artifact))
    expected = original.encode(*arrays[:3])
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        adapter = api.NativeLatentPredictor(support.codec(artifact))
        result = adapter.encode(*arrays[:3])
        assert torch.get_default_dtype() == torch.float64
        assert all(p.dtype == torch.float32 for p in adapter._model.parameters())
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32
        predict = adapter.predict_cf_zones if method == "cf_jepa" else adapter.predict_latents
        assert predict(*(arrays[:3] if method == "cf_jepa" else arrays)).dtype == np.float32
    finally:
        torch.set_default_dtype(previous)


class UnsafeFixture:
    def __reduce__(self):
        # Safe loader must refuse this built-in global; no filesystem payload.
        return eval, ("1+1",)


def test_weights_only_refuses_unsafe_pickle_global():
    api = support.load_api()
    stream = io.BytesIO()
    torch.save({"kind": UnsafeFixture()}, stream)
    stream.seek(0)
    with pytest.raises(
        pickle.UnpicklingError, match="Weights only|WeightsUnpickler|Unsupported global"
    ):
        api.NativeLatentPredictor(stream)


@pytest.mark.parametrize(
    "method,band", [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]
)
def test_complete_codec_replay_and_context_only_negative_strides(method, band):
    api = support.load_api()
    artifact, model = support.fixture(method, band)
    arrays = tuple(a[::-1] for a in support.inputs())
    # Both saves/loads use the real in-memory Torch ZIP codec.
    first = api.NativeLatentPredictor(support.codec(artifact))
    reload_artifact = dict(
        artifact, model={k: v.clone() for k, v in first._model.state_dict().items()}
    )
    second = api.NativeLatentPredictor(support.codec(reload_artifact))
    np.testing.assert_array_equal(first.encode(*arrays[:3]), second.encode(*arrays[:3]))
    first_call = first.predict_cf_zones if method == "cf_jepa" else first.predict_latents
    second_call = second.predict_cf_zones if method == "cf_jepa" else second.predict_latents
    args = arrays[:3] if method == "cf_jepa" else arrays
    np.testing.assert_array_equal(first_call(*args), second_call(*args))
    assert set(first._model.state_dict()) == set(model.state_dict())
