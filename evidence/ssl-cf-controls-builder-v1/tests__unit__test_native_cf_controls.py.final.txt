"""SYNTHETIC_CORRECTNESS_ONLY: actual source factories and safe CPU codecs."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

# Authoritative immutable helpers are imported from MAIN, only the new modules
# are added to the package search path in this isolated implementation checkout.
BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER if BUILDER.name == "marine-echo-jepa" else BUILDER.parent / "marine-echo-jepa"
import sys

sys.path.insert(0, str(MAIN / "src"))
import marine_echo

marine_echo.__path__ = [str(MAIN / "src/marine_echo")]
import marine_echo.inference
import marine_echo.training

marine_echo.training.__path__ = [str(MAIN / "src/marine_echo/training")]
marine_echo.inference.__path__ = [str(MAIN / "src/marine_echo/inference")]
from marine_echo.inference import native_band_acoustic, native_encoder
from marine_echo.training import native_downstream, native_ssl

assert Path(native_ssl.__file__).is_relative_to(MAIN)
assert Path(native_downstream.__file__).is_relative_to(MAIN)
assert Path(native_encoder.__file__).is_relative_to(MAIN)
assert Path(native_band_acoustic.__file__).is_relative_to(MAIN)
for package, folder in ((marine_echo.training, "training"), (marine_echo.inference, "inference")):
    if BUILDER != MAIN:
        package.__path__ = [str(BUILDER / "src/marine_echo" / folder), *package.__path__]

from marine_echo.inference import native_cf_controls as api
from marine_echo.inference.native_encoder import NativeAcousticEncoder
from marine_echo.training import native_cf_controls as controls

core = controls.core
assert Path(core.__file__).is_relative_to(MAIN)
assert Path(controls.downstream.__file__).is_relative_to(MAIN)
torch.set_num_threads(2)


def small_config(method="cf_random_frozen", seed=7):
    return controls.Config(
        method=method, seed=seed, width=8, latent=8, blocks=1, batch_size=4, updates=4, cadence=1
    )


def context(n=4):
    rng = np.random.default_rng(1729)
    x = rng.normal(-70, 4, (n, 96, 4)).astype(np.float32)
    mask = np.ones(x.shape, dtype=bool)
    mask[:, ::3, 1:] = False
    metadata = np.zeros((n, 4, 10), dtype=np.float32)
    metadata[:, :, 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    metadata[:, :, 4] = np.array([230, 250, 250, 250]) / 250
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[:, :, 9] = [1, 3, 6]
    return x, mask, metadata, query


def scalers():
    return core.Scalers(
        np.array([-70, -71, -72, -73], np.float32),
        np.array([4, 5, 6, 7], np.float32),
        np.array([-75, -76, -77], np.float32),
        np.array([2, 3, 4], np.float32),
    )


def batch():
    x, mask, metadata, query = context()
    return {
        "x": torch.from_numpy(scalers().channels(x, mask)),
        "observed": torch.from_numpy(mask),
        "metadata": torch.from_numpy(metadata),
        "query": torch.from_numpy(query),
        "y": torch.zeros(4, 3),
        "y_observed": torch.ones(4, 3, dtype=torch.bool),
    }


def artifact(method="cf_random_frozen", *, encoder=False):
    cfg = small_config(method)
    model = controls.prepare_model(cfg)
    return {
        "kind": controls.ENCODER_KINDS[method] if encoder else controls.INFERENCE_KIND,
        "encoder" if encoder else "model": core.cpu_state(model.encoder if encoder else model),
        "config": cfg.to_dict(),
        "core_config": controls.core_config(cfg).to_dict(),
        "scalers": scalers().to_dict(),
        "bindings": {"Z:/inaccessible-ancestor/source.py": "a" * 64},
        "evidence_kind": controls.SYNTHETIC,
        "supervised_ancestry": {
            "mode": cfg.mode,
            "ssl_only": False,
            "supervised_updates": 4,
            "selected_supervised_step": 4,
            "ancestor_encoder_sha256": None,
            "ancestor_run_sha256": None,
            "ssl_updates": 0,
            "encoder_supervised_updates": 0 if encoder_method_frozen(method) else 4,
            "initialization_seed": 7,
            "readout_initialization_seed": 100007,
        },
    }


def encoder_method_frozen(method):
    return method == "cf_random_frozen"


def codec(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return stream


def test_frozen_encoder_and_batchnorm_buffers_never_change():
    cfg = small_config()
    model = controls.prepare_model(cfg)
    before = core.cpu_state(model.encoder)
    data = batch()
    predictions = controls.forecast_train(model, data, cfg)
    core.pinball(predictions, data["y"], data["y_observed"]).backward()
    assert all(torch.equal(value, model.encoder.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.readout.parameters())


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_real_initial_encoder_and_head_equal_source_strong_endpoint(seed):
    cfg = controls.Config(seed=seed)
    before = torch.get_rng_state().clone()
    source = core.initialize_model(
        seed, **core.model_dimensions(controls.core_config(cfg)), method="cf_jepa"
    )
    model = controls.prepare_model(cfg)
    assert torch.equal(before, torch.get_rng_state())
    for part in ("encoder", "readout"):
        assert all(
            torch.equal(v, getattr(model, part).state_dict()[k])
            for k, v in getattr(source, part).state_dict().items()
        )
    assert sum(p.numel() for p in model.readout.parameters()) == 18565
    direct = controls.prepare_model(
        replace(cfg, method="cf_direct_supervised", updates=3000, cadence=750)
    )
    assert all(torch.equal(v, direct.state_dict()[k]) for k, v in model.state_dict().items())
    strong = controls.downstream.prepare_model(
        controls.downstream.DownstreamConfig(method="cf_jepa", seed=seed),
        controls.core_config(cfg),
        None,
    )
    for part in ("encoder", "readout"):
        assert all(
            torch.equal(v, getattr(strong, part).state_dict()[k])
            for k, v in getattr(model, part).state_dict().items()
        )
    assert not hasattr(model, "online") and not hasattr(model, "predictors")


def test_direct_backpropagates_encoder_without_ssl_or_ema():
    cfg = small_config("cf_direct_supervised")
    model = controls.prepare_model(cfg)
    before = core.cpu_state(model.encoder)
    data = batch()
    out = controls.forecast_train(model, data, cfg)
    core.pinball(out, data["y"], data["y_observed"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(not torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())
    assert all(k.startswith(("encoder.", "readout.")) for k in model.state_dict())


def test_samples_targets_masks_and_ignored_future_equal_original():
    pool = np.arange(30)
    for seed in (7, 13, 23):
        cfg = small_config(seed=seed)
        other = replace(cfg, method="cf_direct_supervised")
        for step in range(5):
            expected = core.batch_indices(pool, cfg.batch_size, seed, "readout", step)
            np.testing.assert_array_equal(controls.supervised_indices(pool, cfg, step), expected)
            np.testing.assert_array_equal(controls.supervised_indices(pool, other, step), expected)
    data = dict(zip(("x", "observed", "metadata", "query"), context(), strict=True))
    data.update(y=np.array([[1, 2, 3]] * 4, np.float32), y_observed=np.ones((4, 3), bool))
    data["y_observed"][0, 0] = False
    data["y"][0, 0] = np.nan
    got = core.tensor_batch(data, np.arange(4), scalers(), 96, "cpu")
    assert "future" not in got and got["y"][0, 0] == 0
    prediction = torch.zeros(4, 3, 5, requires_grad=True)
    core.pinball(prediction, got["y"], got["y_observed"]).backward()
    assert prediction.grad[0, 0].count_nonzero() == 0


@pytest.mark.parametrize("method", controls.METHODS)
def test_safe_forecast_and_encoder_exact_source_replay(method, monkeypatch):
    saved = artifact(method)
    original = controls.prepare_model(small_config(method)).eval()
    calls = []
    real_load = torch.load

    def observed_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observed_load)
    before = torch.get_rng_state().clone()
    loaded = api.load_inference(codec(saved))
    encoder = api.load_encoder(codec(artifact(method, encoder=True)))
    assert torch.equal(before, torch.get_rng_state())
    assert all(c["weights_only"] is True and c["map_location"] == "cpu" for c in calls)
    x, mask, metadata, query = context(7)
    with torch.no_grad():
        values = original.forecast(
            torch.from_numpy(scalers().channels(x, mask)),
            torch.from_numpy(mask),
            torch.from_numpy(metadata),
            torch.from_numpy(query),
        ).numpy()
        expected = (
            values * scalers().target_std[None, :, None] + scalers().target_mean[None, :, None]
        )
        latent = original.encoder.encode(
            torch.from_numpy(scalers().channels(x, mask)),
            torch.from_numpy(mask),
            torch.from_numpy(metadata),
        ).numpy()
    np.testing.assert_allclose(
        loaded.forecast(x, mask, metadata, query), expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(encoder.encode(x, mask, metadata), latent, rtol=1e-6, atol=1e-6)
    state = core.cpu_state(loaded.model)
    x[~mask] = np.nan
    np.testing.assert_array_equal(
        loaded.forecast(x, mask, metadata, query),
        loaded.forecast(np.where(mask, x, 1e20), mask, metadata, query),
    )
    assert all(torch.equal(v, loaded.model.state_dict()[k]) for k, v in state.items())
    assert all(not p.requires_grad and p.grad is None for p in loaded.model.parameters())
    assert np.all(query[:, :, 4] * 250 == 230)
    wrong = query.copy()
    wrong[:, :, 4] = 200 / 250
    with pytest.raises(ValueError):
        loaded.forecast(x, mask, metadata, wrong)
    exposed = loaded.config_metadata
    exposed["seed"] = 999
    assert loaded.config_metadata["seed"] == 7


def test_load_fixed_float32_without_default_dtype_or_rng_change():
    saved = artifact()
    before = torch.get_rng_state().clone()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        loaded = api.load_inference(codec(saved))
        assert torch.get_default_dtype() == torch.float64
        assert all(p.dtype == torch.float32 for p in loaded.model.parameters())
        assert torch.equal(before, torch.get_rng_state())
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize(
    "defect",
    [
        "shape",
        "mask_dtype",
        "primary_missing",
        "observed_nan",
        "metadata_nan",
        "frequency",
        "interval",
        "horizon",
        "extra_future",
    ],
)
def test_native_input_guards_and_context_only_signatures(defect):
    loaded = api.load_inference(codec(artifact()))
    x, mask, metadata, query = context()
    if defect == "shape":
        x = x[:, :24]
    elif defect == "mask_dtype":
        mask = mask.astype(np.int32)
    elif defect == "primary_missing":
        mask[0, 0, 0] = False
    elif defect == "observed_nan":
        x[0, 0, 0] = np.nan
    elif defect == "metadata_nan":
        metadata[0, 0, 0] = np.nan
    elif defect == "frequency":
        metadata[:, 0, 0] = 125000 / 455000
    elif defect == "interval":
        metadata[:, :, 1] = 0.5
    elif defect == "horizon":
        query[:, :, 9] = [1, 2, 3]
    else:
        with pytest.raises(TypeError):
            loaded.forecast(x, mask, metadata, query, targets=np.zeros((4, 3)))
        with pytest.raises(TypeError):
            loaded.encode(x, mask, metadata, query)
        return
    with pytest.raises(ValueError):
        loaded.forecast(x, mask, metadata, query)


@pytest.mark.parametrize(
    "change",
    [
        "old_kind",
        "resume",
        "missing_tensor",
        "extra_tensor",
        "nan_tensor",
        "dtype",
        "shape",
        "seed",
        "scaler",
        "ancestry",
        "core_config",
    ],
)
def test_bad_artifacts_fail_closed(change):
    saved = artifact()
    if change == "old_kind":
        saved["kind"] = "native_ssl_weights_only_inference_v1"
    elif change == "resume":
        saved["optimizer"] = {}
    elif change == "missing_tensor":
        saved["model"].pop(next(iter(saved["model"])))
    elif change == "extra_tensor":
        saved["model"]["unsafe.extra"] = torch.zeros(1)
    elif change in ("nan_tensor", "dtype", "shape"):
        key = next(k for k, v in saved["model"].items() if v.is_floating_point())
        if change == "nan_tensor":
            saved["model"][key].fill_(float("nan"))
        elif change == "dtype":
            saved["model"][key] = saved["model"][key].double()
        else:
            saved["model"][key] = torch.zeros(1)
    elif change == "seed":
        saved["config"]["seed"] = 9
    elif change == "scaler":
        saved["scalers"]["channel_std"][0] = 0
    elif change == "ancestry":
        saved["supervised_ancestry"]["ssl_updates"] = 1
    else:
        saved["core_config"]["lr"] = 0.3
    with pytest.raises(ValueError):
        api.load_inference(codec(saved))


@pytest.mark.parametrize("method", controls.METHODS)
def test_control_kinds_cannot_be_loaded_as_pretrained_ssl(method):
    with pytest.raises(ValueError):
        NativeAcousticEncoder(codec(artifact(method, encoder=True)))
    wrong = artifact(method, encoder=True)
    wrong["kind"] = controls.ENCODER_KINDS[
        controls.METHODS[1] if method == controls.METHODS[0] else controls.METHODS[0]
    ]
    with pytest.raises(ValueError):
        api.load_encoder(codec(wrong))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 9),
        ("history", 24),
        ("width", 192),
        ("latent", 64),
        ("blocks", 4),
        ("lr", 0.01),
        ("updates", 5001),
        ("seed", True),
    ],
)
def test_real_recipe_has_no_variants(field, value):
    with pytest.raises(ValueError):
        replace(controls.Config(), **{field: value}).validate()
