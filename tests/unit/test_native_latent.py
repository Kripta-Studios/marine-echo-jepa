"""SYNTHETIC_CORRECTNESS_ONLY native latent API checks, CPU and in-memory."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "_latent_test_support",
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


@pytest.fixture
def api():
    return support.load_api()


@pytest.mark.parametrize(
    "method,band",
    [("shared_ssl", False), ("permuted_ssl", False), ("shared_ssl", True), ("permuted_ssl", True)],
)
def test_shared_predictor_and_selected_embedding_exact_replay(api, method, band):
    artifact, model = support.fixture(method, band)
    arrays = support.inputs()
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    with torch.inference_mode():
        x, observed, metadata, query = support.tensors(artifact, arrays)
        encoded = model.encoder.encode(x, observed, metadata)
        expected = model.predictor(encoded, query)
    np.testing.assert_allclose(adapter.encode(*arrays[:3]), encoded.numpy(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        adapter.predict_latents(*arrays), expected.numpy(), atol=2e-6, rtol=2e-6
    )
    assert adapter.predict_latents(*arrays).shape == (3, 3, 4)
    assert adapter.training_kind == (
        "permuted_pairing_ssl_control" if method == "permuted_ssl" else "ssl_pretrained"
    )


def test_cf_uses_online_sequence_and_selected_ema_separately(api):
    artifact, model = support.fixture("cf_jepa")
    arrays = support.inputs()
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    with torch.inference_mode():
        x, observed, metadata, _ = support.tensors(artifact, arrays)
        expected = torch.stack(
            [p(model.online.sequence(x, observed, metadata)) for p in model.predictors], dim=2
        ).numpy()
        wrong_branch = torch.stack(
            [p(model.encoder.sequence(x, observed, metadata)) for p in model.predictors], dim=2
        ).numpy()
        selected = model.encoder.encode(x, observed, metadata).numpy()
    assert np.max(np.abs(expected - wrong_branch)) > 0.01
    np.testing.assert_allclose(
        adapter.predict_cf_zones(*arrays[:3]), expected, atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(adapter.encode(*arrays[:3]), selected, atol=2e-6, rtol=2e-6)
    assert adapter.predict_cf_zones(*arrays[:3]).shape == (3, 96, 3, 6)
    assert adapter.feature_branch == "selected_ema"
    assert "outside" in adapter.objective_metadata["input_view"]
    with pytest.raises(ValueError, match="ordinal|zones"):
        adapter.predict_latents(*arrays)


@pytest.mark.parametrize(
    "method,band",
    [
        ("masked_ssl", False),
        ("random_frozen", False),
        ("direct", False),
        ("masked_ssl", True),
        ("random_frozen", True),
        ("direct", True),
    ],
)
def test_untrained_latent_heads_cannot_be_presented_as_learned(api, method, band):
    artifact, _ = support.fixture(method, band)
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    assert np.isfinite(adapter.encode(*support.inputs()[:3])).all()
    with pytest.raises(ValueError, match="untrained"):
        adapter.predict_latents(*support.inputs())


@pytest.mark.parametrize(
    "method,band", [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]
)
def test_masked_fills_batching_and_state_are_invariant(api, method, band):
    artifact, _ = support.fixture(method, band)
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    arrays = support.inputs(5)
    before = {k: v.clone() for k, v in adapter._model.state_dict().items()}
    predict = adapter.predict_cf_zones if method == "cf_jepa" else adapter.predict_latents
    args = arrays[:3] if method == "cf_jepa" else arrays
    expected = predict(*args)
    changed = list(arrays)
    changed[0] = arrays[0].copy()
    changed[0][~arrays[1]] = np.nan
    np.testing.assert_array_equal(
        predict(*(changed[:3] if method == "cf_jepa" else changed)), expected
    )
    pieces = [predict(*(a[i : i + 1] for a in args)) for i in range(5)]
    np.testing.assert_allclose(np.concatenate(pieces), expected, atol=3e-6, rtol=3e-6)
    for k, v in adapter._model.state_dict().items():
        assert torch.equal(v, before[k])
    assert all(not p.requires_grad and p.grad is None for p in adapter._model.parameters())
    assert not any(m.training for m in adapter._model.modules())
    assert expected.dtype == np.float32


@pytest.mark.parametrize(
    "change",
    [
        "x_shape",
        "mask_type",
        "mask_shape",
        "primary_missing",
        "observed_nan",
        "metadata_nan",
        "frequency",
        "interval",
        "bounds",
        "offset",
        "query_frequency",
        "query_bounds",
        "query_horizon",
        "query_nan",
    ],
)
def test_native_input_failures(api, change):
    artifact, _ = support.fixture()
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    x, observed, metadata, query = support.inputs()
    if change == "x_shape":
        x = x[:, :-1]
    elif change == "mask_type":
        observed = observed.astype(np.int8)
    elif change == "mask_shape":
        observed = observed[:, :-1]
    elif change == "primary_missing":
        observed[0, 0, 0] = False
    elif change == "observed_nan":
        x[0, 0, 0] = np.nan
    elif change == "metadata_nan":
        metadata[0, 0, 5] = np.nan
    elif change == "frequency":
        metadata[0, 0, 0] = 0.2
    elif change == "interval":
        metadata[0, 0, 1] = 0.5
    elif change == "bounds":
        metadata[0, 0, 4] = 0
    elif change == "offset":
        metadata[0, 0, 9] = 1
    elif change == "query_frequency":
        query[0, 0, 0] = 0.2
    elif change == "query_bounds":
        query[0, 0, 4] = 0.8
    elif change == "query_horizon":
        query[0, 0, 9] = 2
    elif change == "query_nan":
        query[0, 0, 3] = np.nan
    with pytest.raises(ValueError):
        adapter.predict_latents(x, observed, metadata, query)


@pytest.mark.parametrize(
    "kind",
    [
        "native_ssl_resume_v1",
        "native_ssl_selected_encoder_v1",
        "native_band_ssl_resume_v1",
        "native_downstream_resume_v1",
        "native_band_downstream_supervised_encoder_v1",
        "bogus",
    ],
)
def test_unsafe_artifact_kinds_rejected(api, kind):
    artifact, _ = support.fixture()
    artifact["kind"] = kind
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))


@pytest.mark.parametrize(
    "change",
    ["extra", "missing", "shape", "dtype", "nonfinite", "not_tensor", "negative_bn_variance"],
)
def test_exact_finite_state_required(api, change):
    artifact, _ = support.fixture("cf_jepa")
    state = artifact["model"]
    k = "online.input_fc.weight"
    if change == "extra":
        state["bogus"] = torch.zeros(1)
    elif change == "missing":
        del state[k]
    elif change == "shape":
        state[k] = state[k][:1]
    elif change == "dtype":
        state[k] = state[k].double()
    elif change == "nonfinite":
        state[k][0, 0] = float("nan")
    elif change == "not_tensor":
        state[k] = 5
    elif change == "negative_bn_variance":
        state["online.blocks.0.bn1.running_var"][0] = -1
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))


@pytest.mark.parametrize(
    "change",
    [
        "missing_config",
        "missing_field",
        "unknown_field",
        "float_heads",
        "nan_lr",
        "zero_width",
        "history24",
        "bad_method",
        "missing_scaler",
        "std_zero",
        "scaler_nan",
        "scaler_overflow",
        "missing_bindings",
        "bad_digest",
        "wrong_architecture",
        "downstream",
        "optimizer",
        "zero_selected_step",
        "no_step",
        "unmarked_tiny",
    ],
)
def test_config_scaler_ancestry_and_objective_guards(api, change):
    artifact, _ = support.fixture()
    if change == "missing_config":
        del artifact["config"]
    elif change == "missing_field":
        del artifact["config"]["heads"]
    elif change == "unknown_field":
        artifact["config"]["unknown"] = 1
    elif change == "float_heads":
        artifact["config"]["heads"] = 2.0
    elif change == "nan_lr":
        artifact["config"]["lr"] = float("nan")
    elif change == "zero_width":
        artifact["config"]["width"] = 0
    elif change == "history24":
        artifact["config"]["history"] = 24
    elif change == "bad_method":
        artifact["config"]["method"] = "bogus"
    elif change == "missing_scaler":
        del artifact["scalers"]["target_std"]
    elif change == "std_zero":
        artifact["scalers"]["channel_std"][0] = 0
    elif change == "scaler_nan":
        artifact["scalers"]["channel_mean"][0] = float("nan")
    elif change == "scaler_overflow":
        artifact["scalers"]["channel_mean"][0] = 1e300
    elif change == "missing_bindings":
        artifact["bindings"] = {}
    elif change == "bad_digest":
        artifact["bindings"] = {"not/executed": "bad"}
    elif change == "wrong_architecture":
        artifact["architecture"] = "nonlinear_frequency_conditioned_v1"
    elif change == "downstream":
        artifact["supervised_ancestry"] = {"mode": "full_finetune"}
    elif change == "optimizer":
        artifact["optimizer"] = {}
    elif change == "zero_selected_step":
        artifact["selected_pretrain_step"] = 0
    elif change == "no_step":
        del artifact["selected_pretrain_step"]
    elif change == "unmarked_tiny":
        del artifact["evidence_kind"]
        del artifact["correctness_smoke"]
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "cpu:0", "meta", "mps", "bogus"])
def test_only_explicit_cpu_or_cuda0(api, device):
    artifact, _ = support.fixture()
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact), device=device)


def test_metadata_copies_and_context_only_signatures(api):
    artifact, _ = support.fixture("permuted_ssl")
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    config = adapter.config_metadata
    config["history"] = 1
    scalers = adapter.scaler_metadata
    scalers["channel_mean"][0] = 1e10
    bindings = adapter.source_bindings
    bindings.clear()
    objective = adapter.objective_metadata
    objective["horizons"][0] = 50
    assert adapter.config_metadata["history"] == 96
    assert adapter.scaler_metadata == artifact["scalers"]
    assert adapter.source_bindings == artifact["bindings"]
    assert adapter.objective_metadata["horizons"] == [1, 3, 6]
    assert list(inspect.signature(adapter.encode).parameters) == ["x", "observed", "metadata"]
    assert list(inspect.signature(adapter.predict_cf_zones).parameters) == [
        "x",
        "observed",
        "metadata",
    ]
    with pytest.raises(TypeError):
        adapter.encode(*support.inputs())
    with pytest.raises(TypeError):
        adapter.predict_latents(*support.inputs(), future=np.zeros(1))


def test_primary230_native_query_and_metadata_affect_predictor(api):
    artifact, _ = support.fixture(band=True)
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    arrays = support.inputs()
    original = adapter.predict_latents(*arrays)
    assert arrays[2][0, 0, 4] * 250 == pytest.approx(230)
    assert arrays[3][0, 0, 4] * 250 == pytest.approx(230)
    changed = support.inputs(lower=225)
    assert np.max(np.abs(adapter.predict_latents(*changed) - original)) > 1e-7
    assert arrays[3][0, 0, 4] * 250 == pytest.approx(230)


def test_band_and_legacy_schema_not_reinterpreted(api):
    artifact, _ = support.fixture(band=True)
    del artifact["architecture"]
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))
    artifact, _ = support.fixture()
    artifact["kind"] = "native_band_ssl_weights_only_inference_v1"
    artifact["architecture"] = "nonlinear_frequency_conditioned_v1"
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))


def test_batch_failures_are_not_silently_nonfinite(api):
    artifact, _ = support.fixture()
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    arrays = list(support.inputs())
    arrays[0] = arrays[0].astype(np.float64)
    arrays[0][0, 0, 0] = np.finfo(np.float64).max  # normalization overflows float32
    with pytest.raises(ValueError):
        adapter.predict_latents(*arrays)


@pytest.mark.parametrize(
    "method,band", [("shared_ssl", False), ("cf_jepa", False), ("shared_ssl", True)]
)
def test_empty_secondary_channels_and_caller_arrays_preserved(api, method, band):
    artifact, _ = support.fixture(method, band)
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    arrays = list(support.inputs())
    arrays[1][:, :, 1:] = False
    before = [a.copy() for a in arrays]
    predict = adapter.predict_cf_zones if method == "cf_jepa" else adapter.predict_latents
    args = arrays[:3] if method == "cf_jepa" else arrays
    expected = predict(*args)
    for actual, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, original)
    arrays[0][~arrays[1]] = np.inf
    np.testing.assert_array_equal(
        predict(*(arrays[:3] if method == "cf_jepa" else arrays)), expected
    )
    assert np.isfinite(expected).all()


def test_finite_but_extreme_weights_cannot_return_nonfinite_coordinates(api):
    artifact, _ = support.fixture()
    artifact["model"]["predictor.net.0.weight"].zero_()
    artifact["model"]["predictor.net.0.bias"].fill_(2)
    artifact["model"]["predictor.net.2.weight"].fill_(torch.finfo(torch.float32).max)
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    assert np.isfinite(adapter.encode(*support.inputs()[:3])).all()
    with pytest.raises(ValueError, match="finite float32"):
        adapter.predict_latents(*support.inputs())


def test_unused_readout_and_regularizer_are_validated_but_never_executed(api, monkeypatch):
    artifact, _ = support.fixture()
    artifact["model"]["regularizer.global_step"].fill_(19)
    adapter = api.NativeLatentPredictor(support.codec(artifact))

    def forbidden(*args, **kwargs):
        raise AssertionError("No readout, reconstruction, loss or regularizer forward is needed.")

    monkeypatch.setattr(adapter._model.readout, "forward", forbidden)
    monkeypatch.setattr(adapter._model.regularizer, "forward", forbidden)
    monkeypatch.setattr(adapter._model.reconstruction, "forward", forbidden)
    monkeypatch.setattr(adapter._model, "shared_loss", forbidden)
    assert np.isfinite(adapter.predict_latents(*support.inputs())).all()
    assert adapter._model.regularizer.global_step.item() == 19
    artifact["model"]["readout.net.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        api.NativeLatentPredictor(support.codec(artifact))


def test_shared_cannot_call_cf_or_omit_issued_query(api):
    artifact, _ = support.fixture()
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    with pytest.raises(ValueError, match="Only CF"):
        adapter.predict_cf_zones(*support.inputs()[:3])
    with pytest.raises(ValueError, match="query"):
        adapter.predict_latents(*support.inputs()[:3], None)
    result = adapter.predict_latents(*support.inputs())
    assert not np.allclose(result[:, 0], result[:, 2])


@pytest.mark.parametrize("band", [False, True])
def test_exact_production_schema_with_only_synthetic_zero_tensors(api, band):
    """Synthetic schema codec, not a fitted artifact or scientific report.

    Original inference has no evidence_kind field; band explicitly does. Use
    full-size empty meta templates, with zero tensors only, to test that exact
    production schema without opening any public selected/checkpoint file.
    """
    cls = support.BandConfig if band else support.Config
    config = cls()
    with torch.device("meta"):
        model = (support.NativeBandTemporalModel if band else support.NativeTemporalModel)().float()
    state = {
        k: torch.zeros(v.shape, dtype=v.dtype, device="cpu") for k, v in model.state_dict().items()
    }
    artifact, _ = support.fixture(band=band)
    artifact.update(config=config.to_dict(), model=state, selected_pretrain_step=1500)
    if band:
        artifact.update(evidence_kind="REAL_TRAIN_DEVELOPMENT_FIT", correctness_smoke=False)
    else:
        del artifact["evidence_kind"]
        del artifact["correctness_smoke"]
    adapter = api.NativeLatentPredictor(support.codec(artifact))
    output = adapter.predict_latents(*support.inputs(1))
    assert output.shape == (1, 3, 64)
    np.testing.assert_array_equal(output, np.zeros_like(output))
