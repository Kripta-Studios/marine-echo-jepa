"""SYNTHETIC_CORRECTNESS_ONLY selected-encoder CPU API correctness."""

import copy
import hashlib
import importlib.util
import io
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
from marine_echo.models import native_temporal
from marine_echo.training import native_ssl as core

spec = importlib.util.spec_from_file_location(
    "native_encoder_candidate", BUILDER / "src/marine_echo/inference/native_encoder.py"
)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


@pytest.fixture(autouse=True)
def cpu_only():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    assert (
        Path(native_temporal.__file__).resolve()
        == MAIN / "src/marine_echo/models/native_temporal.py"
    )
    assert Path(core.__file__).resolve() == MAIN / "src/marine_echo/training/native_ssl.py"
    assert not torch.cuda.is_initialized()
    yield
    torch.set_num_threads(threads)


def codec(artifact):
    stream = io.BytesIO()
    torch.save(artifact, stream)
    stream.seek(0)
    return stream


def fixture(method="shared_ssl", history=24, batch_size=3):
    config = core.Config(
        method=method,
        history=history,
        batch_size=batch_size,
        width=24,
        latent=8,
        blocks=1,
        heads=4,
        cf_width=24,
        cf_latent=8,
        cf_blocks=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        if method == "cf_jepa":
            encoder = native_temporal.CFTemporalEncoder(
                config.cf_width, config.cf_latent, config.cf_blocks
            )
            # Serialize a distinguishable selected EMA state, including BN buffers.
            with torch.no_grad():
                for key, value in encoder.state_dict().items():
                    if "running_mean" in key:
                        value.add_(0.17)
                    elif "running_var" in key:
                        value.mul_(1.3)
        else:
            encoder = native_temporal.SharedTemporalEncoder(
                config.width, config.latent, config.blocks, config.heads
            )
    encoder.eval().requires_grad_(False)
    scalers = core.Scalers(
        np.array([-81, -76, -73, -68], np.float32),
        np.array([4, 5, 6, 7], np.float32),
        np.array([-80, -79, -78], np.float32),
        np.array([3, 4, 5], np.float32),
    )
    artifact = {
        "kind": "native_ssl_selected_encoder_v1",
        "config": config.to_dict(),
        "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
        "scalers": scalers.to_dict(),
        "bindings": {
            str((MAIN / "SYNTHETIC_CORRECTNESS_ONLY_NEVER_OPEN.py").resolve()): hashlib.sha256(
                b"synthetic provenance, not executable source"
            ).hexdigest()
        },
    }
    n = 7
    rng = np.random.default_rng(29)
    x = rng.normal(-78, 4, (n, history, 4)).astype(np.float32)
    observed = rng.random(x.shape) > 0.35
    observed[:, :, 0] = True
    observed[0, :, 3] = False
    metadata = np.zeros((n, 4, 10), np.float32)
    metadata[:, :, 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    bounds = np.array(
        [[0, 230], [0, 200], [10, 190], [20, 250], [0, 170], [25, 225], [5, 230]], np.float32
    )
    metadata[:, :, 3:5] = bounds[:, None] / 250
    metadata[:, :, 6] = 1
    return artifact, encoder, scalers, (x, observed, metadata)


@pytest.mark.parametrize(
    "method", ["shared_ssl", "masked_ssl", "cf_jepa", "permuted_ssl", "direct", "random_frozen"]
)
@pytest.mark.parametrize("history", [24, 96])
def test_exact_main_backbone_replay_geometry_masks_and_frozen_state(method, history):
    artifact, expected_encoder, scalers, inputs = fixture(method, history)
    instance = api.NativeAcousticEncoder(codec(artifact))
    state = {k: v.clone() for k, v in instance.encoder.state_dict().items()}
    original = tuple(value.copy() for value in inputs)
    expected = []
    with torch.inference_mode():
        for start in range(0, len(inputs[0]), 3):
            x, observed, metadata = (v[start : start + 3] for v in inputs)
            expected.append(
                expected_encoder.encode(
                    torch.from_numpy(scalers.channels(x, observed)),
                    torch.from_numpy(observed),
                    torch.from_numpy(metadata),
                ).numpy()
            )
    actual = instance.encode(*inputs)
    np.testing.assert_array_equal(actual, np.concatenate(expected))
    assert actual.shape == (7, 8) and actual.dtype == np.float32
    for before, after in zip(original, inputs, strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(inputs[2][0, :, 3:5], np.tile([0, np.float32(230 / 250)], (4, 1)))
    again = instance.encode(*inputs)
    np.testing.assert_array_equal(actual, again)
    assert all(
        torch.equal(value, instance.encoder.state_dict()[key]) for key, value in state.items()
    )
    assert all(not p.requires_grad and p.grad is None for p in instance.encoder.parameters())
    assert not instance.encoder.training
    assert instance.method == method
    assert instance.source_bindings == artifact["bindings"]
    assert instance.scaler_metadata == artifact["scalers"]
    assert instance.config_metadata == artifact["config"]


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
def test_missing_secondary_fill_invariance_and_geometry_affects_features(method):
    artifact, _, _, inputs = fixture(method)
    instance = api.NativeAcousticEncoder(codec(artifact))
    x, observed, metadata = inputs
    expected = instance.encode(x, observed, metadata)
    for fill in (0.0, 1e9, float("nan"), float("inf")):
        changed = x.copy()
        changed[~observed] = fill
        np.testing.assert_array_equal(instance.encode(changed, observed, metadata), expected)
    changed_meta = metadata.copy()
    changed_meta[0, :, 4] = 200 / 250
    changed = instance.encode(x, observed, changed_meta)
    assert not np.array_equal(changed[0], expected[0])
    np.testing.assert_array_equal(changed[1:], expected[1:])


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
def test_batch_split_and_no_head_query_target_dependency(monkeypatch, method):
    artifact, _, _, inputs = fixture(method)
    expected = api.NativeAcousticEncoder(codec(artifact)).encode(*inputs)
    for name in ("initialize_model", "check_prefit", "load_corpus"):
        monkeypatch.setattr(
            core,
            name,
            lambda *a, **k: pytest.fail("Inference entered training/full-model machinery."),
        )
    monkeypatch.setattr(
        core.Scalers, "fit", lambda *a, **k: pytest.fail("Inference refitted a scaler.")
    )
    for name in ("NativeTemporalModel", "CFNativeModel", "QueryHead"):
        monkeypatch.setattr(
            native_temporal,
            name,
            lambda *a, **k: pytest.fail("Inference constructed a head/full model."),
        )
    monkeypatch.setattr(
        torch, "manual_seed", lambda *a, **k: pytest.fail("Inference reset randomness.")
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda *a, **k: pytest.fail("Inference changed determinism/cache policy."),
    )
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("Inference initialized an optimizer.")
    )
    instance = api.NativeAcousticEncoder(codec(artifact))
    np.testing.assert_array_equal(instance.encode(*inputs), expected)
    parts = [instance.encode(*(v[a:b] for v in inputs)) for a, b in ((0, 3), (3, 6), (6, 7))]
    np.testing.assert_array_equal(np.concatenate(parts), expected)
    for key in ("query", "future", "targets", "y"):
        with pytest.raises(TypeError):
            instance.encode(*inputs, **{key: np.zeros(1)})
    with pytest.raises(ValueError):
        instance.encode({"x": inputs[0], "future": np.zeros(1)}, inputs[1], inputs[2])
    assert not hasattr(instance, "model") and not hasattr(instance, "readout")


@pytest.mark.parametrize(
    "change",
    [
        "history",
        "channels",
        "mask_shape",
        "mask_dtype",
        "primary_missing",
        "empty",
        "future_suffix",
        "metadata_shape",
        "frequency",
        "channel_order",
        "lower",
        "upper",
        "geometry",
        "interval",
        "offset",
        "metadata_nan",
        "observed_nan",
        "observed_inf",
        "orientation",
        "known_flag",
    ],
)
def test_invalid_native_inputs_rejected(change):
    artifact, _, _, (x, mask, meta) = fixture()
    instance = api.NativeAcousticEncoder(codec(artifact))
    if change == "history":
        x, mask = x[:, :-1], mask[:, :-1]
    elif change == "channels":
        x, mask = x[:, :, :3], mask[:, :, :3]
    elif change == "mask_shape":
        mask = mask[:, :-1]
    elif change == "mask_dtype":
        mask = mask.astype(np.float32)
    elif change == "primary_missing":
        mask[0, 0, 0] = False
    elif change == "empty":
        x, mask, meta = x[:0], mask[:0], meta[:0]
    elif change == "future_suffix":
        x, mask = np.concatenate([x, x[:, :4]], 1), np.concatenate([mask, mask[:, :4]], 1)
    elif change == "metadata_shape":
        meta = meta[:, :, :9]
    elif change in ("observed_nan", "observed_inf"):
        x[0, 0, 0] = np.nan if change == "observed_nan" else np.inf
    else:
        field, value = {
            "frequency": (0, 1),
            "lower": (4, 0),
            "upper": (3, -0.1),
            "geometry": (2, 0),
            "interval": (1, 0.5),
            "offset": (9, 3),
            "metadata_nan": (8, np.nan),
            "orientation": (5, 1),
            "known_flag": (6, 2),
        }.get(change, (0, 0))
        if change == "channel_order":
            meta[:, :, 0] = meta[:, ::-1, 0]
        else:
            meta[0, 0, field] = value
    with pytest.raises(ValueError):
        instance.encode(x, mask, meta)


@pytest.mark.parametrize(
    "change",
    [
        "kind",
        "config_missing",
        "dimension_missing",
        "dimension_type",
        "dimension_zero",
        "history",
        "batch",
        "method",
        "state_missing",
        "tensor_missing",
        "extra_tensor",
        "tensor_shape",
        "tensor_nan",
        "tensor_dtype",
        "scalers_missing",
        "scaler_shape",
        "scaler_nan",
        "scaler_zero",
        "bindings_missing",
    ],
)
def test_incompatible_artifacts_fail_closed(change):
    artifact, _, _, _ = fixture()
    if change == "kind":
        artifact["kind"] = "native_downstream_supervised_encoder_v1"
    elif change == "config_missing":
        del artifact["config"]
    elif change.startswith("dimension"):
        if change == "dimension_missing":
            del artifact["config"]["width"]
        else:
            artifact["config"]["width"] = 24.0 if change == "dimension_type" else 0
    elif change in ("history", "batch", "method"):
        key, value = {
            "history": ("history", 48),
            "batch": ("batch_size", 0),
            "method": ("method", "invented"),
        }[change]
        artifact["config"][key] = value
    elif change == "state_missing":
        del artifact["encoder"]
    elif change.startswith("tensor") or change == "extra_tensor":
        key = next(iter(artifact["encoder"]))
        if change == "tensor_missing":
            del artifact["encoder"][key]
        elif change == "extra_tensor":
            artifact["encoder"]["readout.fake_head"] = torch.zeros(1)
        elif change == "tensor_shape":
            artifact["encoder"][key] = torch.zeros(1)
        elif change == "tensor_dtype":
            artifact["encoder"][key] = artifact["encoder"][key].double()
        else:
            artifact["encoder"][key].flatten()[0] = torch.nan
    elif change == "scalers_missing":
        del artifact["scalers"]
    elif change.startswith("scaler"):
        if change == "scaler_shape":
            artifact["scalers"]["channel_mean"] = [0]
        elif change == "scaler_nan":
            artifact["scalers"]["target_mean"][0] = float("nan")
        else:
            artifact["scalers"]["channel_std"][0] = 0
    else:
        del artifact["bindings"]
    with pytest.raises(ValueError):
        api.NativeAcousticEncoder(codec(artifact))


@pytest.mark.parametrize(
    "method,label",
    [
        ("shared_ssl", "ssl_pretrained"),
        ("masked_ssl", "ssl_pretrained"),
        ("cf_jepa", "ssl_pretrained"),
        ("permuted_ssl", "permuted_pairing_ssl_control"),
        ("direct", "supervised_feature_encoder"),
        ("random_frozen", "untrained_control"),
    ],
)
def test_training_kind_and_metadata_are_honest_and_defensive(method, label):
    artifact, _, _, _ = fixture(method)
    instance = api.NativeAcousticEncoder(codec(artifact))
    assert instance.training_kind == label
    assert instance.selected_pretrain_step is None
    exposed = instance.scaler_metadata
    exposed["channel_mean"][0] = 1234
    assert instance.scaler_metadata == artifact["scalers"]
    exposed = instance.source_bindings
    exposed.clear()
    assert instance.source_bindings == artifact["bindings"]
    assert instance.feature_branch == ("selected_ema" if method == "cf_jepa" else "shared_encoder")


def test_safe_loading_exact_arguments_and_no_source_execution(monkeypatch):
    artifact, _, _, inputs = fixture()
    original_load = torch.load
    calls = []

    def safe_load(weights, **kwargs):
        calls.append(kwargs)
        return original_load(weights, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)
    monkeypatch.setattr(
        Path, "open", lambda *a, **k: pytest.fail("Ancestral source binding was opened/executed.")
    )
    instance = api.NativeAcousticEncoder(codec(artifact))
    assert calls == [{"weights_only": True, "map_location": "cpu"}]
    assert instance.encode(*inputs).shape == (7, 8)


def test_selected_step_optional_and_cf_buffers_remain_eval_frozen():
    artifact, _, _, inputs = fixture("cf_jepa")
    artifact["selected_pretrain_step"] = 1500
    instance = api.NativeAcousticEncoder(codec(artifact))
    assert instance.selected_pretrain_step == 1500
    state = copy.deepcopy(instance.encoder.state_dict())
    instance.encoder.train()  # encode reasserts inference semantics.
    instance.encode(*inputs)
    assert all(torch.equal(v, instance.encoder.state_dict()[k]) for k, v in state.items())
    assert not instance.encoder.training


class UnsafePayload:
    def __reduce__(self):
        return eval, ("1 + 1",)


@pytest.mark.parametrize("method,dimension", [("shared_ssl", 64), ("cf_jepa", 128)])
def test_default_config_backbone_dimensions_replay_without_predictor(method, dimension):
    artifact, _, scalers, inputs = fixture(method)
    config = core.Config(method=method, history=24, batch_size=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        if method == "cf_jepa":
            reference = native_temporal.CFTemporalEncoder(
                config.cf_width, config.cf_latent, config.cf_blocks
            )
        else:
            reference = native_temporal.SharedTemporalEncoder(
                config.width, config.latent, config.blocks, config.heads
            )
    reference.eval().requires_grad_(False)
    artifact["config"] = config.to_dict()
    artifact["encoder"] = reference.state_dict()
    instance = api.NativeAcousticEncoder(codec(artifact))
    inputs = tuple(v[:3] for v in inputs)
    expected = []
    with torch.inference_mode():
        for start in (0, 2):
            x, observed, metadata = (v[start : start + 2] for v in inputs)
            expected.append(
                reference.encode(
                    torch.from_numpy(scalers.channels(x, observed)),
                    torch.from_numpy(observed),
                    torch.from_numpy(metadata),
                ).numpy()
            )
    assert instance.embedding_dimension == dimension
    np.testing.assert_array_equal(instance.encode(*inputs), np.concatenate(expected))


def test_real_weights_only_codec_rejects_unsafe_pickle_payload():
    artifact, _, _, _ = fixture()
    artifact["unsafe_extra"] = UnsafePayload()
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        api.NativeAcousticEncoder(codec(artifact))


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
def test_load_encode_preserves_rng_and_determinism_policy(method):
    artifact, _, _, inputs = fixture(method)
    stream = codec(artifact)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    api.NativeAcousticEncoder(stream).encode(*inputs)
    assert random.getstate() == python_rng
    for before, after in zip(numpy_rng, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(before, after)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
def test_saved_float32_backbone_independent_of_caller_default_dtype(method):
    artifact, _, _, inputs = fixture(method)
    expected = api.NativeAcousticEncoder(codec(artifact)).encode(*inputs)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        instance = api.NativeAcousticEncoder(codec(artifact))
        np.testing.assert_array_equal(instance.encode(*inputs), expected)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
def test_float64_and_reversed_numpy_batches_keep_native_values(method):
    artifact, _, _, inputs = fixture(method)
    instance = api.NativeAcousticEncoder(codec(artifact))
    reversed_inputs = tuple(v[::-1] for v in inputs)
    expected = instance.encode(*(v.copy() for v in reversed_inputs))
    np.testing.assert_array_equal(instance.encode(*reversed_inputs), expected)
    x, mask, meta = reversed_inputs
    # The saved helper normalizes float64 raw dB before casting to float32;
    # arithmetic rounding can differ slightly from float32 normalization.
    np.testing.assert_allclose(
        instance.encode(x.astype(np.float64), mask, meta.astype(np.float64)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "change",
    [
        "observed_overflow",
        "metadata_overflow",
        "bounds_collapse",
        "scaler_overflow",
        "scaler_underflow",
        "bad_binding",
        "negative_step",
        "cf_buffer_dtype",
        "cf_buffer_nan",
    ],
)
def test_precision_and_cf_state_admission_fail_closed(change):
    artifact, _, _, (x, mask, meta) = fixture("cf_jepa")
    if change.startswith("scaler"):
        field, value = (
            ("channel_mean", 1e100) if change == "scaler_overflow" else ("channel_std", 1e-100)
        )
        artifact["scalers"][field][0] = value
    elif change == "bad_binding":
        artifact["bindings"] = {"never_open_this": "not-a-digest"}
    elif change == "negative_step":
        artifact["selected_pretrain_step"] = -1
    elif change.startswith("cf_buffer"):
        key = next(k for k in artifact["encoder"] if "running_var" in k)
        if change == "cf_buffer_dtype":
            artifact["encoder"][key] = artifact["encoder"][key].double()
        else:
            artifact["encoder"][key][0] = torch.nan
    else:
        x, meta = x.astype(np.float64), meta.astype(np.float64)
        if change == "observed_overflow":
            x[0, 0, 0] = 1e100
        elif change == "metadata_overflow":
            meta[0, 0, 4] = 1e100
        else:
            meta[0, 0, 3:5] = [0.8, 0.8 + 1e-12]
        instance = api.NativeAcousticEncoder(codec(artifact))
        with pytest.raises(ValueError):
            instance.encode(x, mask, meta)
        return
    with pytest.raises(ValueError):
        api.NativeAcousticEncoder(codec(artifact))
