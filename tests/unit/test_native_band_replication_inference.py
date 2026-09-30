"""SYNTHETIC_CORRECTNESS_ONLY safe CPU codecs/inference, no scientific weights."""

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
from marine_echo.inference.native_acoustic import NativeAcousticPredictor
from replication_test_support import artifact, codec, synthetic_inputs

from marine_echo.inference.native_band_replication_acoustic import (
    TRAINING_KINDS,
    NativeBandAcousticEncoder,
    NativeBandAcousticPredictor,
)
from marine_echo.inference.native_encoder import NativeAcousticEncoder


@pytest.mark.parametrize("method", list(TRAINING_KINDS))
def test_safe_selected_encoder_replays_without_head_query_rng_or_state_updates(method):
    saved, model = artifact(method=method, selected=True)
    x, mask, metadata, _ = synthetic_inputs(n=5)
    state = torch.get_rng_state().clone()
    encoder = NativeBandAcousticEncoder(codec(saved))
    assert torch.equal(state, torch.get_rng_state())
    assert not hasattr(encoder, "model") and not hasattr(encoder, "readout")
    before = {k: v.clone() for k, v in encoder.encoder.state_dict().items()}
    values = encoder.scalers.channels(x, mask)
    with torch.no_grad():
        expected = model.encoder.encode(
            torch.from_numpy(values), torch.from_numpy(mask), torch.from_numpy(metadata)
        ).numpy()
    actual = encoder.encode(x, mask, metadata)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual.shape == (5, 8) and actual.dtype == np.float32
    assert encoder.training_kind == TRAINING_KINDS[method]
    assert all(torch.equal(v, encoder.encoder.state_dict()[k]) for k, v in before.items())
    assert not encoder.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in encoder.encoder.parameters())
    bindings = encoder.source_bindings
    bindings.clear()
    assert encoder.source_bindings == saved["bindings"]
    bad_fill = x.copy()
    bad_fill[~mask] = np.nan
    np.testing.assert_array_equal(actual, encoder.encode(bad_fill, mask, metadata))


def test_forecast_target_free_replay_batch_split_native_bounds_and_frozen_full_state():
    saved, model = artifact()
    x, mask, metadata, query = synthetic_inputs(n=5)
    metadata[:, 1:, 3] = 10 / 250
    metadata[:, 1:, 4] = 210 / 250
    source = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    predictor = NativeBandAcousticPredictor(codec(saved))
    assert torch.equal(rng, torch.get_rng_state())
    values = predictor.scalers.channels(x, mask)
    with torch.no_grad():
        expected = model.forecast(
            torch.from_numpy(values),
            torch.from_numpy(mask),
            torch.from_numpy(metadata),
            torch.from_numpy(query),
        ).numpy()
    expected = (
        expected * np.array(saved["scalers"]["target_std"])[None, :, None]
        + np.array(saved["scalers"]["target_mean"])[None, :, None]
    )
    actual = predictor.forecast(x, mask, metadata, query)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-6)
    assert actual.shape == (5, 3, 5) and (np.diff(actual, axis=-1) >= 0).all()
    np.testing.assert_allclose(
        predictor.encode(x, mask, metadata),
        np.concatenate(
            [predictor.encode(x[i : i + 1], mask[i : i + 1], metadata[i : i + 1]) for i in range(5)]
        ),
        atol=2e-6,
        rtol=2e-6,
    )
    assert metadata[0, 0, 4] == np.float32(230 / 250)
    assert query[0, 0, 4] == np.float32(230 / 250)
    assert all(torch.equal(v, predictor.model.state_dict()[k]) for k, v in source.items())
    assert all(not p.requires_grad and p.grad is None for p in predictor.model.parameters())
    with pytest.raises(TypeError):
        predictor.forecast(x, mask, metadata, query, future=np.zeros((5, 3, 4, 4)))
    with pytest.raises(TypeError):
        predictor.encode(x, mask, metadata, query=query)


@pytest.mark.parametrize(
    "change",
    [
        "shape",
        "future_suffix",
        "mask_dtype",
        "missing_primary",
        "observed_nan",
        "metadata_nan",
        "frequency",
        "interval",
        "geometry",
        "offset",
        "orientation",
    ],
)
def test_bad_native_inputs_are_rejected(change):
    saved, _ = artifact(selected=True)
    encoder = NativeBandAcousticEncoder(codec(saved))
    x, mask, meta, _ = synthetic_inputs()
    if change == "shape":
        x, mask = x[:, :, :3], mask[:, :, :3]
    elif change == "future_suffix":
        x, mask = np.concatenate([x, x[:, :4]], 1), np.concatenate([mask, mask[:, :4]], 1)
    elif change == "mask_dtype":
        mask = mask.astype(np.int8)
    elif change == "missing_primary":
        mask[0, 0, 0] = False
    elif change == "observed_nan":
        x[0, 0, 0] = np.nan
    elif change == "metadata_nan":
        meta[0, 0, 0] = np.nan
    else:
        index, value = {
            "frequency": (0, 1),
            "interval": (1, 0.5),
            "geometry": (4, 0),
            "offset": (9, 3),
            "orientation": (5, 1),
        }[change]
        meta[0, 0, index] = value
    with pytest.raises(ValueError):
        encoder.encode(x, mask, meta)


@pytest.mark.parametrize(
    "change",
    [
        "legacy_kind",
        "supervised_kind",
        "wrong_architecture",
        "missing_architecture",
        "cf",
        "missing_tensor",
        "dtype",
        "nonfinite",
        "bad_scaler",
        "missing_scaler",
        "unsafe_bindings",
        "tiny_without_guard",
    ],
)
def test_unsafe_or_incompatible_artifacts_rejected(change):
    saved, _ = artifact(selected=True)
    saved = copy.deepcopy(saved)
    if change == "legacy_kind":
        saved["kind"] = "native_ssl_selected_encoder_v1"
    elif change == "supervised_kind":
        saved["kind"] = "native_band_replication_downstream_supervised_encoder_v2"
    elif change == "wrong_architecture":
        saved["config"]["architecture"] = "legacy"
    elif change == "missing_architecture":
        del saved["architecture"]
    elif change == "cf":
        saved["config"]["method"] = "cf_jepa"
    elif change == "missing_tensor":
        del saved["encoder"]["patch_projection.weight"]
    elif change == "dtype":
        saved["encoder"]["patch_projection.weight"] = saved["encoder"][
            "patch_projection.weight"
        ].double()
    elif change == "nonfinite":
        saved["encoder"]["patch_projection.weight"][0, 0] = np.nan
    elif change == "bad_scaler":
        saved["scalers"]["channel_std"][0] = 0
    elif change == "missing_scaler":
        del saved["scalers"]["target_std"]
    elif change == "unsafe_bindings":
        saved["bindings"] = {}
    else:
        saved["correctness_smoke"] = False
    with pytest.raises(ValueError):
        NativeBandAcousticEncoder(codec(saved))


def test_legacy_loaders_reject_band_kinds_and_safe_load_rejects_pickle_objects():
    selected, _ = artifact(selected=True)
    full, _ = artifact()
    with pytest.raises(ValueError):
        NativeAcousticEncoder(codec(selected))
    with pytest.raises(ValueError):
        NativeAcousticPredictor(codec(full))
    full["kind"] = "native_ssl_weights_only_inference_v1"
    with pytest.raises(ValueError):
        NativeBandAcousticPredictor(codec(full))
    selected["forbidden_pickle"] = object()
    with pytest.raises(pickle.UnpicklingError):
        NativeBandAcousticEncoder(codec(selected))


def test_supervised_downstream_forecast_label_and_ancestry_rejection():
    from marine_echo.training.native_band_replication_downstream import (
        DownstreamConfig,
        load_inference,
    )

    saved, _ = artifact(method="direct")
    cfg = DownstreamConfig(
        mode="direct_end_to_end", method="direct", seed=13, updates=2, cadence=1, batch_size=2
    )
    saved["downstream_config"] = cfg.to_dict()
    saved["supervised_ancestry"] = {
        "mode": "direct_end_to_end",
        "ssl_only": False,
        "supervised_updates": 2,
        "selected_supervised_step": 1,
        "ancestor_encoder_sha256": None,
        "ancestor_run_sha256": None,
    }
    predictor = load_inference(codec(saved))
    assert predictor.training_kind == "supervised_downstream_direct_end_to_end"
    assert predictor.feature_training_kind == "supervised_feature_encoder"
    x, mask, meta, query = synthetic_inputs()
    assert predictor.forecast(x, mask, meta, query).shape == (3, 3, 5)
    saved["supervised_ancestry"]["ssl_only"] = True
    with pytest.raises(ValueError, match="ancestry"):
        load_inference(codec(saved))


@pytest.mark.parametrize("field", [0, 1, 4, 9])
def test_forecast_rejects_native_query_mismatch(field):
    saved, _ = artifact()
    predictor = NativeBandAcousticPredictor(codec(saved))
    x, mask, meta, query = synthetic_inputs()
    query[0, 0, field] += 0.1
    with pytest.raises(ValueError, match="query"):
        predictor.forecast(x, mask, meta, query)
