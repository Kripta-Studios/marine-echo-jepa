"""SYNTHETIC_CORRECTNESS_ONLY CPU mechanism tests, no scientific weights."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-builder-v1"))
from band_test_support import synthetic_inputs

from marine_echo.models.native_temporal import SharedTemporalEncoder


def test_proposed_hypothesis_breaks_channel_cancellation_with_identical_tensors():
    # Before the candidate exists, use the actual legacy path to preserve a
    # meaningful failing mechanism assertion, rather than an import-only failure.
    if importlib.util.find_spec("marine_echo.models.native_band_temporal"):
        from marine_echo.models.native_band_temporal import NativeBandEncoder
    else:
        NativeBandEncoder = SharedTemporalEncoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        legacy = SharedTemporalEncoder(width=16, latent=8, blocks=2, heads=4).double().eval()
        torch.manual_seed(7)
        candidate = NativeBandEncoder(width=16, latent=8, blocks=2, heads=4).double().eval()
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name])
    x, _, metadata, _ = synthetic_inputs(history=96)
    x = torch.from_numpy((x + 78) / 3).double()
    metadata = torch.from_numpy(metadata).double()
    mask = torch.ones_like(x, dtype=torch.bool)
    opposite = x.clone()
    opposite[..., 0] += 2
    opposite[..., 1] -= 2
    with torch.no_grad():
        old_difference = (
            (legacy.encode(x, mask, metadata) - legacy.encode(opposite, mask, metadata)).abs().max()
        )
        new_difference = (
            (candidate.encode(x, mask, metadata) - candidate.encode(opposite, mask, metadata))
            .abs()
            .max()
        )
    assert old_difference < 1e-12
    assert new_difference > 1e-4, (
        "Per-channel frequency-conditioned features must survive opposing band changes."
    )


def small_model():
    from marine_echo.training.native_band_ssl import initialize_model

    return initialize_model(7, width=16, latent=8, blocks=2, heads=4)


def tensors(history=96):
    values = synthetic_inputs(history=history)
    return tuple(torch.from_numpy(v) for v in values)


def test_initialization_parameters_and_every_head_match_legacy_exactly():
    from marine_echo.training import native_band_ssl as band
    from marine_echo.training import native_ssl as legacy

    baseline = legacy.initialize_model(7)
    total = sum(p.numel() for p in baseline.parameters())
    encoder_total = sum(p.numel() for p in baseline.encoder.parameters())
    for method in band.METHODS:
        candidate = band.initialize_model(7, method=method)
        assert sum(p.numel() for p in candidate.parameters()) == total
        assert sum(p.numel() for p in candidate.encoder.parameters()) == encoder_total
        assert set(candidate.state_dict()) == set(baseline.state_dict())
        assert all(
            torch.equal(v, candidate.state_dict()[k]) for k, v in baseline.state_dict().items()
        )
    with pytest.raises(ValueError, match="CF"):
        band.initialize_model(7, method="cf_jepa")


@pytest.mark.parametrize("history", [3, 5, 95, 96, 99])
def test_patch_padding_masked_fills_empty_channels_and_empty_crops(history):
    model = small_model().eval()
    x, mask, meta, _ = tensors(history)
    mask[0, :, 1:] = False
    mask[1] = False
    mask[2, -1] = False
    dirty = x.clone()
    dirty[~mask] = float("nan")
    with torch.no_grad():
        first = model.encoder.encode(x, mask, meta)
        second = model.encoder.encode(dirty, mask, meta)
        tokens, valid = model.encoder.tokens(dirty, mask, meta)
    assert torch.equal(first, second)
    assert torch.equal(first[1], torch.zeros_like(first[1]))
    assert tokens.shape == (3, (history + 3) // 4, 16)
    assert torch.isfinite(tokens).all() and not valid[1].any()
    # Padding zeros and false masks must preserve exact features.
    padding = (-history) % 4
    xp = torch.nn.functional.pad(x, (0, 0, 0, padding))
    mp = torch.nn.functional.pad(mask, (0, 0, 0, padding))
    with torch.no_grad():
        assert torch.equal(first, model.encoder.encode(xp, mp, meta))


def test_native_metadata_query_and_temporal_order_change_actual_features():
    model = small_model().eval()
    x, mask, meta, query = tensors()
    shifted = meta.clone()
    shifted[:, 1, 0] += 0.1
    shifted[:, 1, 4] -= 0.1
    with torch.no_grad():
        base = model.encoder.encode(x, mask, meta)
        assert not torch.allclose(base, model.encoder.encode(x, mask, shifted))
        assert not torch.allclose(base, model.encoder.encode(x.flip(1), mask.flip(1), meta))
        prediction = model.forecast(x, mask, meta, query)
        new_query = query.clone()
        new_query[..., 4] = 200 / 250
        assert not torch.allclose(prediction, model.forecast(x, mask, meta, new_query))
    calls = []
    handles = [b.register_forward_hook(lambda *args: calls.append(1)) for b in model.encoder.blocks]
    model.encoder.encode(x, mask, meta)
    for handle in handles:
        handle.remove()
    assert len(calls) == 2  # genuine temporal blocks, not a flattened suffix head.
    assert torch.equal(meta[..., 4], torch.full_like(meta[..., 4], 230 / 250))


def test_shared_context_and_independent_target_gradients_and_encoded_sigreg():
    model = small_model()
    x, mask, meta, query = tensors()
    x.requires_grad_(True)
    future = x[:, :48].reshape(3, 3, 4, 4, 4)[:, :, 0].clone().detach().requires_grad_(True)
    fm = torch.ones_like(future, dtype=torch.bool)
    encoded = []
    regularized = []
    original = model.encoder.encode

    # A local hook on THIS candidate instance captures real latents; no legacy
    # module/factory is patched. Predictor changes must not enter SIGReg input.
    def capture(*args):
        value = original(*args)
        value.retain_grad()
        encoded.append(value)
        return value

    model.encoder.encode = capture
    hook = model.regularizer.register_forward_pre_hook(
        lambda module, args: regularized.append(args[0])
    )
    details = model.shared_loss(x, mask, meta, future, fm, query)
    expected = torch.cat([encoded[0], torch.stack(encoded[1:], 1).flatten(0, 1)])
    assert torch.equal(regularized[0], expected)
    assert details["loss"] == details["prediction"] + 0.03 * details["regularization"]
    context_reg, target_reg = torch.autograd.grad(
        details["regularization"], (x, future), retain_graph=True
    )
    assert context_reg.abs().sum() > 0 and target_reg.abs().sum() > 0
    details["loss"].backward()
    assert x.grad.abs().sum() > 0 and future.grad.abs().sum() > 0
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in encoded)
    assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
    assert model.regularizer.global_step.item() == 1
    hook.remove()
    assert not hasattr(model, "online")


def test_future_perturbation_never_changes_context_forecast_and_regularizer_replays():
    from band_test_support import codec

    from marine_echo.training.native_band_ssl import cpu_state

    model = small_model()
    x, mask, meta, query = tensors()
    future = x[:, :12].reshape(3, 3, 4, 4).clone()
    fm = torch.ones_like(future, dtype=torch.bool)
    before = model.forecast(x, mask, meta, query)
    state = torch.load(codec(cpu_state(model)), weights_only=True)
    first = model.shared_loss(x, mask, meta, future, fm, query)
    model.load_state_dict(state, strict=True)
    second = model.shared_loss(x, mask, meta, future, fm, query)
    assert torch.equal(first["regularization"], second["regularization"])
    assert torch.equal(before, model.forecast(x, mask, meta, query))
    model.load_state_dict(state, strict=True)
    changed = model.shared_loss(x, mask, meta, future + 5, fm, query)
    assert not torch.equal(first["prediction"], changed["prediction"])
    assert torch.equal(before, model.forecast(x, mask, meta, query))
