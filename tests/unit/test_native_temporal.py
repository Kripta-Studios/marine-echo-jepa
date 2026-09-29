"""Synthetic CPU correctness tests; no scientific fitting or metrics."""

import copy

import pytest
import torch

from marine_echo.models.native_temporal import NativeTemporalModel, SharedTemporalEncoder


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def inputs(length=9, batch=4):
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(batch, length, 4, generator=generator)
    observed = torch.rand(batch, length, 4, generator=generator) > 0.2
    metadata = torch.zeros(batch, 4, 10)
    metadata[:, :, 0] = torch.tensor([38000, 125000, 200000, 455000]) / 455000
    metadata[:, :, 1:3] = 1
    metadata[:, :, 4] = 230 / 250
    query = metadata[:, :1].expand(-1, 3, -1).clone()
    query[:, :, -1] = torch.tensor([1, 3, 6])
    return x, observed, metadata, query


def small():
    return NativeTemporalModel(width=24, latent=8, blocks=2, heads=4)


def test_masked_fill_padding_and_all_missing():
    x, mask, meta, _ = inputs()
    encoder = SharedTemporalEncoder(width=24, latent=8, blocks=2, heads=4).eval()
    altered = x.masked_fill(~mask, float("nan"))
    expected = encoder.encode(x, mask, meta)
    torch.testing.assert_close(expected, encoder.encode(altered, mask, meta))
    padded = torch.cat([x, torch.full((4, 3, 4), 1e9)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(4, 3, 4, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(expected, encoder.encode(padded, padded_mask, meta))
    empty = encoder.encode(altered, torch.zeros_like(mask), meta)
    assert torch.isfinite(empty).all() and torch.count_nonzero(empty) == 0


def test_native_bounds_condition_both_encoder_and_query():
    x, mask, meta, query = inputs()
    model = small().eval()
    base = model.forecast(x, mask, meta, query)
    changed = meta.clone()
    changed[:, :, 4] = 200 / 250
    assert not torch.equal(
        model.encoder.encode(x, mask, meta), model.encoder.encode(x, mask, changed)
    )
    other_query = query.clone()
    other_query[:, :, 4] = 220 / 250
    assert not torch.equal(base, model.forecast(x, mask, meta, other_query))
    assert torch.equal(query[:, :, 4], torch.full((4, 3), 230 / 250))
    assert base.shape == (4, 3, 5)
    assert (base.diff(dim=-1) >= 0).all()


def test_shared_target_gradients_and_encoded_regularizer():
    x, mask, meta, query = inputs()
    model = small()
    future = torch.randn(4, 3, 4, 4, requires_grad=True)
    x = x.requires_grad_()
    details = model.shared_loss(
        x, mask, meta, future, torch.ones_like(future, dtype=torch.bool), query
    )
    details["prediction"].backward(retain_graph=True)
    assert x.grad.abs().sum() > 0 and future.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    details["regularization"].backward()
    assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.predictor.parameters())
    assert int(model.regularizer.global_step) == 1
    cloned = small()
    cloned.load_state_dict(model.state_dict())
    assert torch.equal(cloned.regularizer.global_step, model.regularizer.global_step)


def test_future_values_cannot_change_issued_forecast():
    x, mask, meta, query = inputs()
    model = small().eval()
    before = model.forecast(x, mask, meta, query)
    future = torch.randn(4, 3, 4, 4)
    model.shared_loss(x, mask, meta, future, torch.ones_like(future, dtype=torch.bool), query)
    torch.testing.assert_close(before, model.forecast(x, mask, meta, query), rtol=0, atol=0)


def test_matched_initialization_including_every_head():
    from marine_echo.training.native_ssl import initialize_model

    reference = initialize_model(7, width=24, latent=8, blocks=2, heads=4)
    for method in ("shared_ssl", "masked_ssl", "direct", "random_frozen", "permuted_ssl"):
        candidate = initialize_model(7, width=24, latent=8, blocks=2, heads=4)
        for key, value in reference.state_dict().items():
            assert torch.equal(value, candidate.state_dict()[key]), (method, key)


def test_frozen_readout_preserves_encoder():
    x, mask, meta, query = inputs()
    model = small()
    model.freeze_encoder()
    before = copy.deepcopy(model.encoder.state_dict())
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=0.01)
    model.forecast(x, mask, meta, query).square().mean().backward()
    optimizer.step()
    assert not model.encoder.training
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())


def test_cf_ema_detachment_and_parameter_only_update():
    from marine_echo.models.native_temporal import CFNativeModel

    x, mask, meta, query = inputs(length=24)
    x = x.requires_grad_()
    future = torch.randn(4, 3, 4, 4, requires_grad=True)
    model = CFNativeModel(width=24, latent=8, blocks=2)
    initial = copy.deepcopy(model.encoder.state_dict())
    loss = model.cf_loss(
        x, mask, meta, future, torch.ones_like(future, dtype=torch.bool), query, step=0, total=4
    )
    loss.backward()
    assert torch.isfinite(loss) and x.grad.abs().sum() > 0
    # TRAIN role can shift into online crops. The target encoder itself remains
    # detached, while the final three trajectory intervals never enter a crop.
    assert future.grad is None or torch.count_nonzero(future.grad[:, 2, 1:]) == 0
    assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    assert model.online.input_fc.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.online.parameters(), lr=0.01)
    optimizer.step()
    model.update_ema(0, 4)
    assert not torch.equal(initial["input_fc.weight"], model.encoder.input_fc.weight)
    for key, value in initial.items():
        if "running_" in key or "num_batches_tracked" in key:
            assert torch.equal(value, model.encoder.state_dict()[key])


def test_masked_ssl_trains_shared_latent_projection_without_predictor():
    x, observed, meta, _ = inputs()
    model = small()
    loss = model.masked_loss(x, observed, meta, generator=torch.Generator().manual_seed(8))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.output[-1].weight.grad.abs().sum() > 0
    assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.predictor.parameters())


def test_cf_and_shared_readout_initial_tensors_match():
    from marine_echo.training.native_ssl import initialize_model

    shared = initialize_model(7, width=24, latent=8, blocks=2)
    cf = initialize_model(7, width=24, latent=8, blocks=2, method="cf_jepa")
    assert all(
        torch.equal(value, cf.readout.state_dict()[key])
        for key, value in shared.readout.state_dict().items()
    )


def test_default_capacity_and_cpu_runtime_evidence():
    import platform

    from marine_echo.training.native_ssl import initialize_model

    model = initialize_model(7)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    assert 2_000_000 <= total <= 8_000_000
    assert not torch.cuda.is_initialized(), "CPU correctness must not acquire the GPU."
    print(
        f"SYNTHETIC_CORRECTNESS_ONLY Python={platform.python_version()} torch={torch.__version__} encoder_parameters={encoder} model_parameters={total}"
    )


def test_cf_encoder_matches_inspected_pinned_source_with_all_observations():
    import importlib.util

    from marine_echo.models.native_temporal import CFTemporalEncoder
    from marine_echo.training.native_ssl import MAIN

    path = MAIN / "external/cf-jepa-vnext/baselines/cf_jepa/encoder.py"
    spec = importlib.util.spec_from_file_location("pinned_cf_encoder_correctness", path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    reference = source.Encoder(
        input_dims=48, hidden=24, output=8, depth=2, strict_causal=False
    ).eval()
    adapted = CFTemporalEncoder(width=24, latent=8, blocks=2).eval()
    adapted.input_fc.load_state_dict(reference.input_fc.state_dict())
    adapted.output_fc.load_state_dict(reference.output_fc.state_dict())
    for actual, expected in zip(adapted.blocks, reference.blocks, strict=True):
        for a, e in zip(actual["dw"], expected.dw_convs, strict=True):
            a.load_state_dict(e.conv.state_dict())
        for key in ("bn1", "pw", "bn2"):
            actual[key].load_state_dict(getattr(expected, key).state_dict())
    x, _, metadata, _ = inputs(length=24)
    observed = torch.ones_like(x, dtype=torch.bool)
    adapter_input = torch.cat(
        [x, observed.float(), metadata.flatten(1).unsqueeze(1).expand(-1, 24, -1)], dim=-1
    )
    torch.testing.assert_close(
        adapted.sequence(x, observed, metadata), reference(adapter_input), atol=0, rtol=0
    )


def test_cf_default_source_dimensions_and_trajectory_zone_input():
    from marine_echo.models.native_temporal import CFNativeModel

    model = CFNativeModel()
    assert model.online.input_fc.out_features == 256
    assert model.online.output_fc.out_features == 128
    assert len(model.online.blocks) == 5
    x, observed, meta, query = inputs(length=24)
    model = CFNativeModel(width=24, latent=8, blocks=2)
    future = torch.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).float()
    captured = []
    # sequence is a reusable method; inspect its target call without altering tensors.
    original = model.encoder.sequence

    def capture(values, mask, metadata):
        captured.append(values.detach().clone())
        return original(values, mask, metadata)

    model.encoder.sequence = capture
    model.cf_loss(
        x, observed, meta, future, torch.ones_like(future, dtype=torch.bool), query, step=0, total=4
    )
    assert len(captured) == 1
    torch.testing.assert_close(
        captured[0],
        torch.cat([x, future[:, 0], future[:, 1, 2:4], future[:, 2, 1:4]], dim=1),
        rtol=0,
        atol=0,
    )
