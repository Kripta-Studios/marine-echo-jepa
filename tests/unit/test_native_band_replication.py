"""SYNTHETIC_CORRECTNESS_ONLY finite-seed replication admission."""

import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUILDER / "evidence/ssl-band-builder-v1"))
import ast
import inspect
from dataclasses import replace

import band_test_support  # noqa: F401
import numpy as np
import torch

from marine_echo.training import native_band_replication_downstream as downstream
from marine_echo.training import native_band_replication_ssl as replication
from marine_echo.training import native_band_ssl as original
from marine_echo.training.native_band_replication_ssl import Config


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_fixed_replication_seeds_admitted(seed):
    Config(seed=seed).validate()
    downstream.DownstreamConfig(seed=seed).validate()
    downstream.DownstreamConfig(method="direct", mode="direct_end_to_end", seed=seed).validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 0),
        ("seed", 17),
        ("seed", True),
        ("history", 24),
        ("width", 128),
        ("latent", 128),
        ("blocks", 5),
        ("heads", 8),
        ("lr", 0.001),
        ("sigreg_weight", 0.04),
        ("pretrain_updates", 5999),
        ("readout_updates", 2000),
        ("method", "cf_jepa"),
    ],
)
def test_unadmitted_seed_or_variant_never_changes_fixed_recipe(field, value):
    with pytest.raises(ValueError):
        replace(Config(seed=13), **{field: value}).validate()


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_actual_tensor_initialization_head_sampler_and_schedule_equal_original(seed):
    dims = {"width": 16, "latent": 8, "blocks": 2, "heads": 4}
    a = original.initialize_model(seed, **dims)
    b = replication.initialize_model(seed, **dims)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    for method in replication.METHODS:
        control = replication.initialize_model(seed, method=method, **dims)
        assert all(torch.equal(v, control.state_dict()[k]) for k, v in b.state_dict().items())
    for phase in ("pretrain", "readout"):
        for step in range(7):
            np.testing.assert_array_equal(
                original.batch_indices(np.arange(100), 64, seed, phase, step),
                replication.batch_indices(np.arange(100), 64, seed, phase, step),
            )
    assert ast.dump(
        ast.parse(inspect.getsource(replication.schedule)), include_attributes=False
    ) == ast.dump(ast.parse(inspect.getsource(original.schedule)), include_attributes=False)


def test_full_architecture_parameter_count_and_order_unchanged():
    a = original.initialize_model(7)
    b = replication.initialize_model(13)
    assert list(a.state_dict()) == list(b.state_dict())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    assert [(k, tuple(v.shape)) for k, v in a.state_dict().items()] == [
        (k, tuple(v.shape)) for k, v in b.state_dict().items()
    ]


def test_direct_core_config_is_fresh_actual_requested_seed_without_ancestor():
    config = downstream.DownstreamConfig(method="direct", mode="direct_end_to_end", seed=23)
    inputs = downstream.RunInputs(*([None] * 10))
    core = downstream._core_config(inputs, config, correctness_smoke=False, smoke_core_config=None)
    assert core.seed == 23 and core.method == "direct"


def test_shared_loss_actual_context_target_gradients_and_encoder_sigreg_replay():
    from band_test_support import synthetic_inputs

    x, mask, metadata, query = synthetic_inputs(n=4)
    context = torch.tensor(x, requires_grad=True)
    future = torch.tensor(x[:, :12].reshape(4, 3, 4, 4), requires_grad=True)
    arguments = (
        context,
        torch.from_numpy(mask),
        torch.from_numpy(metadata),
        future,
        torch.ones_like(future, dtype=torch.bool),
        torch.from_numpy(query),
    )
    a = original.initialize_model(13, width=16, latent=8, blocks=2, heads=4)
    b = replication.initialize_model(13, width=16, latent=8, blocks=2, heads=4)
    captured = []
    hook = b.regularizer.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0])
    )
    expected = a.shared_loss(*arguments)
    actual = b.shared_loss(*arguments)
    hook.remove()
    for key in ("loss", "prediction", "regularization"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert captured[0].shape == (16, 8) and captured[0].requires_grad
    actual["loss"].backward()
    assert context.grad.abs().sum() > 0 and future.grad.abs().sum() > 0
    assert b.encoder.patch_projection.weight.grad.abs().sum() > 0
    assert b.predictor.net[0].weight.grad.abs().sum() > 0
    assert not hasattr(b, "online") and not hasattr(b, "ema")
