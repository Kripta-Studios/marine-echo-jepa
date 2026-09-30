"""SYNTHETIC_CORRECTNESS_ONLY: CPU pooling/objective parity, no optimizer/GPU."""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
spec = importlib.util.spec_from_file_location(
    "native_cf_pooling_candidate", BUILDER / "src/marine_echo/models/native_temporal.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    enabled = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warning)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "length,output", [(1, 1), (1, 8), (3, 8), (5, 2), (7, 3), (9, 8), (11, 17), (17, 5), (31, 8)]
)
@pytest.mark.parametrize("batched", [False, True])
def test_exact_bins_values_and_weighted_gradients(dtype, length, output, batched):
    generator = torch.Generator().manual_seed(31)
    shape = (2, 3, length) if batched else (3, length)
    source = torch.randn(shape, generator=generator, dtype=dtype)
    a, b = source.clone().requires_grad_(), source.clone().requires_grad_()
    actual = model.deterministic_adaptive_avg_pool1d(a, output)
    expected = F.adaptive_avg_pool1d(b, output)
    assert actual.shape == (*shape[:-1], output)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weights = torch.randn(actual.shape, generator=generator, dtype=dtype)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


def test_overlapping_upsampling_bins_noncontiguous_input_and_autograd():
    base = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3)
    a = base.transpose(1, 2).detach().requires_grad_()
    b = a.detach().clone().requires_grad_()
    # L5/O3 intervals are [0,2), [1,4), [3,5), with overlap.
    output = model.deterministic_adaptive_avg_pool1d(a, 3)
    torch.testing.assert_close(
        output,
        torch.stack([a[..., :2].mean(-1), a[..., 1:4].mean(-1), a[..., 3:].mean(-1)], -1),
        rtol=0,
        atol=0,
    )
    weights = torch.tensor([1.0, 2.0, 4.0])
    (output * weights).sum().backward()
    (F.adaptive_avg_pool1d(b, 3) * weights).sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-15)
    assert torch.autograd.gradcheck(lambda x: model.deterministic_adaptive_avg_pool1d(x, 7), (a,))


@pytest.mark.parametrize(
    "shape,output",
    [((2, 3, 0), 2), ((2, 3), 0), ((2, 3), -1), ((2, 3), True), ((2, 3), 2.5), ((3,), 2)],
)
def test_invalid_pooling_requests_fail_explicitly(shape, output):
    with pytest.raises(ValueError):
        model.deterministic_adaptive_avg_pool1d(torch.zeros(shape), output)


def toy_model_and_inputs(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        instance = model.CFNativeModel(width=12, latent=8, blocks=1).to(dtype=dtype)
        x = torch.randn(4, 96, 4, dtype=dtype)
        future = torch.randn(4, 3, 4, 4, dtype=dtype)
    metadata = torch.zeros(4, 4, 10, dtype=dtype)
    metadata[:, :, 3:5] = torch.tensor([0.04, 0.92], dtype=dtype)
    query = metadata[:, :1].repeat(1, 3, 1)
    query[:, :, -1] = torch.tensor([1, 3, 6], dtype=dtype)
    return instance, (
        x,
        torch.ones_like(x, dtype=torch.bool),
        metadata,
        future,
        torch.ones_like(future, dtype=torch.bool),
        query,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cf_loss_all_pooling_sites_and_gradient_parity(monkeypatch, dtype):
    candidate, inputs = toy_model_and_inputs(dtype)
    reference = copy.deepcopy(candidate)
    utility = model.deterministic_adaptive_avg_pool1d
    original_pool = F.adaptive_avg_pool1d
    calls = []

    def traced(x, size):
        calls.append((x.shape[-1], size))
        return utility(x, size)

    monkeypatch.setattr(model, "deterministic_adaptive_avg_pool1d", traced)
    monkeypatch.setattr(
        F,
        "adaptive_avg_pool1d",
        lambda *a, **k: pytest.fail("CF reached rejected native adaptive CUDA pooling path."),
    )
    np.random.seed(23)
    actual = candidate.cf_loss(*inputs, step=0, total=10)
    actual.backward()
    rng = np.random.get_state()
    # Four variance pools and four pools for each of 2/4/8 invariance,
    # plus prediction resizing. Other output sizes prove that path executed.
    assert sum(size == 8 for _, size in calls) >= 8
    assert any(size not in (2, 4, 8) for _, size in calls)
    assert all(p.grad is None for p in candidate.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.online.parameters())
    monkeypatch.setattr(model, "deterministic_adaptive_avg_pool1d", original_pool)
    np.random.seed(23)
    expected = reference.cf_loss(*inputs, step=0, total=10)
    expected.backward()
    reference_rng = np.random.get_state()
    np.testing.assert_array_equal(rng[1], reference_rng[1])
    assert rng[2:] == reference_rng[2:]
    tolerance = 2e-5 if dtype == torch.float32 else 5e-12
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for (name, a), (_, b) in zip(
        candidate.named_parameters(), reference.named_parameters(), strict=True
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance, msg=name)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_cf_loss_backward_replay_is_exact():
    initial, inputs = toy_model_and_inputs(torch.float32)
    first, second = copy.deepcopy(initial), copy.deepcopy(initial)
    results = []
    for instance in (first, second):
        np.random.seed(13)
        loss = instance.cf_loss(*inputs, step=2, total=10)
        loss.backward()
        results.append(loss)
    assert torch.equal(*results)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


def test_cf_loss_never_calls_native_adaptive_pool(monkeypatch):
    candidate, inputs = toy_model_and_inputs(torch.float32)
    monkeypatch.setattr(
        F,
        "adaptive_avg_pool1d",
        lambda *a, **k: pytest.fail("CF invokes the CUDA-rejected adaptive pooling operation."),
    )
    np.random.seed(23)
    loss = candidate.cf_loss(*inputs, step=0, total=10)
    loss.backward()
    assert torch.isfinite(loss)


def test_pooling_autograd_graph_uses_means_slices_and_stack_only(monkeypatch):
    monkeypatch.setattr(
        F,
        "adaptive_avg_pool1d",
        lambda *a, **k: pytest.fail("Utility reached native adaptive pooling."),
    )
    values = torch.randn(2, 3, 11, requires_grad=True)
    pooled = model.deterministic_adaptive_avg_pool1d(values, 7)
    names, pending, seen = set(), [pooled.grad_fn], set()
    while pending:
        operation = pending.pop()
        if operation is None or operation in seen:
            continue
        seen.add(operation)
        names.add(type(operation).__name__)
        pending.extend(child for child, _ in operation.next_functions)
    assert any(name.startswith("MeanBackward") for name in names)
    assert any(name.startswith("SliceBackward") for name in names)
    assert any(name.startswith("StackBackward") for name in names)
    assert not any("Pool" in name for name in names)
    pooled.square().sum().backward()
    assert torch.isfinite(values.grad).all()
