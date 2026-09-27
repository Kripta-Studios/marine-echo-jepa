"""Focused gradient and causal-view checks for the AEON hourly SSL models."""

from __future__ import annotations

import torch

from marine_echo.models.aeon_ssl import AeonDirect, AeonTemporalSSL


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.randn(8, 24, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 3, 2] = False
    return values, mask


def test_ema_teacher_has_no_gradient_and_updates_only_after_explicit_step() -> None:
    values, mask = _batch()
    model = AeonTemporalSSL(mode="ema", width=128, layers=3)
    before = [parameter.detach().clone() for parameter in model.teacher.parameters()]
    loss = model.pretrain_loss(values, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert all(torch.equal(x, y) for x, y in zip(before, model.teacher.parameters()))
    with torch.no_grad():
        next(model.encoder.parameters()).add_(0.25)
    model.update_teacher(momentum=0.9)
    assert not torch.equal(before[0], next(model.teacher.parameters()))


def test_shared_sigreg_propagates_gradient_to_shared_encoder() -> None:
    values, mask = _batch()
    model = AeonTemporalSSL(mode="shared_sigreg", width=128, layers=3)
    loss = model.pretrain_loss(values, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert (
        sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None) > 0
    )


def test_shared_predictive_target_keeps_gradient_through_held_out_past() -> None:
    for mode in ("ema", "shared_sigreg"):
        values, mask = _batch()
        values.requires_grad_(True)
        model = AeonTemporalSSL(mode=mode, width=128, layers=3, regularizer_weight=0)
        model.pretrain_loss(values, mask).backward()
        assert values.grad is not None
        target_gradient = values.grad[:, 18:].abs().sum().item()
        if mode == "ema":
            assert target_gradient == 0
        else:
            assert target_gradient > 0


def test_masked_context_view_cannot_read_later_six_products() -> None:
    values, mask = _batch()
    model = AeonTemporalSSL(mode="ema", width=128, layers=3)
    a = model.context_view(values, mask)
    changed = values.clone()
    changed[:, 18:] += 10_000
    b = model.context_view(changed, mask)
    assert torch.equal(a, b)
    assert not torch.equal(model.teacher_view(values, mask), model.teacher_view(changed, mask))


def test_teacher_view_encodes_only_held_out_six_past_products() -> None:
    values, mask = _batch()
    for mode in ("ema", "shared_sigreg"):
        model = AeonTemporalSSL(mode=mode, width=128, layers=3)
        target = model.teacher_view(values, mask)
        changed_context = values.clone()
        changed_context[:, :18] += 10_000
        assert torch.equal(target, model.teacher_view(changed_context, mask))
        changed_target = values.clone()
        changed_target[:, 18:] += 10_000
        assert not torch.equal(target, model.teacher_view(changed_target, mask))


def test_supervised_heads_have_identical_output_contract() -> None:
    values, mask = _batch()
    direct = AeonDirect(width=128, layers=3)
    ssl = AeonTemporalSSL(mode="ema", width=128, layers=3)
    for model in (direct, ssl):
        forecast = model(values, mask)
        assert forecast.shape == (8, 3, 5)
        assert torch.isfinite(forecast).all()
        assert torch.all(torch.diff(forecast, dim=-1) >= 0)
