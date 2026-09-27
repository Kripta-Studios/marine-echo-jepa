"""Past-only masked patch transformer behavior and fail-closed run contract."""

import json
from pathlib import Path

import pytest
import torch

from marine_echo.models.aeon_patchtst import AeonMaskedPatchTransformer
from marine_echo.training.aeon_patchtst import _gate

CONFIG = Path(__file__).resolve().parents[2] / "configs/aeon_patchtst_exploratory.json"


def test_shape_monotonicity_and_masked_value_invariance() -> None:
    torch.manual_seed(7)
    model = AeonMaskedPatchTransformer(width=32, layers=1, heads=4, dropout=0)
    model.eval()
    values = torch.randn(2, 24, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 3:8, 2] = False
    with torch.no_grad():
        first = model(values, mask)
        changed = values.clone()
        changed[0, 3:8, 2] = float("nan")
        second = model(changed, mask)
    assert first.shape == (2, 3, 5)
    assert torch.isfinite(first).all()
    assert torch.all(first[..., 1:] >= first[..., :-1])
    torch.testing.assert_close(first, second)


def test_shared_channel_encoder_is_independent_before_head() -> None:
    torch.manual_seed(7)
    model = AeonMaskedPatchTransformer(width=32, layers=1, heads=4, dropout=0)
    model.eval()
    values = torch.randn(2, 24, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    with torch.no_grad():
        first = model.encode_channels(values, mask)
        values[:, :, 1:] += 3.0
        second = model.encode_channels(values, mask)
    torch.testing.assert_close(first[:, 0], second[:, 0])
    assert not torch.allclose(first[:, 1:], second[:, 1:])


def test_quantile_loss_propagates_gradient_to_patch_projection() -> None:
    model = AeonMaskedPatchTransformer(width=32, layers=1, heads=4, dropout=0)
    values = torch.randn(4, 24, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    prediction = model(values, mask)
    prediction.mean().backward()
    assert model.patch_projection.weight.grad is not None
    assert model.patch_projection.weight.grad.abs().sum() > 0


def test_exploratory_gate_rejects_partition_or_cohort_change() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _gate(config, config["cohort_sha256"], config["split_review_sha256"])
    config["assessment_partition"] = "test"
    with pytest.raises(ValueError, match="contract"):
        _gate(config, config["cohort_sha256"], config["split_review_sha256"])
    config["assessment_partition"] = "validation"
    with pytest.raises(ValueError, match="contract"):
        _gate(config, "0" * 64, config["split_review_sha256"])
