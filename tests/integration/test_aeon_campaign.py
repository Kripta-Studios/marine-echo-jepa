"""Finite campaign scheduling and provenance contracts without training."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.models.aeon_ssl import AeonTemporalSSL
from marine_echo.training import aeon_campaign
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _row() -> AeonHourlyWindow:
    cutoff = np.datetime64("2024-10-08T23:52", "us")
    return AeonHourlyWindow(
        row_id="f" * 64,
        partition="validation",
        cutoff_source_timestamp=cutoff,
        cutoff_interval_id=480120,
        context_db=np.full((24, 4), -80.0),
        context_mask=np.ones((24, 4), dtype=bool),
        context_interval_ids=np.arange(480097, 480121),
        context_source_timestamps=cutoff - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
        target_interval_ids=np.array([480121, 480123, 480126]),
        target_source_timestamps=cutoff + np.array([1, 3, 6]) * np.timedelta64(1, "h"),
        target_db=np.array([-79.0, -78.0, -77.0]),
        target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256="a" * 64,
        past_members=("train.csv",),
        target_members=("validation.csv",),
    )


def _config() -> dict:
    path = Path(__file__).resolve().parents[2] / "configs/aeon_campaign.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_campaign_has_finite_slot_order_and_test_access_gate() -> None:
    config = _config()
    aeon_campaign._config_gate(config)
    slots = aeon_campaign.campaign_slots(config)
    assert len(slots) == 17
    assert [slot.run_id for slot in slots[:4]] == list(aeon_campaign.CONVENTIONAL)
    assert slots[-1].run_id == "temporally_shuffled_pretrain_target_shared_sigreg_seed7"
    config["test_access"] = "ALLOWED"
    with pytest.raises(ValueError, match="source/split"):
        aeon_campaign._config_gate(config)


def test_campaign_manifest_restarts_and_detects_artifact_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aeon_campaign, "_campaign_code_sha256", lambda: "b" * 64)
    config = _config()
    assess = [_row()]
    calls: list[str] = []

    def stub(slot, fit, assess_rows, config_arg, stage, device):
        calls.append(slot.run_id)
        model = stage / "model.bin"
        model.write_bytes(slot.run_id.encode())
        prediction = np.full((len(assess_rows), 3, 5), -78.0)
        return prediction, model, {"updates": 0}

    output = tmp_path / "campaign"
    args = dict(
        config=config,
        config_sha256="c" * 64,
        cohort_sha256="d" * 64,
        review_sha256="e" * 64,
        device="cpu",
        executor=stub,
    )
    result = aeon_campaign.execute_campaign_slots([], assess, output, **args)
    assert result["status"] == "COMPLETED_TRAIN_VALIDATION_CAMPAIGN_NOT_FINAL_EVALUATION"
    assert len(calls) == 17
    assert len(result["slots"]) == 17
    aeon_campaign.execute_campaign_slots([], assess, output, **args)
    assert len(calls) == 17
    (output / "direct_seed7/model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact digest"):
        aeon_campaign.execute_campaign_slots([], assess, output, **args)


def test_seasonal_24_step_reference_uses_matching_hour_for_each_horizon(tmp_path: Path) -> None:
    base = _row()
    fit_context = np.full((24, 4), -80.0)
    fit_context[:, 0] = np.arange(24)
    assess_context = fit_context + 100
    fit = [replace(base, context_db=fit_context, target_db=np.array([1.0, 3.0, 6.0]))]
    assess = [replace(base, context_db=assess_context)]
    slot = aeon_campaign.CampaignSlot(
        "seasonal_24_source_intervals", "seasonal_24_source_intervals", None
    )
    forecast, model_path, _ = aeon_campaign._conventional_prediction(
        slot, fit, assess, _config(), tmp_path
    )
    assert model_path.exists()
    assert np.array_equal(forecast[0, :, 2], [101.0, 103.0, 106.0])


def test_shuffled_control_preserves_mode_specific_target_gradient() -> None:
    for mode in ("ema", "shared_sigreg"):
        torch.manual_seed(7)
        model = AeonTemporalSSL(mode=mode, width=16, layers=2, regularizer_weight=0)
        values = torch.randn(8, 24, 4, requires_grad=True)
        mask = torch.ones_like(values, dtype=torch.bool)
        aeon_campaign._shuffled_pretrain_loss(model, values, mask, shift=1).backward()
        assert values.grad is not None
        held_out_gradient = values.grad[:, 18:].abs().sum().item()
        if mode == "ema":
            assert held_out_gradient == 0
        else:
            assert held_out_gradient > 0


def test_all_neural_campaign_families_execute_bounded_fixture(tmp_path: Path) -> None:
    base = _row()
    fit = [
        replace(
            base,
            row_id=f"{index:064x}",
            partition="train",
            cutoff_interval_id=480120 + 24 * index,
            context_db=np.full((24, 4), -80.0 + index),
            target_db=np.array([-79.0, -78.0, -77.0]) + index,
        )
        for index in range(8)
    ]
    config = _config()
    config["neural"].update(
        {
            "encoder_width": 16,
            "encoder_layers": 2,
            "batch_size": 8,
            "direct_supervised_updates": 1,
            "ssl_pretrain_updates": 1,
            "ssl_supervised_updates": 1,
            "checkpoint_every_updates": 1,
        }
    )
    config["controls"].update(
        {"random_encoder_supervised_updates": 1, "shuffled_pretrain_updates": 1}
    )
    for family in ("direct", "ema_jepa", "shared_sigreg") + aeon_campaign.CONTROLS:
        stage = tmp_path / family
        stage.mkdir()
        slot = aeon_campaign.CampaignSlot(family + "_seed7", family, 7)
        forecast, checkpoint, details = aeon_campaign._neural_prediction(
            slot, fit, [base], config, stage, "cpu"
        )
        assert forecast.shape == (1, 3, 5)
        assert np.isfinite(forecast).all()
        assert checkpoint.exists()
        assert len(details["validation_checks"]) == 1
        assert details["supervised_updates"] == 1
        expected_pretrain = 0 if family == "direct" or family.startswith("random_encoder") else 1
        assert details["pretrain_updates"] == expected_pretrain
        assert details["random_encoder_unequal_total_learned_update_budget"] == family.startswith(
            "random_encoder"
        )
