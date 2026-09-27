"""Finite campaign scheduling and provenance contracts without training."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
    assert len(slots) == 15
    assert [slot.run_id for slot in slots[:4]] == list(aeon_campaign.CONVENTIONAL)
    assert slots[-1].run_id == "temporally_shuffled_pretrain_target_seed7"
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
    assert len(calls) == 15
    assert len(result["slots"]) == 15
    aeon_campaign.execute_campaign_slots([], assess, output, **args)
    assert len(calls) == 15
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
