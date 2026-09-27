"""Validation-only protocol rescore of immutable AEON campaign artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training import aeon_campaign
from marine_echo.training.aeon_rescore import rescore_campaign
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _validation_row(index: int) -> AeonHourlyWindow:
    cutoff = np.datetime64("2024-10-08T00:00:00", "us") + index * np.timedelta64(1, "h")
    if index == 18:
        cutoff = np.datetime64("2024-10-10T00:00:00", "us")
    return AeonHourlyWindow(
        row_id=f"{index:064x}",
        partition="validation",
        cutoff_source_timestamp=cutoff,
        cutoff_interval_id=480120 + index,
        context_db=np.full((24, 4), -80.0),
        context_mask=np.ones((24, 4), dtype=bool),
        context_interval_ids=np.arange(480097, 480121) + index,
        context_source_timestamps=cutoff - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
        target_interval_ids=np.array([480121, 480123, 480126]) + index,
        target_source_timestamps=cutoff + np.array([1, 3, 6]) * np.timedelta64(1, "h"),
        target_db=np.array([100.0, 100.0, 100.0]) if index == 18 else np.full(3, -80.0),
        target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256="a" * 64,
        past_members=("validation.csv",),
        target_members=("validation.csv",),
    )


def _completed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    config_path = Path(__file__).resolve().parents[2] / "configs/aeon_campaign.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    code_sha256 = "b" * 64
    monkeypatch.setattr(aeon_campaign, "_campaign_code_sha256", lambda: code_sha256)
    rows = [_validation_row(index) for index in range(19)]

    def stub(slot, fit, assess, config_arg, stage, device):
        model = stage / "model.bin"
        model.write_bytes(slot.run_id.encode("ascii"))
        forecast = np.broadcast_to(
            np.array([-90.0, -85.0, -80.0, -75.0, -70.0]), (len(assess), 3, 5)
        ).copy()
        return forecast, model, {"updates": 0}

    campaign = tmp_path / "campaign"
    aeon_campaign.execute_campaign_slots(
        [], rows, campaign, config=config, config_sha256=config_sha256,
        cohort_sha256="c" * 64, review_sha256="d" * 64, device="cpu", executor=stub,
    )
    correction = tmp_path / "correction.json"
    correction.write_text(
        json.dumps({
            "disposition": "APPROVE_EXECUTION_ONLY_SELECTION_REQUIRES_PROTOCOL_RESCORE",
            "campaign_code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "test_access": "PROHIBITED",
        }), encoding="utf-8"
    )
    return campaign, config_path, correction


def test_rescore_all_slots_with_exact_eligible_date_and_old_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, config, correction = _completed_fixture(tmp_path, monkeypatch)
    output = tmp_path / "rescore"
    result = rescore_campaign(campaign, config, correction, output)
    assert len(result["slots"]) == 17
    assert result["model_selection"] == "NOT_PERFORMED_PENDING_INDEPENDENT_REVIEW"
    assert result["eligible_source_dates_by_horizon"] == [["2024-10-08"]] * 3
    expected_hash = hashlib.sha256(b"2024-10-08\n").hexdigest()
    assert result["eligible_source_date_sha256_by_horizon"] == [expected_hash] * 3
    for slot in result["slots"].values():
        metrics = slot["protocol_validation_metrics"]
        assert metrics["eligible_days_per_horizon"] == [1, 1, 1]
        assert metrics["eligible_scored_rows_per_horizon"] == [18, 18, 18]
        assert len(metrics["daily_pinball_db_by_horizon_date_quantile"][0]) == 1
        assert slot["original_all_scored_day_metric_status"] == "NON_PROTOCOL_DIAGNOSTIC"
        assert slot["original_all_scored_day_primary_pinball_db"] > metrics[
            "primary_daily_mean_pinball_db"
        ]
    assert rescore_campaign(campaign, config, correction, output) == result


def test_rescore_rejects_tampered_saved_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, config, correction = _completed_fixture(tmp_path, monkeypatch)
    prediction = campaign / "direct_seed7" / "validation-predictions.npz"
    prediction.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact digest"):
        rescore_campaign(campaign, config, correction, tmp_path / "rescore")


def test_rescore_rejects_missing_slot_without_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, config, correction = _completed_fixture(tmp_path, monkeypatch)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slots"].pop("direct_seed7")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="all 17 completed"):
        rescore_campaign(campaign, config, correction, tmp_path / "rescore")


def test_rescore_rejects_manifest_metric_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign, config, correction = _completed_fixture(tmp_path, monkeypatch)
    manifest_path = campaign / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["slots"]["direct_seed7"]["primary_daily_mean_pinball_db"] = 123.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="original metric"):
        rescore_campaign(campaign, config, correction, tmp_path / "rescore")
