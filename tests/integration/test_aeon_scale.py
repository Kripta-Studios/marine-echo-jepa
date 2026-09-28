"""New AEON scaling campaign contracts; never uses CAL or TEST."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training import aeon_scale
from marine_echo.training.aeon_development import _save_predictions
from marine_echo.training.aeon_windows import AeonHourlyWindow


def _config() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / "configs/aeon_scale_30k.json").read_text())


def test_scale_config_and_staged_slots_are_bounded() -> None:
    config = _config()
    aeon_scale.validate_config(config)
    assert [(slot.family, slot.seed) for slot in aeon_scale.stage_slots(1)] == [
        ("direct", 7), ("ema_jepa", 7)
    ]
    assert [(slot.family, slot.seed) for slot in aeon_scale.stage_slots(2)] == [
        ("direct", 13), ("ema_jepa", 13),
        ("direct", 23), ("ema_jepa", 23),
    ]
    config["test_access"] = "ALLOWED"
    with pytest.raises(ValueError, match="scale contract"):
        aeon_scale.validate_config(config)
    config = _config()
    config["neural"]["direct_supervised_updates"] = 50000
    with pytest.raises(ValueError, match="scale contract"):
        aeon_scale.validate_config(config)


def test_validation_checks_require_scheduled_endpoint() -> None:
    checks = [
        {"step": 2500, "primary_daily_mean_pinball_db": 0.61, "path": "a.npz"},
        {"step": 5000, "primary_daily_mean_pinball_db": 0.59, "path": "b.npz"},
        {"step": 7500, "primary_daily_mean_pinball_db": 0.59, "path": "c.npz"},
    ]
    aeon_scale.validate_validation_checks(checks, supervised_updates=7500)
    with pytest.raises(ValueError, match="endpoint"):
        aeon_scale.validate_validation_checks(checks[:-1], supervised_updates=7500)


def test_second_stage_requires_corrected_endpoint_gain() -> None:
    old = {"direct": 0.60, "ema_jepa": 0.60}
    new = {
        "direct": {"protocol_validation_metrics": {"primary_daily_mean_pinball_db": 0.595}},
        "ema_jepa": {"protocol_validation_metrics": {"primary_daily_mean_pinball_db": 0.57}},
    }
    assert aeon_scale.authorize_stage_two(old, new) is True
    new["ema_jepa"]["protocol_validation_metrics"]["primary_daily_mean_pinball_db"] = 0.595
    assert aeon_scale.authorize_stage_two(old, new) is False


def test_neural_adapter_uses_selected_checkpoint_and_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(slot, fit, assess, config, stage, device):
        for step in (2500, 5000):
            (stage / f"checkpoint-supervised-{step}.pt").write_bytes(str(step).encode())
            with (stage / f"validation-at-supervised-{step}.npz").open("wb") as stream:
                np.savez_compressed(stream, quantiles_db=np.full((1, 3, 5), float(step)))
        return np.full((1, 3, 5), 5000.0), stage / "checkpoint-supervised-5000.pt", {
            "checkpoints": [
                {"phase": "supervised", "step": step, "path": f"checkpoint-supervised-{step}.pt"}
                for step in (2500, 5000)
            ],
            "validation_checks": [
                {"step": step, "path": f"validation-at-supervised-{step}.npz",
                 "sha256": aeon_scale._sha256(stage / f"validation-at-supervised-{step}.npz"),
                 "primary_daily_mean_pinball_db": score}
                for step, score in ((2500, 0.55), (5000, 0.6))
            ],
        }

    monkeypatch.setattr(aeon_scale.aeon_campaign, "_neural_prediction", fake_train)
    monkeypatch.setattr(
        aeon_scale, "_protocol_score_prediction",
        lambda path, rows: (
            {"primary_daily_mean_pinball_db": 0.55 if "2500" in path.name else 0.6},
            ["d" * 64] * 3,
        ),
    )
    config = _config()
    config["neural"]["direct_supervised_updates"] = 5000
    forecast, checkpoint, details = aeon_scale._execute_neural(
        aeon_scale.stage_slots(1)[0], [], [], config, tmp_path, "cpu"
    )
    assert np.all(forecast == 5000)
    assert checkpoint.name == "checkpoint-supervised-5000.pt"
    assert details["final_endpoint_step"] == 5000
    assert details["protocol_validation_checks"][-1]["protocol_primary_daily_mean_pinball_db"] == 0.6
    assert details["protocol_validation_checks"][0]["protocol_primary_daily_mean_pinball_db"] == 0.55
    assert "validation_checks" not in details


def test_protocol_checkpoint_score_excludes_sub_18_anchor_day(tmp_path: Path) -> None:
    time = np.datetime64("2024-10-09T01:00", "us")
    base = AeonHourlyWindow(
        row_id="a" * 64, partition="validation", cutoff_source_timestamp=time,
        cutoff_interval_id=1, context_db=np.full((24, 4), -80.0),
        context_mask=np.ones((24, 4), dtype=bool),
        context_interval_ids=np.arange(24),
        context_source_timestamps=time - np.arange(23, -1, -1) * np.timedelta64(1, "h"),
        target_interval_ids=np.array([2, 4, 7]),
        target_source_timestamps=time + np.array([1, 3, 6]) * np.timedelta64(1, "h"),
        target_db=np.full(3, -80.0), target_mask=np.ones(3, dtype=bool),
        target_qc_status=("OBSERVED_SOURCE_PRODUCT",) * 3,
        source_archive_sha256="b" * 64, past_members=("past.csv",),
        target_members=("future.csv",),
    )
    rows = [
        replace(
            base, row_id=f"{index:064x}", cutoff_interval_id=index,
            target_source_timestamps=(
                time + np.timedelta64(index, "m") + np.array([1, 3, 6]) * np.timedelta64(1, "h")
            ),
        )
        for index in range(18)
    ]
    rows.append(replace(
        base, row_id=f"{18:064x}", cutoff_interval_id=18,
        target_source_timestamps=(
            np.datetime64("2024-10-10T01:00", "us")
            + np.array([1, 3, 6]) * np.timedelta64(1, "h")
        ),
    ))
    forecast = np.full((19, 3, 5), -80.0)
    forecast[-1] = -40.0
    saved = tmp_path / "validation.npz"
    _save_predictions(saved, rows, forecast)
    score, dates = aeon_scale._protocol_score_prediction(saved, rows)
    assert score["primary_daily_mean_pinball_db"] == 0.0
    assert score["eligible_days_per_horizon"] == [1, 1, 1]
    assert len(dates) == 3
