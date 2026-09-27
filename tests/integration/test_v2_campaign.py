"""The v2 scheduler accounts for each finite slot using saved validation rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from marine_echo.models.v2_development import JointPrediction
from marine_echo.training.v2_campaign import SlotResult, execute_v2_campaign, v2_plan
from marine_echo.training.v2_executor import V2_PROTOCOL_SHA256, _write_predictions
from marine_echo.training.v2_stream import HourlyWindow


def _validation_row() -> HourlyWindow:
    cutoff = np.datetime64("2020-06-01T00:00")
    return HourlyWindow(
        row_id="fixture-validation-1",
        partition="validation",
        cutoff=cutoff,
        context=np.full((96, 4, 64), np.nan),
        context_mask=np.zeros((96, 4, 64), dtype=bool),
        context_acquisition_fraction=np.ones(96),
        context_detection_fraction=np.full(96, 0.2),
        context_age_minutes=np.arange(96, dtype=float),
        target_interval_start=np.array([cutoff + np.timedelta64(h, "h") for h in (0, 2, 5)]),
        target_interval_end=np.array([cutoff + np.timedelta64(h, "h") for h in (1, 3, 6)]),
        target_db=np.full(3, -90.0),
        target_mask=np.ones(3, dtype=bool),
        target_detection_fraction=np.full(3, 0.2),
        target_detection_mask=np.ones(3, dtype=bool),
        target_acquisition_fraction=np.ones(3),
        future_train_db=np.full((3, 4, 4, 64), np.nan),
        future_train_mask=np.zeros((3, 4, 4, 64), dtype=bool),
        source_sha256=("a" * 64,),
    )


def test_v2_plan_has_exact_finite_dependencies() -> None:
    plan = v2_plan()
    assert len(plan) == len({slot.run_id for slot in plan}) == 25
    assert sum(slot.phase == "baseline" for slot in plan) == 4
    assert sum(slot.phase == "development" for slot in plan) == 6
    assert sum(slot.phase == "selected" for slot in plan) == 9
    assert sum(slot.phase == "hybrid" for slot in plan) == 2
    assert sum(slot.phase in ("random_encoder", "shuffled_future") for slot in plan) == 4
    seen = set()
    for slot in plan:
        assert set(slot.dependencies).issubset(seen)
        seen.add(slot.run_id)


def test_fixture_campaign_selects_from_saved_rows_and_reuses_seed7(tmp_path: Path) -> None:
    rows = [_validation_row()]
    invoked = []

    def backend(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        invoked.append(slot.run_id)
        offset = 20.0 if selected_config == 1 else 0.0
        path = run_dir / "validation-predictions.npz"
        _write_predictions(
            path,
            rows,
            JointPrediction(
                quantiles_db=np.full((1, 3, 5), -90.0 + offset),
                detection_fraction=np.full((1, 3), 0.2),
            ),
        )
        return SlotResult(path, updates=2, reusable_seed7=slot.phase == "development")

    ledger = execute_v2_campaign(
        tmp_path / "campaign",
        validation_rows=rows,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
        max_slots=10,
    )
    assert ledger["completed_slots"] == 10
    assert ledger["status"] == "PARTIAL_SYNTHETIC_FIXTURE"
    ledger = execute_v2_campaign(
        tmp_path / "campaign",
        validation_rows=rows,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
    )
    assert ledger["completed_slots"] == 25
    assert ledger["status"] == "COMPLETE_SYNTHETIC_FIXTURE"
    assert ledger["test_opened"] is False
    assert ledger["selected_configs"] == {"direct": 0, "ema_jepa": 0, "shared_sigreg": 0}
    assert len(invoked) == 22  # three selected seed-7 endpoints reuse development artifacts
    for family in ("direct", "ema_jepa", "shared_sigreg"):
        selected = ledger["runs"][f"{family}-seed7"]
        assert selected["status"] == "REUSED_SYNTHETIC_FIXTURE"
        assert selected["reuse_of"] == f"{family}-development0-seed7"


def test_real_campaign_requires_review_before_backend_invocation(tmp_path: Path) -> None:
    invoked = []

    def backend(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        invoked.append(slot.run_id)
        raise AssertionError("Unapproved campaign invoked an executor.")

    with pytest.raises(ValueError, match="independent.*review"):
        execute_v2_campaign(
            tmp_path / "real",
            validation_rows=[_validation_row()],
            executor=backend,
            protocol_sha256=V2_PROTOCOL_SHA256,
        )
    assert not invoked


def test_failed_native_support_stops_real_campaign_before_executor(tmp_path: Path) -> None:
    report = tmp_path / "support.json"
    report.write_text(
        json.dumps({"status": "NATIVE_CANDIDATE_INELIGIBLE_STOP_D1_TARGET_SEARCH"}),
        encoding="utf-8",
    )
    review = tmp_path / "review.json"
    review.write_text("{}", encoding="utf-8")

    def forbidden(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        raise AssertionError("Ineligible D1 support reached an executor.")

    with pytest.raises(ValueError, match="ineligible"):
        execute_v2_campaign(
            tmp_path / "real",
            validation_rows=[_validation_row()],
            executor=forbidden,
            protocol_sha256=V2_PROTOCOL_SHA256,
            review_path=review,
            review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
            support_report_path=report,
            support_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        )
