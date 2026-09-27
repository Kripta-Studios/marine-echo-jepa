"""The v2 scheduler accounts for each finite slot using saved validation rows."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.models.v2_development import JointPrediction
from marine_echo.training.v2_campaign import (
    SlotResult,
    _canonical_digest,
    _validation_digest,
    _verified_training_phases,
    _verify_ledger,
    execute_v2_campaign,
    v2_plan,
    validate_update_cadence,
)
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


def test_validation_identity_includes_inputs_and_provenance() -> None:
    row = _validation_row()
    original = _validation_digest([row])
    changed = replace(row, context_age_minutes=row.context_age_minutes + 1)
    assert _validation_digest([changed]) != original
    changed = replace(row, context_mask=np.ones_like(row.context_mask))
    assert _validation_digest([changed]) != original
    changed = replace(row, target_acquisition_fraction=np.zeros(3))
    assert _validation_digest([changed]) != original
    changed = replace(row, context_index_db=np.full(96, -90.0))
    assert _validation_digest([changed]) != original
    changed = replace(row, past_source_sha256=("c" * 64,))
    assert _validation_digest([changed]) != original


def test_frozen_250_update_cadence_and_early_stop() -> None:
    scores = (0.5, 0.4, 0.5, 0.5, 0.5, 0.5)
    validate_update_cadence(
        updates=1500,
        checkpoint_steps=(250, 500, 750, 1000, 1250, 1500),
        validation_scores=scores,
    )
    with pytest.raises(ValueError, match="early stop"):
        validate_update_cadence(
            updates=1750,
            checkpoint_steps=(250, 500, 750, 1000, 1250, 1500, 1750),
            validation_scores=(*scores, 0.6),
        )
    with pytest.raises(ValueError, match="250"):
        validate_update_cadence(
            updates=1000,
            checkpoint_steps=(250, 500, 1000),
            validation_scores=(0.5, 0.4, 0.3, 0.2),
        )


def test_real_learned_slot_cannot_skip_checkpoint_phases(tmp_path: Path) -> None:
    slot = next(slot for slot in v2_plan() if slot.run_id == "direct-development0-seed7")
    with pytest.raises(ValueError, match="declared training phases"):
        _verified_training_phases(
            slot,
            SlotResult(tmp_path / "absent.npz", updates=3000),
            tmp_path,
            [_validation_row()],
            V2_PROTOCOL_SHA256,
        )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "direct"),
        ("phase", "development"),
        ("seed", 13),
        ("selected_configuration", 0),
        ("updates", 9),
        ("reusable_seed7", True),
        ("status", "COMPLETED_TRAIN_VALIDATION"),
    ],
)
def test_campaign_resume_rejects_tampered_completed_slot_metadata(
    field: str, value: object, tmp_path: Path
) -> None:
    rows = [_validation_row()]
    root = tmp_path / "campaign"

    def backend(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        path = run_dir / "validation-predictions.npz"
        _write_predictions(
            path,
            rows,
            JointPrediction(np.full((1, 3, 5), -90.0), np.full((1, 3), 0.2)),
        )
        return SlotResult(path, updates=0)

    execute_v2_campaign(
        root,
        validation_rows=rows,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
        max_slots=1,
    )
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["runs"]["persistence-seed7"][field] = value
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="slot|status|artifact"):
        execute_v2_campaign(
            root,
            validation_rows=rows,
            executor=backend,
            protocol_sha256="b" * 64,
            fixture_only=True,
            max_slots=1,
        )


def test_real_ledger_revalidates_declared_training_phases(tmp_path: Path) -> None:
    rows = [_validation_row()]
    root = tmp_path / "campaign"

    def backend(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        path = run_dir / "validation-predictions.npz"
        _write_predictions(
            path,
            rows,
            JointPrediction(np.full((1, 3, 5), -90.0), np.full((1, 3), 0.2)),
        )
        return SlotResult(path, updates=0)

    ledger = execute_v2_campaign(
        root,
        validation_rows=rows,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
        max_slots=5,
    )
    ledger["identity"]["scope"] = "reviewed-train-validation"
    ledger["status"] = "PARTIAL_TRAIN_VALIDATION"
    for run in ledger["runs"].values():
        if run["status"] == "COMPLETED_SYNTHETIC_FIXTURE":
            run["status"] = "COMPLETED_TRAIN_VALIDATION"
            run["attempt"]["campaign_identity_sha256"] = _canonical_digest(ledger["identity"])
            attempt_path = Path(run["artifact_root"]) / ".attempt.json"
            attempt_path.write_text(json.dumps(run["attempt"], sort_keys=True), encoding="utf-8")
            run["attempt_sha256"] = hashlib.sha256(attempt_path.read_bytes()).hexdigest()
            for item in run["artifacts"]:
                if item["path"] == ".attempt.json":
                    item["sha256"] = run["attempt_sha256"]
            for entry in run["lease_history"]:
                lease = entry["lease"]
                lease["attempt_sha256"] = run["attempt_sha256"]
                lease["campaign_identity_sha256"] = _canonical_digest(ledger["identity"])
                lease_path = Path(run["artifact_root"]) / f".lease-{lease['generation']}.json"
                lease_path.write_text(json.dumps(lease, sort_keys=True), encoding="utf-8")
                entry["sha256"] = hashlib.sha256(lease_path.read_bytes()).hexdigest()
                for item in run["artifacts"]:
                    if item["path"] == lease_path.name:
                        item["sha256"] = entry["sha256"]
            run["active_lease"] = run["lease_history"][-1]["lease"]
    with pytest.raises(ValueError, match="declared training phases"):
        _verify_ledger(
            ledger,
            identity=ledger["identity"],
            slots=v2_plan(),
            rows=rows,
            root=root,
        )


def test_reused_slot_metadata_must_exactly_copy_selected_source(tmp_path: Path) -> None:
    rows = [_validation_row()]
    root = tmp_path / "campaign"

    def backend(slot, selected_config, run_dir):  # type: ignore[no-untyped-def]
        path = run_dir / "validation-predictions.npz"
        _write_predictions(
            path,
            rows,
            JointPrediction(np.full((1, 3, 5), -90.0), np.full((1, 3), 0.2)),
        )
        return SlotResult(path, updates=0, reusable_seed7=slot.phase == "development")

    ledger = execute_v2_campaign(
        root,
        validation_rows=rows,
        executor=backend,
        protocol_sha256="b" * 64,
        fixture_only=True,
    )
    reused_id = "ema_jepa-seed7"
    assert ledger["runs"][reused_id]["status"] == "REUSED_SYNTHETIC_FIXTURE"
    original = deepcopy(ledger["runs"][reused_id])
    ledger_path = root / "ledger.json"
    for field, value in (
        ("artifact_root", "C:/wrong-artifact"),
        ("artifacts", []),
        ("prediction_sha256", "c" * 64),
        ("training_phases", [{"name": "forged"}]),
    ):
        ledger["runs"][reused_id] = {**original, field: value}
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        with pytest.raises(ValueError, match="Reused campaign slot artifacts"):
            execute_v2_campaign(
                root,
                validation_rows=rows,
                executor=backend,
                protocol_sha256="b" * 64,
                fixture_only=True,
            )
    ledger["runs"][reused_id] = original
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
