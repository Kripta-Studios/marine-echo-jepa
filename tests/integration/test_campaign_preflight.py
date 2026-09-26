"""Read-only pre-test campaign evidence checks; no acoustic payload access."""

import hashlib
import json
from pathlib import Path

from marine_echo.training.campaign_preflight import inspect_campaign_preflight


def _write(root: Path, name: str, value: dict) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _case(tmp_path: Path) -> Path:
    _write(
        tmp_path,
        "reports/active/protocol.json",
        {"status": "DRAFT_REQUIRES_R1_AND_R2", "test_opened": False},
    )
    _write(tmp_path, "orchestration/reviews/R0.json", {"verdict": "BLOCKED"})
    _write(tmp_path, "orchestration/reviews/R1.json", {"verdict": "REQUEST_CHANGES"})
    _write(
        tmp_path,
        "orchestration/reviews/R0-R1-continuation-20260926.json",
        {"R0": "BLOCKED", "R1": "REQUEST_CHANGES", "approval": False},
    )
    _write(
        tmp_path,
        "evidence/continuation/train_census_execution.json",
        {
            "status": "RUNNING_TRAIN_ONLY_CENSUS",
            "days": {},
            "held_out_acoustic_payloads_processed": False,
            "completed_benchmark_runs": 0,
        },
    )
    _write(
        tmp_path,
        "reports/active/training_registry.json",
        {"runs": [{"status": "BLOCKED", "metrics": None}]},
    )
    return tmp_path


def test_preflight_reports_current_gates_without_mutating_registry(tmp_path: Path) -> None:
    root = _case(tmp_path)
    registry = root / "reports/active/training_registry.json"
    before = registry.read_bytes()
    result = inspect_campaign_preflight(root)
    assert result["status"] == "BLOCKED"
    assert result["can_execute"] is False
    assert result["new_benchmark_runs"] == 0
    assert result["executor_status"] == "REAL_EXECUTOR_MISSING"
    assert {
        "R0_NOT_APPROVED",
        "R1_NOT_APPROVED",
        "CENSUS_INCOMPLETE",
        "REAL_EXECUTOR_MISSING",
    }.issubset(set(result["blockers"]))
    assert (
        result["evidence_sha256"]["reports/active/training_registry.json"]
        == hashlib.sha256(before).hexdigest()
    )
    assert registry.read_bytes() == before


def test_preflight_rejects_unbound_completed_census_and_impossible_day_bound(
    tmp_path: Path,
) -> None:
    root = _case(tmp_path)
    protocol = root / "reports/active/protocol.json"
    execution = root / "evidence/continuation/train_census_execution.json"
    execution_doc = json.loads(execution.read_text())
    execution_doc.update(
        status="COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK",
        identity={
            "bindings": {"protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}
        },
    )
    execution.write_text(json.dumps(execution_doc), encoding="utf-8")
    _write(
        root,
        "evidence/continuation/train_census_eligibility.json",
        {
            "status": "TRAIN_CENSUS_COMPLETE_D1_INELIGIBLE_UPPER_BOUND",
            "execution_report_sha256": hashlib.sha256(execution.read_bytes()).hexdigest(),
            "method_identity": execution_doc["identity"],
            "processed_train_calendar_days": 100,
            "unresolved_train_days": 0,
            "eligible_train_cutoff_days": 0,
            "overall_eligible_day_upper_bound": 67,
            "minimum_overall_days": 90,
            "held_out_acoustic_payloads_processed": False,
            "completed_benchmark_runs": 0,
            "benchmark_eligible": False,
        },
    )
    result = inspect_campaign_preflight(root)
    assert "D1_MINIMUM_IMPOSSIBLE" in result["blockers"]
    assert (
        result["evidence_sha256"]["reports/active/protocol.json"]
        == hashlib.sha256(protocol.read_bytes()).hexdigest()
    )
    protocol.write_text(protocol.read_text(encoding="utf-8") + " ", encoding="utf-8")
    changed = inspect_campaign_preflight(root)
    assert "CENSUS_PROTOCOL_HASH_MISMATCH" in changed["blockers"]
    assert "D1_MINIMUM_IMPOSSIBLE" not in changed["blockers"]


def test_forged_approval_flags_and_fixture_manifest_never_authorize_execution(
    tmp_path: Path,
) -> None:
    root = _case(tmp_path)
    _write(root, "orchestration/reviews/R0.json", {"verdict": "APPROVED"})
    _write(root, "orchestration/reviews/R1.json", {"verdict": "APPROVED"})
    _write(
        root,
        "orchestration/reviews/R0-R1-continuation-20260926.json",
        {
            "R0": "APPROVED",
            "R1": "APPROVED",
            "approval": True,
        },
    )
    _write(
        root,
        "data/processed/canonical/processing_manifest.json",
        {
            "promotion_status": "NONPROMOTABLE_ENGINEERING_FIXTURE",
            "schema_version": "2.0",
        },
    )
    result = inspect_campaign_preflight(root)
    assert result["can_execute"] is False
    assert "FIXTURE_CORPUS_CANNOT_PROMOTE" in result["blockers"]
    assert "REAL_EXECUTOR_MISSING" in result["blockers"]


def test_preflight_never_opens_npz_and_rejects_linked_evidence(tmp_path: Path) -> None:
    root = _case(tmp_path)
    protected = root / "data/processed/canonical/test.npz"
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(b"protected sentinel")
    outside = root / "outside.json"
    outside.write_text('{"verdict":"APPROVED"}', encoding="utf-8")
    linked = root / "orchestration/reviews/R0.json"
    linked.unlink()
    try:
        linked.symlink_to(outside)
    except OSError:
        return
    result = inspect_campaign_preflight(root)
    assert "LINKED_EVIDENCE" in result["blockers"]
    assert protected.read_bytes() == b"protected sentinel"
