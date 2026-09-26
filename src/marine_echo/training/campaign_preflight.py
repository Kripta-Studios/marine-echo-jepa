"""Read-only evidence preflight for the unavailable real pre-test campaign.

This inspection never authorizes execution. The current canonical adapter and
all campaign executors are synthetic-only; independent promotion has no real
corpus interface. The result is suitable for a blocked CLI diagnostic only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_PATHS = {
    "protocol": "reports/active/protocol.json",
    "r0": "orchestration/reviews/R0.json",
    "r1": "orchestration/reviews/R1.json",
    "continuation": "orchestration/reviews/R0-R1-continuation-20260926.json",
    "census_execution": "evidence/continuation/train_census_execution.json",
    "census_eligibility": "evidence/continuation/train_census_eligibility.json",
    "canonical": "data/processed/canonical/processing_manifest.json",
    "registry": "reports/active/training_registry.json",
}
_MAX_JSON_BYTES = 4 * 1024 * 1024


def _read_evidence(
    root: Path, relative: str, hashes: dict[str, str], blockers: list[str]
) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or path.is_junction() or not path.is_file():
        blockers.append("LINKED_EVIDENCE")
        return None
    if path.stat().st_size > _MAX_JSON_BYTES:
        blockers.append("OVERSIZED_EVIDENCE")
        return None
    payload = path.read_bytes()
    hashes[relative] = hashlib.sha256(payload).hexdigest()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        blockers.append("INVALID_JSON_EVIDENCE")
        return None
    if not isinstance(document, dict):
        blockers.append("INVALID_JSON_EVIDENCE")
        return None
    return document


def inspect_campaign_preflight(root: Path) -> dict[str, Any]:
    """Inspect fixed JSON evidence only; return a permanently blocked diagnostic.

    A future real adapter needs a separate reviewed promotion contract. No
    combination of caller-authored approval flags can make this function run a
    model or return an execution-ready decision.
    """
    if root.is_symlink() or root.is_junction() or not root.is_dir():
        raise ValueError("Campaign evidence root must be an unlinked local directory.")
    hashes: dict[str, str] = {}
    blockers: list[str] = []
    evidence = {
        key: _read_evidence(root, relative, hashes, blockers) for key, relative in _PATHS.items()
    }
    protocol = evidence["protocol"]
    if protocol is None or protocol.get("status") == "DRAFT_REQUIRES_R1_AND_R2":
        blockers.append("PROTOCOL_NOT_REVIEWED")
    if protocol is not None and protocol.get("test_opened") is not False:
        blockers.append("PROTECTED_EXPOSURE_REVIEW_REQUIRED")
    r0, r1, continuation = evidence["r0"], evidence["r1"], evidence["continuation"]
    if (
        r0 is None
        or r0.get("verdict") != "APPROVED"
        or (continuation is None or continuation.get("R0") != "APPROVED")
    ):
        blockers.append("R0_NOT_APPROVED")
    if (
        r1 is None
        or r1.get("verdict") != "APPROVED"
        or (continuation is None or continuation.get("R1") != "APPROVED")
    ):
        blockers.append("R1_NOT_APPROVED")
    execution = evidence["census_execution"]
    eligibility = evidence["census_eligibility"]
    if execution is None or execution.get("status") != "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK":
        blockers.append("CENSUS_INCOMPLETE")
    if eligibility is None:
        blockers.append("CENSUS_ELIGIBILITY_MISSING")
    if execution is not None and eligibility is not None:
        protocol_sha = hashes.get(_PATHS["protocol"])
        execution_sha = hashes.get(_PATHS["census_execution"])
        identity = execution.get("identity")
        bound_protocol = (
            identity.get("bindings", {}).get("protocol_sha256")
            if isinstance(identity, dict)
            else None
        )
        if protocol_sha is None or bound_protocol != protocol_sha:
            blockers.append("CENSUS_PROTOCOL_HASH_MISMATCH")
        integrity = (
            execution_sha is not None
            and eligibility.get("execution_report_sha256") == execution_sha
            and eligibility.get("method_identity") == identity
            and execution.get("held_out_acoustic_payloads_processed") is False
            and execution.get("completed_benchmark_runs") == 0
            and eligibility.get("held_out_acoustic_payloads_processed") is False
            and eligibility.get("completed_benchmark_runs") == 0
            and eligibility.get("benchmark_eligible") is False
            and eligibility.get("processed_train_calendar_days") == 100
            and eligibility.get("unresolved_train_days") == 0
            and eligibility.get("minimum_overall_days") == 90
            and type(eligibility.get("eligible_train_cutoff_days")) is int
            and type(eligibility.get("overall_eligible_day_upper_bound")) is int
            and eligibility["overall_eligible_day_upper_bound"]
            == eligibility["eligible_train_cutoff_days"] + 67
        )
        if not integrity:
            blockers.append("CENSUS_INTEGRITY_UNVERIFIED")
        elif (
            bound_protocol == protocol_sha and eligibility["overall_eligible_day_upper_bound"] < 90
        ):
            blockers.append("D1_MINIMUM_IMPOSSIBLE")
    canonical = evidence["canonical"]
    if canonical is None:
        blockers.append("REAL_CORPUS_INTERFACE_MISSING")
    elif canonical.get("promotion_status") == "NONPROMOTABLE_ENGINEERING_FIXTURE":
        blockers.append("FIXTURE_CORPUS_CANNOT_PROMOTE")
    else:
        blockers.append("REAL_CORPUS_PROMOTION_UNVERIFIED")
    blockers.append("REAL_EXECUTOR_MISSING")
    return {
        "status": "BLOCKED",
        "can_execute": False,
        "executor_status": "REAL_EXECUTOR_MISSING",
        "new_benchmark_runs": 0,
        "test_acoustic_values_opened_by_preflight": False,
        "blockers": list(dict.fromkeys(blockers)),
        "evidence_sha256": hashes,
    }
