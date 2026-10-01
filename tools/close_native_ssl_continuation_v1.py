"""Record this engineering delivery without a scientific approval or ledger write."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close():
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    expected = "91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0"
    if sha(ledger_path) != expected:
        raise ValueError("Scientific ledger changed; do not invent a closed accounting state")
    ledger = json.loads(ledger_path.read_bytes())
    evidence = ROOT / "evidence/ssl-cf-controls-root-v1"
    handoff = ROOT / "evidence/ssl-cf-controls-builder-v1/handoff-v1.json"
    implementation = json.loads(handoff.read_bytes())
    for name, expected_sha in implementation["dependencies_sha256"].items():
        if sha(name) != expected_sha:
            raise ValueError("Protected source changed after delivery")
    for name, expected_sha in implementation["files_sha256"].items():
        if sha(ROOT / name) != expected_sha:
            raise ValueError("Integrated delivery differs from closed builder source")
    manifests = [ROOT / "orchestration/native_cf_matched_controls_v1.json"]
    proposals = list((ROOT / "evidence/ssl-research-v1").glob("cf_*_seed*-prefit-proposal-v1.json"))
    if len(proposals) != 6:
        raise ValueError("Exactly six prospective requests required")
    for path in [*manifests, *proposals]:
        value = json.loads(path.read_bytes())
        if not value["status"].startswith(("REGISTERED_", "PROPOSED_")):
            raise ValueError("Registration is not scientific approval")
        for name, expected_sha in value["bindings"].items():
            if sha(name) != expected_sha:
                raise ValueError("Prospective source/data bindings changed")
    checks = {}
    for name in (
        "integrated-correctness-02",
        "integrated-lint-01",
        "prefit-preparation-checks-01",
        "prefit-preparation-lint-01",
        "register-control-configs-01",
        "register-control-prefit-01",
    ):
        receipt = evidence / (name + ".json")
        value = json.loads(receipt.read_bytes())
        if value["exit_code"] != 0:
            raise ValueError("Actual current check did not pass")
        checks[name] = {
            "receipt": str(receipt),
            "sha256": sha(receipt),
            "log_sha256": sha(value["log"]),
        }
    value = {
        "updated_utc": datetime.now(UTC).isoformat(),
        "status": "ENGINEERING_DELIVERED_RESEARCH_BLOCKED_BY_REVIEW_SERVICE",
        "branch": "research/marine-jepa-vnext",
        "preserved_head_before_changes": "2d51c43dbb624f980c3a964d51bff6d413bde1f7",
        "checkpoint_reference_only": "dd012f0",
        "reset": False,
        "builder": {
            "session_id": implementation["implementer_session_id"],
            "route": "unchanged GPT-6.1 Sol/high workspace-write/never",
            "actual_exec_session": 7721,
            "exit_code": 0,
            "handoff": str(handoff),
            "handoff_sha256": sha(handoff),
        },
        "verification": {
            "unique_cpu_checks_passed": 130,
            "integrated_checks": 123,
            "prospective_review_checks": 7,
            "root_optimizer_resume_checks": 4,
            "evidence_kind": "SYNTHETIC_SOFTWARE_CORRECTNESS",
            "checks": checks,
            "preserved_integrated_initial_failure": str(
                evidence / "integrated-correctness-01.json"
            ),
        },
        "original_campaign": {
            "completed_neural_endpoints": 40,
            "required_neural_endpoints": 43,
            "required_methods": 47,
            "independently_reconstructed_methods": 28,
            "recent_completed_unreconstructed": 5,
            "missing_fits": [
                "band_direct_end_to_end_seed13_h96_replication_v2_ownership_retry01",
                "band_shared_ssl_frozen_readout_seed23_h96_replication_v3",
                "band_shared_ssl_full_finetune_seed23_h96_replication_v3",
            ],
        },
        "control_extension": {
            "study": "CF_MATCHED_CONTROLS",
            "registered_configs": 6,
            "planned_supervised_updates": 15000,
            "new_ssl_updates": 0,
            "real_fits_completed": 0,
            "independent_prefits": 0,
            "manifest": str(manifests[0]),
            "proposals": [str(p) for p in sorted(proposals)],
        },
        "accounting": {
            "ledger_sha256_before_after": sha(ledger_path),
            "ledger_unchanged": True,
            "aggregate_full_owned_gpu_hours": ledger["gpu_hours_spent_owned_scientific_jobs"],
            "band_full_owned_gpu_hours": ledger["native_band_gpu_hours_spent_full_owned"],
            "cf_extension_full_owned_gpu_hours": 0,
            "aggregate_cap": 96,
            "band_cap": 12,
            "cf_extension_cap": 12,
            "evaluation_reserve": 12,
        },
        "scientific_gates": {
            "independent_review_retry": "NOT_RUN",
            "real_training": "NOT_RUN",
            "expanded_numeric_reconstruction": "NOT_RUN",
            "finalist_freeze": "PENDING_COMPLETE_DEV_AND_CONTROLS",
            "reserved_numeric_access": "NOT_RUN",
            "prefix_adaptation": "NOT_RUN",
            "sota": "NOT_ESTABLISHED",
        },
        "remaining_dependency": {
            "operation": "Existing distinct independent prefit/source/access/numerical review path",
            "record": "docs/NATIVE_SSL_REVIEW_SERVICE_DEPENDENCY_V1.md",
            "remediation": "Supported feedback with preserved session/receipts; workspace/admin/service inspection and escalation; recorded legitimate restoration required",
            "automatic_rejection_is_scientific_approval": False,
        },
        "frozen": ["app", "meeting", "browser", "release"],
        "publication": "NOT_RUN",
        "historical_outputs_preserved": True,
        "protected_source_bindings_unchanged": len(implementation["dependencies_sha256"]),
    }
    destination = ROOT / "orchestration/native_ssl_continuation_state_v1.json"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": value["status"],
                "cpu_checks": 130,
                "new_real_fits": 0,
                "ledger_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    close()
