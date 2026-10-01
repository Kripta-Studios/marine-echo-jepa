"""Close coauthored CF numerical preparation; never change scientific evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
LEDGER_SHA = "91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0"


def main():
    if (
        hashlib.sha256(
            (ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes()
        ).hexdigest()
        != LEDGER_SHA
    ):
        raise ValueError("Scientific ledger changed")
    handoff = json.loads(
        (ROOT / "evidence/ssl-cf-contrasts-builder-v1/handoff-v1.json").read_text(encoding="utf-8")
    )
    for name, expected in handoff["dependency_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Original protected dependency changed")
    prefix = ROOT / "evidence/ssl-transfer-integration-root-v1"
    receipts = {}
    for name, expected_exit in (
        ("cf-saved-schema-red-17", 1),
        ("cf-contrasts-schema-green-18", 0),
        ("cf-contrasts-lint-19", 1),
        ("cf-contrasts-format-20", 0),
        ("cf-contrasts-final-21", 0),
        ("cf-contrasts-lint-final-22", 0),
    ):
        path = prefix / (name + ".json")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["exit_code"] != expected_exit
            or not value["resources"]["owned_tree_cleanup_verified"]
        ):
            raise ValueError("Unclosed or unexpected receipt")
        if any(value[k] != LEDGER_SHA for k in ("ledger_sha256_before", "ledger_sha256_after")):
            raise ValueError("Engineering changed scientific ledger")
        receipts[name] = {
            "path": str(path.relative_to(ROOT)),
            "exit_code": expected_exit,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if "76 passed" not in (prefix / "cf-contrasts-final-21.log").read_text(encoding="utf-8"):
        raise ValueError("Exact final76 checks required")
    paths = [
        ROOT / "src/marine_echo/evaluation/native_cf_matched_contrasts_v1.py",
        ROOT / "tests/unit/test_native_cf_matched_contrasts_v1.py",
        ROOT / "tests/unit/test_native_cf_saved_schema_v1.py",
    ]
    state = {
        "kind": "native_ssl_cf_numerical_preparation_state_v1",
        "status": "COAUTHORED_ENGINEERING_VALIDATED_SCIENCE_INCOMPLETE",
        "source_head_before_delivery": "f36aa8e",
        "checkpoint_reference_only": "dd012f0",
        "builder_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "builder_requested_model": "gpt-6.1-sol",
        "builder_cli_exit": 0,
        "builder_cli_exit_witness": "session75499 chunka2c63c exit0",
        "builder_checks": 58,
        "root_checks": 76,
        "root_skips": 0,
        "protected_dependencies_unchanged": len(handoff["dependency_hashes"]),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "receipts": receipts,
        "pure_in_memory": True,
        "real_numeric_execution": "NOT_RUN",
        "real_fitting": "NOT_RUN",
        "independent_approval": False,
        "source_lineage": "CALLER_AUTHORED_UNVERIFIED",
        "lightgbm_cutoff_alias": "EXPLICIT_PROSPECTIVE_POLICY_PENDING_GENUINE_REVIEW",
        "reviewer_retry": False,
        "builder_v2_followup_dispatched": False,
        "original_campaign": {
            "completed_neural": 40,
            "required_neural": 43,
            "required_methods": 47,
            "independently_reconstructed_methods": 28,
            "newer_provisional_scores": 5,
        },
        "pending": {
            "band_fits": 3,
            "cf_control_fits": 6,
            "finalists_frozen": False,
            "reserved_zero_shot": "NOT_RUN",
            "prefix_adaptation": "NOT_RUN",
        },
        "scientific_ledger_sha256_unchanged": LEDGER_SHA,
        "gpu_hours_used": {
            "aggregate": 17.802355808369175,
            "band": 8.318836388888881,
            "cf_controls": 0,
        },
        "gpu_caps_hours": {"aggregate": 96, "band": 12, "cf_controls_inside_aggregate": 12},
        "evaluation_reserve_hours": 12,
        "failure_attempts_retained": True,
        "frozen_work": ["app", "meeting", "browser", "release"],
        "scientific_claim": None,
        "research_complete": False,
        "report": "docs/NATIVE_SSL_CF_NUMERICAL_PREPARATION_V1.md",
        "remaining_external_dependency": "docs/NATIVE_SSL_REVIEW_SERVICE_DEPENDENCY_V1.md",
    }
    for path in (
        DEST / "closeout.json",
        ROOT / "orchestration/native_ssl_cf_numerical_preparation_state_v1.json",
    ):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Closed76 CF preparation checks; original science and budgets unchanged.")


if __name__ == "__main__":
    main()
