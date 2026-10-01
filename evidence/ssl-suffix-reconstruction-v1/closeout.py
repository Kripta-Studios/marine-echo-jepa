"""Close this author-validated CPU delivery without changing scientific ledgers."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "evidence/ssl-transfer-integration-root-v1"
LEDGER_SHA = "91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0"


def main():
    ledger = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    if hashlib.sha256(ledger.read_bytes()).hexdigest() != LEDGER_SHA:
        raise ValueError("Scientific ledger changed; reconcile before closeout")
    receipts = {}
    for name, code in (
        ("red-04", 2),
        ("green-05", 0),
        ("hardening-red-06", 1),
        ("green-07", 0),
        ("final-08", 0),
        ("lint-09", 0),
        ("format-10", 0),
    ):
        path = PREFIX / ("suffix-reconstruction-" + name + ".json")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["exit_code"] != code or not receipt["resources"]["owned_tree_cleanup_verified"]:
            raise ValueError("Unclosed or unexpected execution receipt: " + name)
        if any(receipt[k] != LEDGER_SHA for k in ("ledger_sha256_before", "ledger_sha256_after")):
            raise ValueError("Execution changed scientific ledger")
        receipts[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "exit_code": code,
        }
    log = (PREFIX / "suffix-reconstruction-final-08.log").read_text(encoding="utf-8")
    if "26 passed" not in log or "skipped" in log:
        raise ValueError("Exact final 26-check success without skips required")
    sources = [
        ROOT / "src/marine_echo/evaluation/native_suffix_reconstruction_v1.py",
        ROOT / "tests/unit/test_native_suffix_reconstruction_v1.py",
    ]
    state = {
        "kind": "native_ssl_suffix_reconstruction_state_v1",
        "status": "AUTHOR_VALIDATED_ENGINEERING_SCIENCE_INCOMPLETE",
        "source_head_before_delivery": "51739b3",
        "checkpoint_reference_only": "dd012f0",
        "author_validation": True,
        "independent_scientific_approval": False,
        "independent_scoring_implementation": True,
        "real_numeric_execution": "NOT_RUN",
        "real_weight_replay": "NOT_RUN",
        "reviewer_retry": False,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "checks_passed": 26,
        "checks_skipped": 0,
        "receipts": receipts,
        "scientific_ledger_sha256_unchanged": LEDGER_SHA,
        "campaign": {
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
        "gpu_budget": {
            "aggregate_used_hours": 17.802355808369175,
            "aggregate_cap_hours": 96,
            "band_used_hours": 8.318836388888881,
            "band_cap_hours": 12,
            "cf_controls_used_hours": 0,
            "cf_controls_cap_hours": 12,
            "evaluation_reserve_hours": 12,
            "failure_attempts_retained": True,
        },
        "frozen_work": ["app", "meeting", "browser", "release"],
        "remediation": "docs/NATIVE_SSL_REVIEW_SERVICE_DEPENDENCY_V1.md",
        "report": "docs/NATIVE_SSL_SUFFIX_RECONSTRUCTION_ENGINEERING_V1.md",
    }
    for path in (
        ROOT / "orchestration/native_ssl_suffix_reconstruction_state_v1.json",
        Path(__file__).parent / "closeout.json",
    ):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Closed author-validated 26-check delivery; scientific ledger unchanged.")


if __name__ == "__main__":
    main()
