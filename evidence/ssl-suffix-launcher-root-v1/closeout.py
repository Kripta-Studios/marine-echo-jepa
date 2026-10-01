"""Close bounded CPU launcher engineering; no scientific ledger mutation."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
EXPECTED_LEDGER_SHA = "91bf4dcbedc6a603d019c28bb058549f2f39e95befc8d81de5ca6448dec2c5b0"


def main():
    if (
        hashlib.sha256(
            (ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes()
        ).hexdigest()
        != EXPECTED_LEDGER_SHA
    ):
        raise ValueError("Scientific ledger differs; reconcile before closeout")
    receipts = {}
    for name, expected_exit in (
        ("red-11", 2),
        ("green-12", 0),
        ("hardening-red-13", 1),
        ("green-14", 0),
        ("worker-red-15", 1),
        ("final-16", 0),
    ):
        path = (
            ROOT
            / "evidence/ssl-transfer-integration-root-v1"
            / ("suffix-launcher-" + name + ".json")
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value["exit_code"] != expected_exit
            or not value["resources"]["owned_tree_cleanup_verified"]
        ):
            raise ValueError("Unexpected or unclosed owned receipt")
        if any(
            value[k] != EXPECTED_LEDGER_SHA for k in ("ledger_sha256_before", "ledger_sha256_after")
        ):
            raise ValueError("Scientific ledger changed during engineering")
        receipts[name] = {
            "path": str(path.relative_to(ROOT)),
            "exit_code": expected_exit,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    log = ROOT / "evidence/ssl-transfer-integration-root-v1/suffix-launcher-final-16.log"
    if "36 passed" not in log.read_text(encoding="utf-8"):
        raise ValueError("Final36 checks required")
    paths = [
        ROOT / "tools/execute_bounded_native_suffix_reconstruction_v1.py",
        ROOT / "tests/unit/test_native_suffix_reconstruction_execution_v1.py",
    ]
    state = {
        "kind": "native_ssl_suffix_launcher_state_v1",
        "status": "AUTHOR_VALIDATED_ENGINEERING",
        "source_head_before_delivery": "8621b23",
        "checkpoint_reference_only": "dd012f0",
        "checks_passed": 36,
        "checks_skipped": 0,
        "actual_synthetic_owned_workers": 2,
        "real_execution": "NOT_RUN",
        "independent_scientific_approval": False,
        "reviewer_retry": False,
        "scientific_ledger_sha256_unchanged": EXPECTED_LEDGER_SHA,
        "receipts": receipts,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "production_cpu_counter": "cpu_reconstruction_hours_owned",
        "gpu_hours_unchanged": {
            "aggregate": 17.802355808369175,
            "band": 8.318836388888881,
            "cf": 0,
        },
        "research_complete": False,
        "finalists_frozen": False,
        "report": "docs/NATIVE_SSL_SUFFIX_RECONSTRUCTION_LAUNCHER_V1.md",
    }
    for path in (
        DEST / "closeout.json",
        ROOT / "orchestration/native_ssl_suffix_launcher_state_v1.json",
    ):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Closed36 author-validated launcher checks; scientific ledger unchanged.")


if __name__ == "__main__":
    main()
