"""Preserve original delivery and bind executed root-only synthetic checks."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = ROOT / "evidence/ssl-prefix-builder-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handoff = json.loads((FOLDER / "handoff-v1.json").read_text(encoding="utf-8"))
    checks = {
        "root-cpu-checks-v1.log": "98 passed",
        "root-cpu-checks-v2.log": "102 passed",
        "root-source-interval-red-v1.log": "3 failed, 1 passed",
        "root-source-interval-green-v1.log": "4 passed",
        "root-completion-kind-green-v1.log": "1 passed",
        "root-durable-fixture-v1.log": "KeyError",
        "root-durable-fixture-v2.log": '"durable_replay": "PASSED"',
    }
    for name, expected in checks.items():
        if expected not in (FOLDER / name).read_text(encoding="utf-8"):
            raise ValueError(f"Actual check evidence missing: {name}")
    snapshot = FOLDER / "original-delivery-sources-v2"
    snapshot.mkdir()
    original = {}
    for relative, expected in handoff["authored_sha256"].items():
        source = BUILDER / relative
        if digest(source) != expected:
            raise ValueError("Original closed builder delivery changed")
        target = snapshot / (relative.replace("/", "__") + ".txt")
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        original[relative] = {"snapshot": str(target.relative_to(ROOT)), "sha256": digest(target)}
    paths = [ROOT / p for p in handoff["authored_sha256"]]
    paths += [FOLDER / name for name in checks]
    paths += [
        FOLDER / "prefix_test_support.py",
        FOLDER / "root_durable_fixture_check.py",
        ROOT / "docs/adr/0024-prefix-native-interval-proof.md",
    ]
    receipt = {
        "status": "ROOT_SYNTHETIC_CHECKS_COMPLETED_PENDING_INDEPENDENT_REVIEW",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "original_sources": original,
        "bindings": {str(p): digest(p) for p in paths},
        "root_changes": [
            "Native source interval indices plus 55–65-minute clock adjacency",
            "Explicit evidence kind in completed and unsupported receipts",
            "Synthetic registry fixtures and focused regression checks",
        ],
        "cpu_checks_before_receipt_label_change": "102 passed95.86s",
        "receipt_label_regression": "1 actual optimizer check and durable Unicode fit/replay passed",
        "root_original_optimizer_checks": "98 passed97.23s including six actual optimizer/resume cases",
        "failed_fixture_preserved": True,
        "public_fit": False,
        "gpu_execution": False,
        "first_snapshot_attempt": "Partial v1 snapshot preserved; same unit/integration basenames collided (root tool chunk9f9c23 exit1). V2 includes full relative names; no permissions denial occurred.",
        "final_site_numeric_access": "NOT_RUN",
        "independent_review": "NOT_RUN",
    }
    with (FOLDER / "root-checks-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "original_snapshots": len(original)}))


if __name__ == "__main__":
    main()
