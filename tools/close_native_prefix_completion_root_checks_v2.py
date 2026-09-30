"""Close observed source/optimizer/durable checks without scientific approval."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-prefix-completion-builder-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_proof = json.loads((FOLDER / "source-proof-final-v2.json").read_text(encoding="utf-8"))
    for name, expected in source_proof["authored_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Integrated compatibility source changed")
    logs = {
        "cpu": (FOLDER / "root-cpu-checks-v2.log", "163 passed, 8 skipped", "Root session90915 chunk9ca609 exit0"),
        "optimizer": (FOLDER / "root-optimizer-checks-v2.log", "8 passed, 20 deselected", "Root session61601 chunkc4bc7e exit0"),
        "durable": (FOLDER / "root-durable-fixture-v2.log", '"durable_replay": "PASSED"', "Root session24876 chunk4c9dcb exit0"),
    }
    checks = {}
    for kind, (path, expected, witness) in logs.items():
        if expected not in path.read_text(encoding="utf-8"):
            raise ValueError("Observed check log does not match closeout")
        checks[kind] = {"path": str(path), "sha256": digest(path),
                        "actual_cli_exit_code": 0, "exit_witness": witness}
    result = {
        "status": "PREFIX_COMPATIBILITY_ROOT_CHECKS_PASSED_NOT_PREFIT_APPROVAL",
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "checks": checks,
        "passed_checks_total": 171,
        "authored_sha256": source_proof["authored_sha256"],
        "source_proof_sha256": digest(FOLDER / "source-proof-final-v2.json"),
        "integration_receipt_sha256": digest(FOLDER / "root-integration-v2.json"),
        "actual_optimizers": "eight explicit root-only CPU checks passed",
        "durable": "actual synthetic four-update fit, saved safe weights and forecast replay passed",
        "independent_review": "NOT_RUN", "public_numeric_access": False,
        "scientific_training": False, "final_numeric_access": False,
        "remaining_native_compatibility": "single deployment with multiple150/180 configurations and nominal structural-absence IDs under separate v3 engineering task",
    }
    with (FOLDER / "root-checks-closeout-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "passed_checks": 171}))


if __name__ == "__main__":
    main()
