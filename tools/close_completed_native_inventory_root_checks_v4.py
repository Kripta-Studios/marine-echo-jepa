"""Record closed root correctness checks without granting scientific authority."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-inventory-preparation-builder-v4"
    log = folder / "root-checks-closed-v4.log"
    if "48 passed in 193.59s" not in log.read_text(encoding="utf-8"):
        raise ValueError("Actual closed root test evidence required")
    handoff = json.loads((folder / "handoff-v4.json").read_bytes())
    for name, expected in handoff["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Closed tested source bytes changed")
    record = {
        "status": "ROOT_METADATA_TOOLING_CORRECTNESS_PASSED_NOT_SCIENTIFIC_APPROVAL",
        "actual_cli_session_id": 87149,
        "actual_exit_code": 0,
        "actual_exit_chunk": "2c6560",
        "tests_passed": 48,
        "elapsed_test_seconds": 193.59,
        "fitting": False,
        "numerical_corpus_decoded": False,
        "complete43_artifact_audit": "NOT_RUN",
        "reference_inventory": "NOT_AUDITED_SEPARATE_SCHEMA_REQUIRED",
        "independent_review": "NOT_RUN",
        "scientific_authority": False,
        "bindings": {str(path): digest(path) for path in (
            log, folder / "handoff-v4.json", folder / "root-integration-v4.json",
            Path(__file__).resolve(), *(ROOT / name for name in handoff["files"]),
        )},
    }
    with (folder / "root-checks-closeout-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "actual_exit_code": 0, "tests_passed": 48}))


if __name__ == "__main__":
    main()
