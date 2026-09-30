"""Close actual root synthetic codec and owned CPU lifecycle checks."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder = ROOT / "evidence/ssl-native-transfer-corpus-builder-v1"
    handoff = json.loads((folder / "handoff-v1.json").read_bytes())
    for name, expected in handoff["authored"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Closed integrated source changed")
    log_path = folder / "root-checks-v1.log"
    if "54 passed in 437.85s" not in log_path.read_text(encoding="utf-8"):
        raise ValueError("Actual combined root check result required")
    lifecycle = folder / "SYNTHETIC_CORRECTNESS_ONLY-owned-e11346bb-593e-4ffe-9399-92a498d1dff7"
    resources = json.loads((lifecycle / "resources.json").read_bytes())
    measured = resources["resources"]
    if (resources.get("evidence_kind") != "SYNTHETIC_CORRECTNESS_ONLY"
            or measured.get("exit_code") != 0 or measured.get("stopped_for") is not None
            or measured.get("owned_tree_cleanup_verified") is not True
            or measured["peak_process_rss_bytes"] >= 22 * 1024**3
            or "1 passed" not in (lifecycle / "stdout.log").read_text(encoding="utf-8")):
        raise ValueError("Actual supervised private codec and cleanup required")
    receipt = {"status": "ROOT_NATIVE_TRANSFER_CORPUS_54_CHECKS_AND_OWNED_CPU_CODEC_PASSED",
               "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
               "combined_suite": {"actual_cli_exit_code": 0, "session_id": 3652,
                                  "actual_exit_chunk_id": "149a3d", "passed": 54, "elapsed_seconds": 437.85},
               "owned_codec": {"actual_cli_exit_code": 0, "session_id": 29252,
                               "actual_exit_chunk_id": "0c2ffd", "resources": measured},
               "bindings": {str(ROOT / name): sha for name, sha in handoff["authored"].items()},
               "public_numeric_access": "NOT_RUN", "fitting": False, "independent_review": "NOT_RUN"}
    for path in (folder / "handoff-v1.json", log_path, lifecycle / "resources.json", lifecycle / "stdout.log",
                 folder / "root_supervised_fixture.py", Path(__file__).resolve()):
        receipt["bindings"][str(path)] = digest(path)
    with (folder / "root-checks-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "public_numeric_access": "NOT_RUN"}))


if __name__ == "__main__":
    main()
