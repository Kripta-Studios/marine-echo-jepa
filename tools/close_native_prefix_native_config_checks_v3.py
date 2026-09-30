"""Close actual root checks while preserving the failed first native attempt."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-prefix-native-config-builder-v3"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    proof = json.loads((FOLDER / "source-proof-final-v3.json").read_bytes())
    for name, expected in proof["authored_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Integrated native configuration source changed")
    authored = {str((ROOT / p).resolve()) for p in proof["authored_sha256"]}
    for name, expected in proof["protected_main_sha256"].items():
        if str(Path(name).resolve()) not in authored and digest(name) != expected:
            raise ValueError(f"Protected original source changed: {name}")
    witness = json.loads((FOLDER / "root-checks-retry-exit-witness-v3.json").read_bytes())
    log = FOLDER / "root-all-checks-retry-01-v3.log"
    if (witness["actual_cli_exit_code"] != 0 or witness["session_id"] != 72811
            or witness["skipped"] != 0 or "237 passed in 195.70s" not in log.read_text(encoding="utf-8")):
        raise ValueError("Actual full unchanged root retry evidence required")
    durable_log = FOLDER / "root-durable-v3.log"
    durable = json.loads(durable_log.read_text(encoding="utf-8"))
    if (durable["status"] != "COMPLETED" or durable["evidence_kind"] != "SYNTHETIC_CORRECTNESS_ONLY"
            or durable["synthetic_optimizer_updates"] != 4 or durable["durable_replay"] != "PASSED"
            or durable["gpu_execution"] is not False or durable["public_scientific_fit"] is not False):
        raise ValueError("Actual physical saved-artifact CPU replay required")
    completion = Path(durable["output"]) / "completion.json"
    if json.loads(completion.read_bytes())["status"] != "COMPLETED":
        raise ValueError("Durable completion receipt differs")
    receipt = {"status": "PREFIX_NATIVE_CONFIG_V3_ROOT_CHECKS_PASSED_NOT_PREFIT_APPROVAL",
               "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "passed_checks_total": 237,
               "prefix_checks": 219, "wrapper_policy_checks": 18, "skipped": 0,
               "cpu_exit_witness": witness,
               "durable_exit_witness": {"actual_cli_exit_code": 0, "actual_tool_chunk_id": "5214eb"},
               "durable": durable, "authored_sha256": proof["authored_sha256"],
               "bindings": {str(p): digest(p) for p in (
                   FOLDER / "source-proof-final-v3.json", FOLDER / "root-integration-v3.json", log,
                   durable_log, completion, FOLDER / "root-all-checks-native-failure-v3.json",
                   FOLDER / "root-native-failure-os-event-v3.json",
                   ROOT / "tools/execute_native_prefix_job.py", ROOT / "tools/execute_native_prefix_worker.py")},
               "prior_failure_preserved": True, "failure_cause": "UNKNOWN",
               "source_recipe_or_permission_changes_for_retry": False,
               "independent_software_review": "NOT_RUN", "scientific_prefit": "NOT_RUN",
               "public_numeric_access": "NOT_RUN", "final_numeric_access": "NOT_RUN"}
    with (FOLDER / "root-checks-closeout-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "passed_checks": 237, "durable": "PASSED"}))


if __name__ == "__main__":
    main()
