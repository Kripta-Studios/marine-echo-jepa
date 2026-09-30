"""Preserve root's observed CLI completion; an empty log alone is not exit proof."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    paths = {"manifest": ROOT / "orchestration/native_development_comparison_v2.json",
             "review": folder / "development-comparison-v2-review-final.json",
             "result": folder / "development-comparison-result-v2.json",
             "log": folder / "development-comparison-execution-v2.log"}
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if result["role"] != "development" or len(result["methods"]) != 20:
        raise ValueError("Completed expanded reconstruction missing")
    receipt = {"status": "COORDINATOR_OBSERVED_EXECUTION_COMPLETION",
               "coordinator_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
               "command": ".venv/Scripts/python.exe -B -m marine_echo.evaluation.native_comparison --manifest orchestration/native_development_comparison_v2.json --review evidence/ssl-research-v1/development-comparison-v2-review-final.json --output evidence/ssl-research-v1/development-comparison-result-v2.json",
               "all_admission_bindings_verified_before_execution": 150,
               "execution_observation": {"source": "coordinator write_stdin completion tool response",
                                         "exec_session_id": 55183, "completion_chunk_id": "b63338", "exit_code": 0},
               "empty_log_is_independent_exit_proof": False,
               "independent_process_exit_observation": "NOT_RUN",
               "independent_numerical_reconstruction": "SEPARATE_REVIEW",
               "bindings": {str(p): digest(p) for p in paths.values()},
               "methods": 20, "final_numeric_access": "NOT_RUN", "fitting": False}
    with (folder / "development-comparison-execution-receipt-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "observed_exit_code": 0,
                      "independent_exit_observation": "NOT_RUN"}))


if __name__ == "__main__":
    main()
