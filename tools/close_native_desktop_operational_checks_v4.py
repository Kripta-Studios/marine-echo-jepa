"""Record actual closed root checks and the separate rejected software review."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-research-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    logs = {"band-operational-policy-checks-v4.log": "114 passed in 436.42s",
            "band-desktop-review-transport-checks-v4.log": "19 passed in 1.04s"}
    for name, expected in logs.items():
        if expected not in (FOLDER / name).read_text(encoding="utf-8"):
            raise ValueError("Actual closed check log required")
    receipt = {
        "status": "ROOT_OPERATIONAL_114_AND_TRANSPORT_19_CHECKS_PASSED_PREFIT_PENDING",
        "closed_actual_commands": [
            {"session_id": 25127, "exit_code": 0, "exit_chunk_id": "d11a21", "passed": 114},
            {"session_id": 19953, "exit_code": 0, "exit_chunk_id": "2edd3c", "passed": 19}],
        "bindings": {str(FOLDER / name): digest(FOLDER / name) for name in logs},
        "historical_sources_unchanged": True, "cuda_execution": "NOT_RUN_UNDER_NEW_POLICY",
        "prefit_authority": "NOT_GRANTED_BY_CHECKS",
    }
    with (FOLDER / "band-desktop-operational-root-checks-closeout-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    events = FOLDER / "native-transfer-corpus-software-review-events-v1.jsonl"
    if "Potentially unintended activity" not in events.read_text(encoding="utf-8"):
        raise ValueError("Preserve actual safety rejection evidence")
    rejected = {
        "status": "SOFTWARE_REVIEW_NOT_APPROVED_AUTOMATIC_SAFETY_REJECTION",
        "reviewer_session_id": "01a0ef27-876b-7692-917e-3975afc6893d",
        "actual_session_id": 78586, "actual_cli_exit_code": 1, "actual_exit_chunk_id": "60525c",
        "reason_verbatim": "This request was blocked by our safety systems. Reason: Potentially unintended activity.",
        "action": "Read-only native transfer-corpus software review; final approval not produced",
        "blocked_step_identification": "UNKNOWN_FROM_RETURNED_REASON",
        "bindings": {str(events): digest(events)}, "approval": False,
        "final_site_numeric_access": "NOT_RUN", "blocked_operation_retried": False,
    }
    with (FOLDER / "native-transfer-corpus-software-review-rejection-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(rejected, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "separate_data_software_review": rejected["status"]}))


if __name__ == "__main__":
    main()
