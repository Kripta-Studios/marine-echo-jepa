"""Preserve automatic prefit rejection without inventing scientific approval."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    events = folder / "band-desktop-operational-prefit-events-v4.jsonl"
    reason = "This request was blocked by our safety systems. Reason: Potentially unintended activity."
    if reason not in events.read_text(encoding="utf-8"):
        raise ValueError("Actual rejection evidence required")
    receipt = {
        "status": "OPERATIONAL_PREFIT_NOT_APPROVED_AUTOMATIC_SAFETY_REJECTION",
        "actual_session_id": 4657, "actual_cli_exit_code": 1, "actual_exit_chunk_id": "eaa562",
        "reviewer_session_id": "01a0ef27-876b-7692-917e-3975afc6893d",
        "action": "Independent prefit review of six prospective owner-authorized desktop operational recipes",
        "reason_verbatim": reason, "specific_blocked_step": "UNKNOWN_FROM_RETURNED_REASON",
        "review_authority": "NONE", "cuda_execution_under_new_policy": "NOT_RUN",
        "blocked_operation_retried": False,
        "bindings": {str(events): hashlib.sha256(events.read_bytes()).hexdigest()},
        "remaining": "Genuine independent prefit is required before the new source can execute training",
    }
    with (folder / "band-desktop-operational-prefit-rejection-closeout-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "training_launched": False}))


if __name__ == "__main__":
    main()
