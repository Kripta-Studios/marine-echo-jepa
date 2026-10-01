"""Preserve the new completed-parent review rejection without issuing approvals."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    events = folder / "band-seed23-strong-prefit-events-v3.jsonl"
    entries = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    reason = "This request was blocked by our safety systems. Reason: Potentially unintended activity."
    if not any(item.get("type") == "turn.failed" and item.get("error", {}).get("message") == reason for item in entries):
        raise ValueError("Actual automatic review rejection required")
    proposal_path = ROOT / "orchestration/native_band_replication_downstream_seed23_v3.json"
    proposal = json.loads(proposal_path.read_bytes())
    if proposal.get("seed") != 23 or len(proposal.get("jobs", [])) != 2:
        raise ValueError("Exactly the two new completed-parent proposals required")
    for job in proposal["jobs"]:
        if Path(job["approval"]["runtime_arguments"]["output"]).exists():
            raise ValueError("Every unapproved prospective fit must remain unexecuted")
    paths = (
        events, proposal_path, ROOT / "orchestration/native_band_downstream_references_seed23_v3.json",
        ROOT / "orchestration/ssl_vnext_band_seed23_strong_prefit_v3.txt",
        folder / "band-seed23-strong-prefit-stderr-v3.log", Path(__file__).resolve(),
    )
    record = {
        "status": "AUTOMATIC_COMPLETED_PARENT_PREFIT_REJECTED_NO_AUTHORITY",
        "reviewer_session_id": "01a0ef27-876b-7692-917e-3975afc6893d",
        "actual_cli_session_id": 24354,
        "actual_exit_code": 1,
        "actual_exit_chunk": "a22862",
        "reason": reason,
        "rejected_job_ids": [job["id"] for job in proposal["jobs"]],
        "scientific_prefit": False,
        "fitting_authorized": False,
        "existing_approvals_changed": False,
        "bindings": {str(path): digest(path) for path in paths},
    }
    with (folder / "band-seed23-strong-prefit-rejection-closeout-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "actual_exit_code": 1, "fitting_authorized": False}))


if __name__ == "__main__":
    main()
