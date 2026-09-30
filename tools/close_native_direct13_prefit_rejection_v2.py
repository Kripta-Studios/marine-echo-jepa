"""Preserve an actual automatic prefit rejection without granting execution authority."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    events = folder / "band-direct13-original-guard-prefit-events-v2.jsonl"
    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    expected = "This request was blocked by our safety systems. Reason: Potentially unintended activity."
    if not any(item.get("type") == "turn.failed" and item.get("error", {}).get("message") == expected for item in records):
        raise ValueError("Actual automatic rejection required")
    proposal = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v2.json"
    document = json.loads(proposal.read_bytes())
    if Path(document["approval"]["runtime_arguments"]["output"]).exists():
        raise ValueError("The unapproved retry must remain unexecuted")
    record = {
        "status": "AUTOMATIC_PREFIT_REVIEW_REJECTED_NO_AUTHORITY",
        "reviewer_session_id": "01a0ef27-876b-7692-917e-3975afc6893d",
        "actual_cli_session_id": 77167,
        "actual_exit_code": 1,
        "exit_evidence_chunk": "c37b6d",
        "reason": expected,
        "scientific_prefit": False,
        "fitting_authorized": False,
        "ledger_modified": False,
        "previous_approvals_changed": False,
        "bindings": {str(path): digest(path) for path in (
            events, proposal, ROOT / "orchestration/ssl_vnext_band_direct13_ownership_retry_prefit_v2.txt",
            folder / "band-direct13-original-guard-prefit-stderr-v2.log", Path(__file__).resolve(),
        )},
    }
    target = folder / "band-direct13-original-guard-prefit-rejection-closeout-v2.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "actual_exit_code": 1, "fitting_authorized": False}))


if __name__ == "__main__":
    main()
