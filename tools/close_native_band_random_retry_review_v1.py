"""Close the distinct exact-recipe retry review without launching a fit."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    proposal_path = ROOT / "orchestration/native_band_random_strong_retry_admission_v1.json"
    proposal = json.loads(proposal_path.read_bytes())
    expected = proposal["approval"]
    review_path = Path(expected["runtime_arguments"]["review"])
    review = json.loads(review_path.read_bytes())
    if (review.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("unresolved_defects") != []
            or review.get("proposal_sha256") != digest(proposal_path)):
        raise ValueError("Closed distinct exact-proposal approval required")
    for key, value in expected.items():
        if key not in ("status", "approval_scope") and review.get(key) != value:
            raise ValueError(f"Proposed scientific field changed: {key}")
    counts = {}
    for field in ("bindings", "proof_bindings"):
        bindings = review.get(field, {})
        if not bindings:
            raise ValueError(f"Missing {field}")
        for name, expected_sha in bindings.items():
            if digest(name) != expected_sha:
                raise ValueError(f"Reviewed bytes changed: {name}")
        counts[field] = len(bindings)
    runtime = review["runtime_arguments"]
    if runtime.get("resume") is not None:
        raise ValueError("Retry must be fresh")
    for field in ("output", "receipt"):
        if Path(runtime[field]).exists():
            raise FileExistsError("Prospective destinations must remain absent")
    receipt = {
        "status": "DISTINCT_EXACT_RANDOM_RETRY_REVIEW_VERIFIED_NOT_LAUNCHED",
        "actual_cli_exit_code": 0,
        "session_id": 95740,
        "actual_exit_chunk_id": "00949c",
        "review_sha256": digest(review_path),
        "proposal_sha256": digest(proposal_path),
        "verified_binding_counts": counts,
        "scientific_fields_unchanged": True,
        "launch_pending": "Four fixed-replication queue closure, reconciled ownership and live budget recomputation",
        "additional_retry_authorized": False,
        "scientific_completion": False,
    }
    with (folder / "band-random-strong-retry-review-closeout-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
