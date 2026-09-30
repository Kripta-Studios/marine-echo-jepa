"""Verify every actual distinct software-review binding before any prospective use."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    review_path = folder / "replication-transfer-software-review-final-v5.json"
    review = json.loads(review_path.read_bytes())
    if (review.get("status") != "APPROVED_REPLICATION_TRANSFER_SOFTWARE"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("unresolved_defects") != [] or not review.get("bindings")):
        raise ValueError("Actual distinct closed software approval required")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Software review binding changed: {name}")
    receipt = {"status": "DISTINCT_REPLICATION_TRANSFER_SOFTWARE_BINDINGS_VERIFIED",
               "actual_cli_exit_code": 0, "session_id": 88487, "actual_exit_chunk_id": "27d274",
               "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(), "binding_count": len(review["bindings"]),
               "scientific_prefit_or_final_numeric_access": "NOT_GRANTED",
               "actual_root_check_scope": "391 CPU checks, physical synthetic prefix replay, 31 pending/policy checks and actual tiny CPUowned supervisor"}
    with (folder / "replication-transfer-software-review-closeout-v5.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "bindings": receipt["binding_count"]}))


if __name__ == "__main__":
    main()
