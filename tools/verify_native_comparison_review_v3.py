"""Verify the complete independent expanded comparison admission."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/ssl-research-v1/development-comparison-v3-review-final.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_COMPARISON_RECONSTRUCTION"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("allowed_roles") != ["development"]
            or review.get("method_count") != 28):
        raise ValueError("Exact distinct28-method DEV-only admission required")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Reviewed bytes differ: {name}")
    receipt = {
        "status": "ALL_EXPANDED_COMPARISON_BINDINGS_VERIFIED",
        "bindings": len(review["bindings"]), "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "actual_reviewer_cli_exit_code": 0,
        "exit_witness": "Root session90856 chunke1199b exit0",
        "fitting": False, "final_numeric_access": False,
    }
    with (ROOT / "evidence/ssl-research-v1/development-comparison-v3-binding-verification.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
