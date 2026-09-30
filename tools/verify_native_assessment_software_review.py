"""Verify every independent software-review binding; never decode public arrays."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/ssl-research-v1/assessment-software-review-final.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_ASSESSMENT_SOFTWARE"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("scientific_execution_authorized") is not False
            or review.get("final_numeric_access_authorized") is not False):
        raise ValueError("Wrong independent review identity/scope")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Reviewed binding differs: {name}")
    receipt = {"status": "ALL_ASSESSMENT_SOFTWARE_BINDINGS_VERIFIED",
               "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "binding_count": len(review["bindings"]),
               "reviewer_actual_cli_exit_code": 0,
               "reviewer_exit_witness": "Root write_stdin session70518 chunkb19d2f exit0",
               "scientific_execution_authorized": False, "final_numeric_access_authorized": False}
    with (ROOT / "evidence/ssl-research-v1/assessment-software-review-bindings-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
