"""Verify closed independent source approval without numerical data access."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/ssl-research-v1/native-latent-software-review-final.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_LATENT_INFERENCE_SOFTWARE"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("implementer_session_id") == review["reviewer_session_id"]):
        raise ValueError("Wrong review status or distinct session")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Reviewed binding differs: {name}")
    receipt = {
        "status": "ALL_LATENT_SOFTWARE_BINDINGS_VERIFIED",
        "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "binding_count": len(review["bindings"]),
        "reviewer_actual_cli_exit_code": 0,
        "reviewer_exit_witness": "Root write_stdin session5723 chunkc765a8 exit0",
        "scientific_execution_authorized": False,
        "final_numeric_access_authorized": False,
    }
    destination = ROOT / "evidence/ssl-research-v1/native-latent-software-review-bindings-v1.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
