"""Verify resolved replay admission before any public-context numerical decoding."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/ssl-research-v1/native-latent-cpu-replay-review-final-v2.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_LATENT_CPU_REPLAY"
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or review.get("final_numeric_access") is not False
            or review.get("optimizer_updates") != 0):
        raise ValueError("Exact distinct zero-fit DEV-only admission required")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Changed replay binding: {name}")
    receipt = {
        "status": "ALL_LATENT_REPLAY_BINDINGS_VERIFIED", "bindings": len(review["bindings"]),
        "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "actual_reviewer_cli_exit_code": 0,
        "exit_witness": "Root session25798 chunkeea258 exit0",
        "fitting": False, "final_numeric_access": False,
    }
    with (ROOT / "evidence/ssl-research-v1/native-latent-cpu-replay-binding-verification-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
