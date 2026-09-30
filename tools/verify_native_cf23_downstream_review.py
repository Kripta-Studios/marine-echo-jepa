"""Check the actual seed23 prefit approval and every exact byte binding."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/ssl-research-v1/cf-seed23-downstream-prefit-review-final.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "APPROVED_DOWNSTREAM_PREFIT" or review.get("allowed_seeds") != [23]
            or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"):
        raise ValueError("Exact distinct seed23 prefit approval required")
    for name, expected in review["bindings"].items():
        with Path(name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Stale reviewed bytes: {name}")
    receipt = {"status": "ALL_CF_SEED23_DOWNSTREAM_BINDINGS_VERIFIED", "bindings": len(review["bindings"]),
               "review_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reviewer_cli_exit_code": 0,
               "reviewer_exit_witness": "Root write_stdin session67643 chunkdde38c exit0",
               "prompt_correction": "Initial seed23 task retained stale seed13 inline hashes/score; independent reviewer identified conflict and verified actual seed23 files. Corrected task_v2 retained; no incorrect scientific bytes were changed.",
               "execution": "NOT_RUN", "final_numeric_access": "NOT_RUN"}
    with (ROOT / "evidence/ssl-research-v1/cf-seed23-downstream-review-bindings-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
