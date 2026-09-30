"""Check every immutable review binding without decoding scientific arrays."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("status") != args.status:
        raise ValueError("Unexpected review status")
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Missing immutable bindings")
    for name, expected in bindings.items():
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Changed review binding: {name}")
    print(json.dumps({"status": "ALL_BINDINGS_VERIFIED", "count": len(bindings),
                      "review": str(args.review), "reviewer_session_id": review.get("reviewer_session_id")}))


if __name__ == "__main__":
    main()
