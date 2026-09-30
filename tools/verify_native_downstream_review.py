"""Verify every independently approved downstream binding without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    sessions = {
        "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "builder_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
        "reviewer_session_id": "01a0ef27-876b-7692-917e-3975afc6893d",
        "coordinator_session_id": "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10",
    }
    if review.get("status") != "APPROVED_DOWNSTREAM_PREFIT":
        raise ValueError("Separate downstream approval is absent.")
    if any(review.get(key) != value for key, value in sessions.items()):
        raise ValueError("Verified distinct author/reviewer identities differ.")
    stale, noncanonical = [], []
    for raw, expected in review["bindings"].items():
        path = Path(raw).resolve(strict=True)
        if str(path) != raw:
            noncanonical.append(raw)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            stale.append(raw)
    report = {
        "status": "PASSED" if not stale and not noncanonical else "REJECTED",
        "review": str(args.review.resolve()),
        "bindings_checked": len(review["bindings"]),
        "stale_bindings": stale,
        "noncanonical_keys": noncanonical,
        "sessions": sessions,
        "allowed_methods": review["allowed_methods"],
        "allowed_modes": review["allowed_modes"],
        "scientific_optimizer_updates": 0,
        "final_test_access": "NOT_RUN",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    if report["status"] != "PASSED":
        raise ValueError("Downstream immutable verification failed.")


if __name__ == "__main__":
    main()
