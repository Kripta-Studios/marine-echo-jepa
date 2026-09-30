"""Verify independent approval identities and every immutable binding without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT_SESSION = "01a0ef1a-b166-7f83-91b7-2a2aff7c1b10"
BUILDER_SESSION = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
REVIEWER_SESSION = "01a0ef27-876b-7692-917e-3975afc6893d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT":
        raise ValueError("Scientific prefit approval is absent.")
    expected = {
        "implementer_session_id": ROOT_SESSION,
        "builder_session_id": BUILDER_SESSION,
        "reviewer_session_id": REVIEWER_SESSION,
    }
    for key, value in expected.items():
        if review.get(key) != value:
            raise ValueError(f"Independent routing identity mismatch: {key}")
    stale, canonical, aliases = [], [], []
    for raw, expected_hash in review["bindings"].items():
        path = Path(raw).resolve(strict=True)
        if str(path) != raw:
            item = {"review_key": raw, "canonical_path": str(path)}
            if Path(raw).is_absolute() and Path(raw).is_symlink():
                # Hugging Face's immutable snapshot uses ordinary file symlinks.
                # Hash the resolved bytes, retaining the reviewed snapshot name.
                aliases.append(item)
            else:
                canonical.append(item)
        with path.open("rb") as stream:
            observed_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed_hash != expected_hash:
            stale.append(str(path))
    report = {
        "status": "PASSED" if not stale and not canonical else "REJECTED",
        "review": str(args.review.resolve()),
        "bindings_checked": len(review["bindings"]),
        "stale_bindings": stale,
        "noncanonical_keys": canonical,
        "reviewed_file_symlinks": aliases,
        "allowed_methods": review["allowed_methods"],
        "sessions": expected,
        "scientific_optimizer_updates": 0,
        "assessment_scoring": "NOT_RUN",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    if report["status"] != "PASSED":
        raise ValueError("Exact prefit verification failed; no scientific fitting authorized.")


if __name__ == "__main__":
    main()
