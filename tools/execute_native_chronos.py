"""Run the reviewed pinned zero-shot comparator with shared GPU/resource ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marine_echo.training.native_chronos import run
from marine_echo.training.native_references import digest
from marine_echo.training.native_ssl import Resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dev", "output", "ownership-output", "review", "snapshot"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--history", type=int, choices=(24, 96), default=96)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT" or "chronos2" not in review.get(
        "allowed_methods", []
    ):
        raise ValueError("Exact distinct Chronos prefit approval required.")
    if review["reviewer_session_id"] in (
        review["implementer_session_id"],
        review.get("builder_session_id"),
    ):
        raise ValueError("Chronos requires a distinct reviewer.")
    for raw, expected in review["bindings"].items():
        if digest(Path(raw)) != expected:
            raise ValueError(f"Changed reviewed Chronos binding: {raw}")
    if review["bindings"].get(str(Path(__file__).resolve())) != digest(Path(__file__)):
        raise ValueError("The shared ownership wrapper must itself be independently bound.")
    args.ownership_output.mkdir(parents=True, exist_ok=False)
    # Separate ownership receipt keeps the restart-bound Chronos manifest folder
    # empty until the executor itself establishes its immutable identity.
    with Resources("cuda:0", args.ownership_output) as resources:
        result = run(args.dev, args.output, args.review, args.snapshot, args.history, device="cuda")
        result["shared_owner_resources"] = resources.snapshot()
        receipt = args.ownership_output / "completed.json"
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "daily_dev_pinball": result["metrics"]["primary_pinball_db"],
            }
        )
    )


if __name__ == "__main__":
    main()
