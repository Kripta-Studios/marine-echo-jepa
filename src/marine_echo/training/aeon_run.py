"""Run the independently reviewed AEON TRAIN/validation development slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from marine_echo.training.aeon_corpus import AeonDevelopmentReader
from marine_echo.training.aeon_development import execute_aeon_development
from marine_echo.training.aeon_windows import AeonWindowPlan, iter_aeon_windows


def run(archive: Path, review: Path, output: Path) -> dict:
    with review.open("rb") as stream:
        review_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    reader = AeonDevelopmentReader(
        archive, review_path=review, review_sha256=review_sha256
    )
    plan = AeonWindowPlan()
    fit = list(
        iter_aeon_windows(
            reader.iter_partition("train"),
            plan=plan,
            partition="train",
            partition_start="2024-03-06",
            partition_end_exclusive="2024-10-08",
        )
    )
    assess = list(
        iter_aeon_windows(
            reader.iter_partition("validation"),
            plan=plan,
            partition="validation",
            partition_start="2024-10-08",
            partition_end_exclusive="2024-12-01",
        )
    )
    return execute_aeon_development(
        fit,
        assess,
        output,
        plan=plan,
        review_path=review,
        review_sha256=review_sha256,
        updates=128,
        batch_size=16,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.review, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
