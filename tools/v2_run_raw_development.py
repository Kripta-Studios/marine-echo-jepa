"""Execute reviewed real raw-response ridge and direct-neural development."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from marine_echo.training.raw_response_development import execute

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--resume", action="store_true", help="Continue an interrupted reviewed step-64 run"
    )
    args = parser.parse_args()
    execute(Path.cwd(), args.output, args.review, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
