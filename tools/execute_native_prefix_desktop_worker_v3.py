"""Execute only a source-bound prefix fit; persist unsupported cells explicitly."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    from marine_echo.training.native_prefix_desktop_transfer_v3 import fit

    result = fit(args.manifest, args.review, args.output, resume=args.resume)
    if result.get("status") == "NOT_ASSESSABLE" and not args.output.exists():
        args.output.mkdir(parents=False, exist_ok=False)
        with (args.output / "completion.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": result["status"], "suffix_fit": False}))


if __name__ == "__main__":
    main()
