"""Root-owned, independently reviewed CPU native transfer corpus CLI.

Run inside the immutable owned-process supervisor, RSS limit 22 GiB. This tool
never grants final numeric access, performs fitting, or changes frozen selection.
"""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    from marine_echo.data.native_transfer_corpus import materialize

    result = materialize(args.manifest, args.review, args.output, args.receipt)
    print(json.dumps({"status": result["status"], "issued": result["issued"], "device": "cpu"}))


if __name__ == "__main__":
    main()
