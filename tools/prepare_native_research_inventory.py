"""Root-owned explicit completed-endpoint metadata inventory; no final selection."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from marine_echo.evaluation.native_ancestry_inventory import derive_inventory

    result = derive_inventory(args.manifest, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(result["models"]),
                "train_rows": result["train_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
