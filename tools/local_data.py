"""Local PANGAEA inventory and bounded extraction commands.

This helper never downloads data. It records local integrity hashes, which are
not publisher signatures because no publisher SHA-256 was supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marine_echo.data.local_archive import (
    extract_local_zip,
    inventory_local_source,
)


def _atomic_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".stage.",
        delete=False,
    ) as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        staged = Path(stream.name)
    os.replace(staged, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory or safely extract immutable local acoustic files."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--source", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--max-expanded-gib", type=int, default=20)
    extract.add_argument("--reserve-gib", type=int, default=10)
    args = parser.parse_args()
    if args.command == "inventory":
        result = inventory_local_source(args.source)
        result["integrity_scope"] = "local_sha256_only_publisher_signature_unavailable"
        _atomic_json(args.output, result)
        print(json.dumps({"output": str(args.output), "files": len(result["files"])}))
        return 0
    result = extract_local_zip(
        args.archive,
        args.destination,
        max_expanded_bytes=args.max_expanded_gib * 1024**3,
        min_free_bytes=args.reserve_gib * 1024**3,
    )
    summary = {
        key: result[key]
        for key in (
            "archive_name",
            "archive_bytes",
            "archive_sha256",
            "members",
            "expanded_bytes",
        )
    }
    summary["destination"] = str(args.destination.resolve())
    summary["integrity_scope"] = (
        "local_sha256_and_zip_crc_only_publisher_signature_unavailable"
    )
    _atomic_json(args.output, summary)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
