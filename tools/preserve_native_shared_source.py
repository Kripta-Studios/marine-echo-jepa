"""Retain byte-identical model source for the completed shared screen ancestry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "src/marine_echo/models/native_temporal.py"
    data = path.read_bytes()
    expected = "85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422"
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Historical shared source must be retained before the CF repair.")
    directory = ROOT / "evidence/ssl-research-v1/ancestor-source"
    directory.mkdir(exist_ok=False)
    snapshot = directory / "native_temporal_shared_screen_v1.py"
    with snapshot.open("xb") as stream:
        stream.write(data)
    manifest = {
        "original_path": str(path),
        "path": str(snapshot),
        "sha256": expected,
        "methods": ["shared_ssl"],
        "status": "ORIGINAL_SOURCE_PRESERVED_NOT_COMPATIBILITY_APPROVAL",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
