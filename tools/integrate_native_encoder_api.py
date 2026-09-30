"""Integrate new encoder-only files without changing active fit sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    folder = BUILDER / "evidence/ssl-encoder-api-builder-v1"
    receipt = json.loads((folder / "handoff-verification.json").read_text())
    for raw, expected in receipt["files_sha256"].items():
        if hashlib.sha256((BUILDER / raw).read_bytes()).hexdigest() != expected:
            raise ValueError("Builder handoff bytes changed.")
    destination = ROOT / "evidence/ssl-encoder-api-builder-v1"
    paths = ("src/marine_echo/inference/native_encoder.py", "tests/unit/test_native_encoder.py")
    if destination.exists() or any((ROOT / path).exists() for path in paths):
        raise ValueError("New integration path exists; preserve all prior files.")
    destination.mkdir()
    for original in sorted(folder.rglob("*")):
        if original.is_file():
            target = destination / original.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(original.read_bytes())
    for raw in paths:
        target = ROOT / raw
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((BUILDER / raw).read_bytes())
    report = {
        "status": "NEW_API_INTEGRATED_NOT_SCIENTIFICALLY_REVIEWED",
        "paths": list(paths),
        "active_fit_sources_changed": False,
    }
    with (ROOT / "evidence/ssl-research-v1/encoder-api-integration.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
