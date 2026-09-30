"""Integrate only the closed protocol- and native-schema-corrected comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    folder = BUILDER / "evidence/ssl-comparison-builder-v1"
    receipt_path = folder / "handoff-verification-delivery-schema-v3.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for raw, expected in receipt["files_sha256"].items():
        if hashlib.sha256((BUILDER / raw).read_bytes()).hexdigest() != expected:
            raise ValueError("Closed comparison delivery hash changed.")
    paths = (
        "src/marine_echo/evaluation/native_comparison.py",
        "tests/unit/test_native_comparison.py",
    )
    destination = ROOT / "evidence/ssl-comparison-builder-v1"
    if destination.exists() or any((ROOT / path).exists() for path in paths):
        raise ValueError("Preserve earlier integration paths.")
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
    proof = {
        "status": "NEW_CORRECTED_COMPARISON_INTEGRATED_NOT_SCIENTIFICALLY_REVIEWED",
        "scientific_scoring": "NOT_RUN",
        "active_fit_sources_changed": False,
        "handoff_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    with (ROOT / "evidence/ssl-research-v1/comparison-integration.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
