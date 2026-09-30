"""Integrate completed scoped downstream bytes into new coordinator-owned paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    relative = [
        Path("src/marine_echo/training/native_downstream.py"),
        Path("tests/integration/test_native_downstream.py"),
        Path("orchestration/reports/NATIVE_DOWNSTREAM_IMPLEMENTATION_V1.md"),
    ]
    relative += [
        path.relative_to(BUILDER)
        for path in (BUILDER / "evidence/ssl-downstream-builder-v1").rglob("*")
        if path.is_file()
    ]
    for path in relative:
        if (ROOT / path).exists():
            raise ValueError(f"Integration target already exists; no overwrite: {path}")
    receipt = []
    for path in relative:
        raw = (BUILDER / path).read_bytes()
        destination = ROOT / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        receipt.append(
            {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    report = {
        "status": "SCOPED_NEW_BYTES_INTEGRATED",
        "builder": str(BUILDER),
        "denied_builder_paths_retried": False,
        "files": receipt,
        "durable_checks": "PENDING_COORDINATOR",
        "scientific_fits": "NOT_RUN",
    }
    target = ROOT / "evidence/ssl-research-v1/downstream-integration.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "files": len(receipt)}))


if __name__ == "__main__":
    main()
