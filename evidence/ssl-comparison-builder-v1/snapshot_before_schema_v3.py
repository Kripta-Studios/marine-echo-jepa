"""Preserve the closed protocol correction before mask-schema compatibility."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    historical = {
        p.name: sha(p.read_bytes())
        for p in sorted(EVIDENCE.iterdir())
        if p.is_file() and p != Path(__file__).resolve()
    }
    archived = {}
    for relative in (
        "src/marine_echo/evaluation/native_comparison.py",
        "tests/unit/test_native_comparison.py",
    ):
        source = BUILDER / relative
        payload = source.read_bytes()
        destination = EVIDENCE / (source.name + ".closed-protocol-v2.txt")
        with destination.open("xb") as stream:
            stream.write(payload)
        archived[relative] = {"sha256": sha(payload), "snapshot": destination.name}
    record = {
        "operation": "PRESERVE_CLOSED_PROTOCOL_V2_BEFORE_SCHEMA_CORRECTION",
        "scientific_assessment_occurred": False,
        "historical_evidence_sha256": historical,
        "closed_source_snapshots": archived,
    }
    with (EVIDENCE / "before-schema-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({"preserved_evidence_files": len(historical), "snapshots": archived}))
