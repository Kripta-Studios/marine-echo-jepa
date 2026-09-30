"""Archive the closed comparison delivery before protocol-only correction."""

import hashlib
import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"

if __name__ == "__main__":
    old_paths = sorted(p for p in EVIDENCE.iterdir() if p.is_file())
    historical = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in old_paths}
    sources = [
        BUILDER / "src/marine_echo/evaluation/native_comparison.py",
        BUILDER / "tests/unit/test_native_comparison.py",
    ]
    source_hashes = {}
    for path in sources:
        payload = path.read_bytes()
        source_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
        destination = EVIDENCE / (path.name + ".original-v1.txt")
        with destination.open("xb") as stream:
            stream.write(payload)
    protocol = MAIN / "docs/adr/0015-native-acoustic-ssl-research.md"
    print(
        json.dumps(
            {
                "operation": "PRESERVE_CLOSED_DELIVERY_BEFORE_PROTOCOL_CORRECTION",
                "scientific_assessment_occurred": False,
                "historical_evidence_sha256": historical,
                "original_source_sha256": source_hashes,
                "protocol": {
                    "path": str(protocol),
                    "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                },
            },
            indent=2,
        )
    )
