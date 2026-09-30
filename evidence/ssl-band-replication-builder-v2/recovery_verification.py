"""Stdlib-only verification of retained closed source/test byte snapshots."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
records = {}
for name in ("final-code-snapshot-v2.json", "final-tests-snapshot-v2.json"):
    for path, record in json.loads((HERE / name).read_bytes()).items():
        current = (BUILDER / path).read_bytes()
        digest = hashlib.sha256(current).hexdigest()
        if digest != record["sha256"] or current != record["text"].encode("utf-8"):
            raise ValueError("Closed authored source/test differs: " + path)
        records[path] = digest
proof = json.loads((HERE / "source-proof-final-v2b.json").read_bytes())
for path, digest in proof["source_closure_sha256"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("Protected source closure changed: " + path)
print(
    json.dumps(
        {
            "kind": "native_band_replication_recovery_verification_v2",
            "authored_sha256": records,
            "snapshot_byte_equality": True,
            "protected_source_closure_unchanged": True,
            "pytest_repeated": False,
            "scientific_approval": False,
        },
        indent=2,
    )
)
