"""Read only the closed, bounded V3 source/evidence delivery."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
proof = json.loads((HERE / "source-proof-final-v3.json").read_bytes())
snapshot = json.loads((HERE / "final-source-snapshot-v3.json").read_bytes())
for path, digest in proof["authored_sha256"].items():
    current = (BUILDER / path).read_bytes()
    if hashlib.sha256(current).hexdigest() != digest:
        raise ValueError("Closed authored source changed: " + path)
    if snapshot[path]["text"].encode("utf-8") != current:
        raise ValueError("Final source snapshot differs: " + path)
print(
    json.dumps(
        {
            "authored_sha256": proof["authored_sha256"],
            "evidence_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(HERE.iterdir())
                if path.is_file()
            },
            "snapshot_verified": True,
        },
        indent=2,
    )
)
