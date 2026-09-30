"""Stdlib-only exact closed byte hashes; no scientific files are decoded."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
proof = json.loads((HERE / "source-proof-closed-v4.json").read_bytes())
for path, expected in proof["authored"].items():
    if hashlib.sha256((BUILDER / path).read_bytes()).hexdigest() != expected:
        raise ValueError("Closed authored bytes changed: " + path)
for path, expected in proof["source_closure"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("Closed source closure changed: " + path)
print(
    json.dumps(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.iterdir())
            if p.is_file()
        },
        indent=2,
    )
)
