"""Stdlib-only final hashes within the explicitly allowed delivery paths."""

import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
builder = here.parents[1]
authored = [
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
]
proof = json.loads((here / "source-proof-final-v2.json").read_bytes())
for path in authored:
    if hashlib.sha256((builder / path).read_bytes()).hexdigest() != proof["authored_sha256"][path]:
        raise ValueError("Frozen authored source changed after final checks.")
print(
    json.dumps(
        {
            str(p.relative_to(builder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *(builder / s for s in authored),
                *(p for p in sorted(here.iterdir()) if p.is_file()),
            ]
        },
        indent=2,
    )
)
