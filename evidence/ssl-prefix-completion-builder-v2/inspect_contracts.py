"""Source and split metadata only; never decode corpus or checkpoint bytes."""

import hashlib
import json
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2].parent / "marine-echo-jepa"
for name, start, stop in (
    ("src/marine_echo/training/native_downstream.py", 341, 387),
    ("src/marine_echo/training/native_downstream.py", 128, 152),
):
    lines = (MAIN / name).read_text(encoding="utf-8").splitlines()
    print(name, start, stop)
    print("\n".join(lines[start - 1 : stop]))
path = MAIN / "configs/native_ssl_split_v1.json"
split = json.loads(path.read_bytes())
print(
    json.dumps(
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "schema_version": split.get("schema_version"),
            "keys": list(split),
            "sources": [
                {k: s.get(k) for k in ("deployment", "site", "role", "archive_sha256")}
                for s in split["sources"]
            ],
        },
        indent=2,
    )
)
