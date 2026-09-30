"""Read-only snapshots for the three specifically authorized prefix paths."""

import hashlib
import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
PATHS = [
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
]

if len(sys.argv) == 2 and sys.argv[1] in ("builder", "main"):
    root = BUILDER if sys.argv[1] == "builder" else MAIN
    result = {}
    for p in PATHS:
        data = (root / p).read_bytes()
        result[p] = {"sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8")}
    print(json.dumps(result))
else:
    raise ValueError("Explicit read-only root required.")
