"""Read immutable source text only; never load numerical artifacts."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "marine-echo-jepa"

if __name__ == "__main__":
    path = ROOT / sys.argv[1]
    lines = path.read_text(encoding="utf-8").splitlines()
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    for span in sys.argv[2:]:
        start, end = map(int, span.split(":"))
        for index in range(start - 1, min(end, len(lines))):
            print(f"{index + 1}: {lines[index]}")
