"""Read-only authored source inspection; no numerical data access."""

import sys
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "src/marine_echo/training/native_prefix_transfer.py"
if len(sys.argv) == 4 and sys.argv[3] == "downstream":
    p = p.parents[3].parent / "marine-echo-jepa/src/marine_echo/training/native_downstream.py"
print(
    "".join(
        p.read_text(encoding="utf-8").splitlines(keepends=True)[int(sys.argv[1]) : int(sys.argv[2])]
    )
)
