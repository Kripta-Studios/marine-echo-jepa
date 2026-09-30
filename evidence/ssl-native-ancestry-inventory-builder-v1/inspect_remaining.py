"""Inspect fixed source ranges needed for metadata compatibility."""

from pathlib import Path

MAIN = Path(__file__).resolve().parents[3] / "marine-echo-jepa"
for name, lo, hi in (
    ("src/marine_echo/training/native_downstream.py", 600, 630),
    ("src/marine_echo/data/native_ssl_corpus.py", 394, 452),
    ("src/marine_echo/evaluation/native_assessment_replication.py", 890, 940),
):
    lines = (MAIN / name).read_text(encoding="utf-8").splitlines()
    print(name)
    print("\n".join(f"{i + 1}: {lines[i]}" for i in range(lo - 1, hi)))
