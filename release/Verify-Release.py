"""Verify every packaged file before running the local diagnostic."""
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
failures = []
for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    expected, name = line.split("  ", 1)
    candidate = root / name
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        failures.append(name)
        continue
    with candidate.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        failures.append(name)
if failures:
    print("Release integrity failed: " + ", ".join(failures))
    sys.exit(2)
print("All packaged file hashes verified.")
