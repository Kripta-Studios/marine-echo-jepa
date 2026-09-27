"""Verify the exact packaged file inventory before serving."""

import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = root / "SHA256SUMS"
expected = {}
for line in manifest.read_text(encoding="utf-8").splitlines():
    digest, separator, name = line.partition("  ")
    if separator != "  " or len(digest) != 64 or name in expected or not name:
        raise SystemExit("Invalid release manifest")
    expected[name] = digest
actual = {}
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("Linked release path rejected")
    if not path.is_file():
        continue
    name = path.relative_to(root).as_posix()
    if name == "SHA256SUMS" or name.startswith(".venv/") or "__pycache__" in path.parts:
        continue
    actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    print("Release integrity failed: missing, extra, or digest mismatch", file=sys.stderr)
    raise SystemExit(2)
print(f"Verified {len(actual)} packaged files.")
