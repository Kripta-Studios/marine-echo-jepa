"""Create one fresh metadata-only audit wrapper without editing the failed version."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = ROOT / "tools/execute_closed_native_inventory_owned_v3a.py"
    raw = original.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "bbbdba09c20026e8a98d220fc6d7e0c28911911c1e7cb259a6c32fc16edb6539":
        raise ValueError("Exact original bounded owned audit wrapper required")
    target = original.with_name("execute_closed_native_inventory_owned_v3b.py")
    with target.open("xb") as stream:
        stream.write(raw.replace(b"v3a", b"v3b"))
    print("Fresh v3b paths; unchanged CPU ownership, deadlines, memory and cleanup gates.")


if __name__ == "__main__":
    main()
