"""Source-only protected snapshots and requested schema inspection."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
MAIN = BASE.parents[2] / "marine-echo-jepa"
START = (
    "tools/prepare_native_seed7_assessment_metadata.py",
    "src/marine_echo/evaluation/native_assessment_replication.py",
    "src/marine_echo/training/native_prefix_transfer.py",
    "src/marine_echo/data/native_transfer_corpus.py",
    "src/marine_echo/training/native_ssl.py",
    "src/marine_echo/training/native_downstream.py",
    "src/marine_echo/training/native_band_ssl.py",
    "src/marine_echo/training/native_band_downstream.py",
    "src/marine_echo/training/native_band_replication_ssl.py",
    "src/marine_echo/training/native_band_replication_downstream.py",
    "src/marine_echo/training/native_references.py",
    "orchestration/ssl_vnext_native_ancestry_inventory_builder_v1.txt",
    "configs/native_ssl_split_v1.json",
)
SLICES = {
    "src/marine_echo/evaluation/native_assessment_replication.py": [(413, 482), (827, 890)],
    "src/marine_echo/training/native_ssl.py": [(1005, 1105), (330, 380)],
    "src/marine_echo/training/native_downstream.py": [(695, 805), (842, 880)],
    "src/marine_echo/training/native_prefix_transfer.py": [(870, 940)],
    "src/marine_echo/training/native_references.py": [(30, 100), (390, 460)],
}


def main():
    pending, found = [MAIN / name for name in START], {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes_base64": base64.b64encode(raw).decode("ascii"),
        }
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + item.name for item in node.names]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            for name in names:
                if not name.startswith("marine_echo"):
                    continue
                parts = name.split(".")
                for n in range(1, len(parts) + 1):
                    candidate = MAIN / "src" / Path(*parts[:n])
                    for source in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if source.is_file():
                            pending.append(source)
    with (BASE / "protected-baseline-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(found, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SOURCE_ONLY", "protected_files": len(found)}))
    for name, ranges in SLICES.items():
        lines = (MAIN / name).read_text(encoding="utf-8").splitlines()
        for lo, hi in ranges:
            print(name, lo, hi)
            print("\n".join(f"{i + 1}: {lines[i]}" for i in range(lo - 1, min(hi, len(lines)))))


if __name__ == "__main__":
    main()
