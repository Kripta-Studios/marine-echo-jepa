"""Compare current non-CF definitions with the original shared source snapshot."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nodes(path):
    return ast.parse(path.read_text(encoding="utf-8")).body


def main():
    original = ROOT / "evidence/ssl-research-v1/ancestor-source/native_temporal_shared_screen_v1.py"
    current = ROOT / "src/marine_echo/models/native_temporal.py"
    old = [node for node in nodes(original) if getattr(node, "name", None) != "CFNativeModel"]
    new = [
        node
        for node in nodes(current)
        if getattr(node, "name", None) not in ("CFNativeModel", "deterministic_adaptive_avg_pool1d")
    ]
    if [ast.dump(node, include_attributes=False) for node in old] != [
        ast.dump(node, include_attributes=False) for node in new
    ]:
        raise ValueError("A definition outside CF changed; do not assume shared compatibility.")
    report = {
        "status": "MATCHED_SHARED_AST_NOT_INDEPENDENT_APPROVAL",
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "current_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "unchanged_non_cf_definition_count": len(old),
        "unchanged_classes": [node.name for node in old if isinstance(node, ast.ClassDef)],
        "allowed_delta": "CFNativeModel pooling calls and new deterministic pooling helper",
        "scientific_fitting": "NOT_RUN",
    }
    with (ROOT / "evidence/ssl-research-v1/shared-source-compatibility.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
