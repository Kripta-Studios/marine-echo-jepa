"""Retain late closed builder reports without rewriting integrated evidence."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    source = BUILDER / "evidence/ssl-platform-builder-v1"
    destination = ROOT / "evidence/ssl-platform-builder-v1"
    copied, preserved = [], []
    for original in sorted(source.rglob("*")):
        if not original.is_file() or "__pycache__" in original.parts:
            continue
        target = destination / original.relative_to(source)
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                preserved.append(str(target.relative_to(ROOT)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(original.read_bytes())
        copied.append(str(target.relative_to(ROOT)))
    report = {
        "new_closed_delivery_files": copied,
        "differing_originals_preserved_without_overwrite": preserved,
        "active_fit_source_mutations": 0,
    }
    with (ROOT / "evidence/ssl-research-v1/platform-handoff-completion.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
