"""Bounded read-only source snapshot, no numerical scientific artifact access."""

import hashlib
import json
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
BUILDER = EVIDENCE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
SOURCES = (
    "models/native_temporal.py",
    "models/sigreg.py",
    "training/native_ssl.py",
    "training/native_downstream.py",
    "training/native_resources.py",
    "training/aeon_corpus.py",
    "data/native_ssl_corpus.py",
    "inference/native_encoder.py",
    "inference/native_acoustic.py",
    "evaluation/native_product.py",
    "__init__.py",
    "models/__init__.py",
    "training/__init__.py",
    "evaluation/__init__.py",
    "data/__init__.py",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    if sys.argv[1:] == ["--source-json"]:
        print(
            json.dumps(
                {
                    name: (EVIDENCE / (name.replace("/", "__") + ".before.txt")).read_text(
                        encoding="utf-8"
                    )
                    for name in SOURCES
                }
            )
        )
    else:
        records = {}
        for name in SOURCES:
            source = MAIN / "src/marine_echo" / name
            payload = source.read_bytes()
            snapshot = EVIDENCE / (name.replace("/", "__") + ".before.txt")
            if snapshot.exists():
                # Partial first snapshot stopped at an absent namespace __init__.
                # Preserve already-authored bytes and only check them read-only.
                assert snapshot.read_bytes() == payload
            else:
                with snapshot.open("xb") as stream:
                    stream.write(payload)
            records[str(source)] = {"sha256": sha(payload), "snapshot": snapshot.name}
        comparison = BUILDER / "evidence/ssl-comparison-builder-v1"
        closed = {
            str(p.relative_to(BUILDER)): sha(p.read_bytes())
            for p in sorted(comparison.iterdir())
            if p.is_file()
        }
        for name in (
            "src/marine_echo/evaluation/native_comparison.py",
            "tests/unit/test_native_comparison.py",
        ):
            closed[name] = sha((BUILDER / name).read_bytes())
        result = {
            "operation": "SOURCE_AND_CLOSED_HANDOFF_PRESERVATION_ONLY",
            "main_sources": records,
            "closed_comparison": closed,
        }
        with (EVIDENCE / "before.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(
            json.dumps(
                {"main_source_count": len(records), "closed_comparison_file_count": len(closed)}
            )
        )
