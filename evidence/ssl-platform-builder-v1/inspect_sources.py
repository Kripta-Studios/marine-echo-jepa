"""Read-only platform source inspection; no corpus, optimizer, or GPU access."""

import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
EXPECTED = "85c4af6c67acb661359cde8bcf22de479e4da6a499cc7843bce0fee6c155e422"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    model = Path("src/marine_echo/models/native_temporal.py")
    paths = [
        BUILDER / model,
        MAIN / model,
        MAIN / "src/marine_echo/training/native_references.py",
        MAIN / ".venv/Lib/site-packages/lightgbm/basic.py",
        MAIN / ".venv/Lib/site-packages/lightgbm/VERSION.txt",
    ]
    hashes = {str(p.resolve()): digest(p) for p in paths}
    print(
        json.dumps(
            {
                "evidence_kind": "SOURCE_INSPECTION_ONLY",
                "expected_main_model_sha256": EXPECTED,
                "builder_model_matches_main": hashes[str((BUILDER / model).resolve())]
                == hashes[str((MAIN / model).resolve())],
                "main_model_matches_expected": hashes[str((MAIN / model).resolve())] == EXPECTED,
                "lightgbm_version": paths[-1].read_text().strip(),
                "sha256": hashes,
            },
            indent=2,
        )
    )
