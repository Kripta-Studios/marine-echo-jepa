"""Read-only final patch/source identity checks; no corpus, fit or GPU access."""

import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
EVIDENCE = Path(__file__).parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    manifest = json.loads((EVIDENCE / "patch-manifest.json").read_text(encoding="utf-8"))
    main_states = []
    for change in manifest["changes"]:
        current = digest(MAIN / change["target"])
        assert current in (change["main_before_sha256"], change["after_sha256"])
        main_states.append(
            {
                "target": change["target"],
                "current_sha256": current,
                "state": "EXACT_PROPOSED_AFTER_BYTES_OBSERVED"
                if current == change["after_sha256"]
                else "ORIGINAL_MAIN_BEFORE_BYTES",
            }
        )
        assert digest(EVIDENCE / change["patch"]) == change["patch_sha256"]
        assert (
            digest(EVIDENCE / (Path(change["target"]).stem + ".after.py")) == change["after_sha256"]
        )
    assert (
        digest(BUILDER / "src/marine_echo/models/native_temporal.py")
        == manifest["changes"][0]["after_sha256"]
    )
    paths = [
        BUILDER / "src/marine_echo/models/native_temporal.py",
        BUILDER / "tests/unit/test_native_cf_pooling.py",
        BUILDER / "tests/unit/test_native_booster_serialization.py",
        *EVIDENCE.glob("*.py"),
    ]
    print(
        json.dumps(
            {
                "status": "IDENTITIES_VERIFIED_MAIN_READ_ONLY_TO_BUILDER",
                "main_states": main_states,
                "evidence_kind": "SOURCE_INSPECTION_ONLY",
                "changes": manifest["changes"],
                "source_sha256": {str(p.resolve()): digest(p) for p in sorted(set(paths))},
            },
            indent=2,
        )
    )
