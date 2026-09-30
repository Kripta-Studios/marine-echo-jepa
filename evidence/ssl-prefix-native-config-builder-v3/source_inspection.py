"""Read-only source snapshots; no corpus or checkpoint access."""

import hashlib
import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
PATHS = (
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
)
if sys.argv[1] == "snapshot":
    files = {}
    for name in PATHS:
        raw = (BUILDER / name).read_bytes()
        if raw != (MAIN / name).read_bytes():
            raise ValueError("Current builder and MAIN prefix bytes differ.")
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}
    old = BUILDER / "evidence/ssl-prefix-completion-builder-v2"
    historical = {
        str(p.relative_to(BUILDER)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(old.iterdir())
        if p.is_file()
    }
    print(
        json.dumps(
            {
                "kind": "native_prefix_v3_before_snapshot",
                "files": files,
                "closed_v2_evidence_sha256": historical,
                "main_bytes_equal": True,
            }
        )
    )
elif sys.argv[1] == "reader":
    p = MAIN / "src/marine_echo/data/native_ssl_corpus.py"
    print(
        "".join(
            p.read_text(encoding="utf-8").splitlines(keepends=True)[
                int(sys.argv[2]) : int(sys.argv[3])
            ]
        )
    )
elif sys.argv[1] == "executor":
    p = BUILDER / PATHS[0]
    print(
        "".join(
            p.read_text(encoding="utf-8").splitlines(keepends=True)[
                int(sys.argv[2]) : int(sys.argv[3])
            ]
        )
    )
elif sys.argv[1] == "fixture":
    raw = (
        BUILDER / "evidence/ssl-prefix-completion-builder-v2/prefix_test_support.py"
    ).read_bytes()
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}))
else:
    raise ValueError("Explicit source-only operation required.")
