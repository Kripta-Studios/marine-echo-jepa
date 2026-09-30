"""Read-only closed source snapshot/hash verification; no numerical artifacts."""

import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
CODE = [
    "src/marine_echo/training/native_band_replication_ssl.py",
    "src/marine_echo/training/native_band_replication_downstream.py",
    "src/marine_echo/inference/native_band_replication_acoustic.py",
    "src/marine_echo/inference/native_band_replication_encoder.py",
    "src/marine_echo/inference/native_band_replication_latent.py",
    "tools/execute_native_band_replication_job.py",
    "tools/prepare_native_band_replication_configs.py",
]
TESTS = [
    str(p.relative_to(BUILDER)).replace("\\", "/")
    for folder in ("unit", "integration")
    for p in sorted((BUILDER / "tests" / folder).glob("test_native_band_replication*.py"))
]
proof = json.loads((HERE / "source-proof-final-v2b.json").read_bytes())
for path, digest in proof["source_closure_sha256"].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("Closed project source changed: " + path)
mode = sys.argv[1]
if mode in ("code", "tests"):
    paths = CODE if mode == "code" else TESTS
    print(
        json.dumps(
            {
                p: {
                    "sha256": hashlib.sha256((BUILDER / p).read_bytes()).hexdigest(),
                    "text": (BUILDER / p).read_text(encoding="utf-8"),
                }
                for p in paths
            }
        )
    )
elif mode == "hashes":
    print(
        json.dumps(
            {
                "authored_sha256": {
                    p: hashlib.sha256((BUILDER / p).read_bytes()).hexdigest() for p in CODE + TESTS
                },
                "evidence_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(HERE.iterdir())
                    if p.is_file()
                },
                "protected_source_closure_verified": True,
            },
            indent=2,
        )
    )
elif mode == "v1bytes":
    original = json.loads((HERE / "v1-source-snapshot.json").read_bytes())
    records = {}
    for path, record in original.items():
        raw = (MAIN / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Original version1 source changed.")
        records[path] = {
            "sha256": record["sha256"],
            "utf8_text_snapshot_byte_exact": record["text"].encode("utf-8") == raw,
        }
        if not records[path]["utf8_text_snapshot_byte_exact"]:
            records[path]["original_bytes_base64"] = base64.b64encode(raw).decode("ascii")
    print(json.dumps(records, indent=2))
else:
    raise ValueError("Explicit bounded source snapshot/hash operation required.")
