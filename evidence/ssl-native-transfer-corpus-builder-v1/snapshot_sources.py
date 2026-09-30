"""Archive source bytes and transitive local imports; no corpus decoding."""

import ast
import base64
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2] / "marine-echo-jepa"
START = (
    "src/marine_echo/data/native_ssl_corpus.py",
    "src/marine_echo/training/native_prefix_transfer.py",
    "src/marine_echo/evaluation/native_assessment.py",
    "tools/native_reference_supervisor.py",
    "orchestration/ssl_vnext_native_transfer_corpus_builder_v1.txt",
)


def collect():
    pending = [ROOT / value for value in START]
    found = {}
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
                names = [node.module] + [node.module + "." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if not name.startswith("marine_echo"):
                    continue
                parts = name.split(".")
                for index in range(1, len(parts) + 1):
                    candidate = ROOT / "src" / Path(*parts[:index])
                    for source in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                        if source.is_file():
                            pending.append(source)
    return found


if __name__ == "__main__":
    snapshot = collect()
    with (BASE / "baseline-source-snapshots-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "SOURCE_ONLY",
                "files": len(snapshot),
                "hashes": {path: value["sha256"] for path, value in snapshot.items()},
            },
            indent=2,
        )
    )
