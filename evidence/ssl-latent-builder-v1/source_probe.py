"""Read-only source/protected-delivery proof; never load data or checkpoints."""

import ast
import hashlib
import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
START = [
    "src/marine_echo/inference/native_encoder.py",
    "src/marine_echo/inference/native_band_acoustic.py",
    "src/marine_echo/inference/native_acoustic.py",
    "src/marine_echo/models/native_temporal.py",
    "src/marine_echo/models/native_band_temporal.py",
    "src/marine_echo/training/native_ssl.py",
    "src/marine_echo/training/native_band_ssl.py",
]


def proof():
    pending = [MAIN / p for p in START]
    found = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *(node.module + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    p = MAIN / "src" / Path(*name.split(".")).with_suffix(".py")
                    if p.is_file():
                        pending.append(p)
    found.update(
        MAIN / p
        for p in (
            "src/marine_echo/__init__.py",
            "src/marine_echo/inference/__init__.py",
            "src/marine_echo/models/__init__.py",
            "src/marine_echo/training/__init__.py",
            "src/marine_echo/data/native_ssl_corpus.py",
            "src/marine_echo/training/aeon_corpus.py",
            "pyproject.toml",
            "uv.lock",
        )
        if (MAIN / p).is_file()
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(found)}
    old = json.loads(
        (BUILDER / "evidence/ssl-band-execution-builder-v1/handoff-v1.json").read_bytes()
    )
    previous = {
        p: hashlib.sha256((BUILDER / p).read_bytes()).hexdigest() for p in old["authored_sha256"]
    }
    if previous != old["authored_sha256"]:
        raise ValueError("Closed execution sources changed.")
    return {
        "kind": "native_latent_source_proof_v1",
        "evidence_kind": "SOURCE_ONLY_NOT_SCIENTIFIC_APPROVAL",
        "protected_main_sha256": hashes,
        "closed_band_execution_sha256": previous,
        "public_numerical_access": "NOT_RUN",
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sections":
        for name, start, end in (
            ("src/marine_echo/training/native_ssl.py", 45, 152),
            ("src/marine_echo/training/native_ssl.py", 270, 304),
            ("src/marine_echo/training/native_ssl.py", 1010, 1040),
            ("src/marine_echo/training/native_band_ssl.py", 52, 170),
            ("src/marine_echo/training/native_band_ssl.py", 1137, 1175),
            ("src/marine_echo/inference/native_acoustic.py", 1, 150),
        ):
            print(name, start, end)
            print(
                "\n".join((MAIN / name).read_text(encoding="utf-8").splitlines()[start - 1 : end])
            )
    else:
        result = proof()
        if len(sys.argv) > 1:
            baseline = json.loads((BUILDER / sys.argv[1]).read_bytes())
            if (
                any(
                    result["protected_main_sha256"].get(p) != h
                    for p, h in baseline["protected_main_sha256"].items()
                )
                or result["closed_band_execution_sha256"]
                != baseline["closed_band_execution_sha256"]
            ):
                raise ValueError("Protected source proof mismatch.")
        print(json.dumps(result, indent=2))
