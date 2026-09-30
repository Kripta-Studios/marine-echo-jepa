"""Source-only bounded prefix delivery proof; no model/corpus decoding."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = (
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous_path = BUILDER / "evidence/ssl-assessment-builder-v1/source-proof-final-v2.json"
    previous = json.loads(previous_path.read_bytes())
    protected = previous["protected_main_sha256"]
    assert all(digest(Path(p)) == h for p, h in protected.items())
    assert all(digest(BUILDER / p) == h for p, h in previous["authored_sha256"].items())
    root = MAIN / "src/marine_echo"
    pending = [BUILDER / AUTHORED[0]]
    sources = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in sources:
            continue
        sources[str(path)] = digest(path)
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                names = [name, *(name + "." + v.name for v in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    tree = ast.parse((BUILDER / AUTHORED[0]).read_bytes())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = (
        "initialize_model",
        "Scalers.fit",
        "shared_loss",
        "masked_loss",
        "cf_loss",
        "subprocess",
        "os.system",
        "exec",
        "eval",
        "import_module",
    )
    assert not any(
        any(v in c for v in forbidden if v not in ("eval", "exec")) or c in ("eval", "exec")
        for c in calls
    )
    loads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) in ("np.load", "torch.load")
    ]
    assert all(
        any(
            k.arg == ("allow_pickle" if ast.unparse(n.func) == "np.load" else "weights_only")
            and isinstance(k.value, ast.Constant)
            and k.value.value is (ast.unparse(n.func) != "np.load")
            for k in n.keywords
        )
        for n in loads
    )
    print(
        json.dumps(
            {
                "evidence_kind": "SOURCE_ONLY_PROOF_NOT_SCIENTIFIC_APPROVAL",
                "authored_sha256": {p: digest(BUILDER / p) for p in AUTHORED},
                "source_bindings": sources,
                "protected_main_sha256": protected,
                "protected_main_unchanged": True,
                "closed_assessment_three_sources_unchanged": True,
                "previous_proof_sha256": digest(previous_path),
                "protocol_sha256": digest(
                    MAIN / "docs/adr/0021-native-prefix-transfer-assessment.md"
                ),
                "source_only_no_initialization_or_fit_executed": True,
                "static_forbidden_call_check": True,
                "safe_numeric_and_tensor_codec_flags": True,
                "write_scope": [*AUTHORED, "evidence/ssl-prefix-builder-v1/**"],
                "public_data_access": False,
                "gpu_execution": False,
                "git_index": "NOT_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
