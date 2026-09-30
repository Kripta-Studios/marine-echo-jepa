"""Bounded source/hash proof only; never parse corpus, weights, or actual ledger."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
NEW = (
    "tools/execute_native_band_job.py",
    "tools/prepare_native_band_configs.py",
    "tests/unit/test_native_band_execution.py",
)
PREFIX = {
    "src/marine_echo/training/native_prefix_transfer.py": "8d91d34ad1efe9ef9867587cf41eaa6448898ce1edf0b0719242b2a463ef8ddd",
    "tests/unit/test_native_prefix_transfer.py": "94ef28270fff35ba859770157dfbac49c96280868202abc8ea42013da14fb61c",
    "tests/integration/test_native_prefix_transfer.py": "f45c34bb5553413191b5894673f40cab416be0f05dac4195d1834c1a3f71ff43",
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protected():
    pending = [
        MAIN / "src/marine_echo/training/native_band_ssl.py",
        MAIN / "src/marine_echo/training/native_band_downstream.py",
    ]
    paths = set()
    while pending:
        p = pending.pop()
        if p in paths:
            continue
        paths.add(p)
        for node in ast.walk(ast.parse(p.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                names = [name, *(name + "." + a.name for a in node.names)]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = MAIN.joinpath("src", *name.split(".")).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    for name in (
        "tools/native_reference_supervisor.py",
        "tools/execute_native_ssl_job.py",
        "tools/execute_native_downstream_job.py",
        "orchestration/native_band_budget_owner_resolution_v1.json",
        "docs/adr/0023-owner-resolved-band-budget.md",
        "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        "docs/adr/0018-native-acoustic-downstream-transfer.md",
        "pyproject.toml",
        "uv.lock",
    ):
        paths.add(MAIN / name)
    for name in (
        "__init__.py",
        "models/__init__.py",
        "training/__init__.py",
        "data/__init__.py",
        "data/native_ssl_corpus.py",
        "training/aeon_corpus.py",
    ):
        paths.add(MAIN / "src/marine_echo" / name)
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    current = protected()
    prefix_unchanged = all(digest(BUILDER / p) == h for p, h in PREFIX.items())
    assert prefix_unchanged
    if args.baseline:
        baseline = json.loads(args.baseline.read_bytes())
        assert all(current[p] == h for p, h in baseline["protected_main_sha256"].items())
    calls = []
    for p in NEW[:2]:
        tree = ast.parse((BUILDER / p).read_bytes())
        calls.extend(ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call))
        # CLI argument lists intentionally have no root/synthetic execution switch.
        cli = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
        ]
        assert not any(
            any(
                isinstance(a, ast.Constant) and a.value in ("--root", "--correctness-smoke")
                for a in n.args
            )
            for n in cli
        )
    assert not any(
        c
        in (
            "torch.load",
            "np.load",
            "numpy.load",
            "torch.cuda.is_available",
            "subprocess.run",
            "subprocess.check_output",
        )
        for c in calls
    )
    result = {
        "kind": "native_band_execution_source_proof_v1",
        "evidence_kind": "SOURCE_ONLY_NOT_SCIENTIFIC_APPROVAL",
        "authored_sha256": {p: digest(BUILDER / p) for p in NEW},
        "protected_main_sha256": current,
        "protected_main_unchanged_since_baseline": bool(args.baseline),
        "closed_prefix_three_sources_unchanged": prefix_unchanged,
        "closed_prefix_sha256": PREFIX,
        "no_numeric_tensor_decode_or_model_gpu_git_calls": True,
        "no_root_or_synthetic_production_cli_switch": True,
        "allowed_writes": [*NEW, "evidence/ssl-band-execution-builder-v1/**"],
        "real_execution": "NOT_RUN",
        "gpu_execution": "NOT_RUN",
        "git_index": "NOT_RUN",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
