"""Standard-library source preservation/AST proof; never parse numerical artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = (
    "src/marine_echo/evaluation/native_assessment.py",
    "tests/unit/test_native_assessment.py",
    "tests/integration/test_native_assessment.py",
)
FOLDER = BUILDER / "evidence/ssl-assessment-builder-v1"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def protected():
    # Bounded source-only preservation check, not a dataset/history inventory.
    names = (
        "evaluation/native_product.py",
        "evaluation/native_comparison.py",
        "inference/native_acoustic.py",
        "inference/native_encoder.py",
        "inference/native_band_acoustic.py",
        "models/native_temporal.py",
        "models/native_band_temporal.py",
        "training/native_ssl.py",
        "training/native_downstream.py",
        "training/native_band_ssl.py",
        "training/native_band_downstream.py",
        "training/native_references.py",
    )
    return {str(MAIN / "src/marine_echo" / n): digest(MAIN / "src/marine_echo" / n) for n in names}


def write_new(p, value):
    with p.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--proof-name", default="source-proof-final-v2.json")
    args = parser.parse_args()
    before = FOLDER / "protected-sources-before-v1.json"
    if args.snapshot:
        write_new(before, {"main_source_sha256": protected()})
        print("Source-only preservation baseline written.")
        return
    current = protected()
    baseline = json.loads(before.read_text(encoding="utf-8"))["main_source_sha256"]
    assert current == baseline, "A read-only main source changed during verification."
    band = json.loads(
        (BUILDER / "evidence/ssl-band-builder-v1/ast-proof-final-v1.json").read_text(
            encoding="utf-8"
        )
    )["authored_sha256"]
    assert all(digest(BUILDER / p) == h for p, h in band.items()), "Closed band source changed."
    source = BUILDER / AUTHORED[0]
    tree = ast.parse(source.read_bytes())
    forbidden = {
        "fit",
        "backward",
        "initialize_model",
        "manual_seed",
        "seed",
        "compile",
        "system",
        "Popen",
        "run",
        "exec",
        "import_module",
    }
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden_calls = [
        getattr(n.func, "attr", getattr(n.func, "id", ""))
        for n in calls
        if getattr(n.func, "attr", getattr(n.func, "id", "")) in forbidden
    ]
    assert not forbidden_calls, forbidden_calls
    assert not any(isinstance(n.func, ast.Name) and n.func.id == "eval" for n in calls)
    loads = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
    assert len(loads) == 2, "Expected one NPZ and one owned Torch codec."
    flags = [
        {
            k.arg: ast.literal_eval(k.value)
            for k in n.keywords
            if k.arg in {"weights_only", "map_location", "allow_pickle"}
        }
        for n in loads
    ]
    assert {"weights_only": True, "map_location": "cpu"} in flags
    assert {"allow_pickle": False} in flags
    proof = {
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "authored_sha256": {p: digest(BUILDER / p) for p in AUTHORED},
        "protected_main_sha256": current,
        "protected_main_unchanged": True,
        "closed_band_eight_sources_unchanged": True,
        "no_fit_optimizer_seed_subprocess_or_provenance_execution_calls": True,
        "codec_flags": flags,
        "write_scope": [*AUTHORED, "evidence/ssl-assessment-builder-v1/**"],
        "scientific_execution": "NOT_RUN_NOT_AUTHORIZED",
        "gpu_execution": "NOT_RUN_NOT_AUTHORIZED",
        "git_index": "NOT_RUN_NO_RETRY",
    }
    if (
        not args.proof_name.startswith("source-proof-")
        or Path(args.proof_name).name != args.proof_name
    ):
        raise ValueError("New source proof basename required.")
    write_new(FOLDER / args.proof_name, proof)
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
