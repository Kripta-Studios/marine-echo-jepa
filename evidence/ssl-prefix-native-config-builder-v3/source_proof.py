"""Stdlib-only bounded hashes, historical preservation and recipe AST proof."""

import ast
import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
before = json.loads((HERE / "before-source-snapshot-v3.json").read_bytes())
old_proof = json.loads(
    (BUILDER / "evidence/ssl-prefix-completion-builder-v2/source-proof-final-v2.json").read_bytes()
)
protected = {}
for path in {
    *old_proof["protected_main_sha256"],
    *old_proof["runtime_package_sha256"],
    str(MAIN / "orchestration/ssl_vnext_prefix_native_configuration_followup.txt"),
}:
    p = Path(path)
    protected[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
for path, record in before["files"].items():
    if hashlib.sha256(record["text"].encode("utf-8")).hexdigest() != record["sha256"]:
        raise ValueError("Archived V2 source bytes changed.")
    if (MAIN / path).read_bytes() != record["text"].encode("utf-8"):
        raise ValueError("MAIN prefix source changed during builder implementation.")
for path, digest in before["closed_v2_evidence_sha256"].items():
    if hashlib.sha256((BUILDER / path).read_bytes()).hexdigest() != digest:
        raise ValueError("Closed V2 evidence changed.")
for path, digest in old_proof["other_closed_builder_sources_unchanged"].items():
    if hashlib.sha256((BUILDER / path).read_bytes()).hexdigest() != digest:
        raise ValueError("Other closed builder source changed.")
baseline = HERE / "protected-main-baseline-v3.json"
if baseline.exists() and json.loads(baseline.read_bytes())["protected_main_sha256"] != protected:
    raise ValueError("Protected MAIN source/protocol changed.")


def nodes(text):
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    result[node.name + "." + member.name] = ast.dump(
                        member, include_attributes=False
                    )
            if node.name == "PrefixConfig":
                result["PrefixConfig"] = ast.dump(node, include_attributes=False)
    return result


path = "src/marine_echo/training/native_prefix_transfer.py"
previous = nodes(before["files"][path]["text"])
current = nodes((BUILDER / path).read_bytes())
same = [
    *old_proof["unchanged_recipe_ast"],
    "PrefixConfig",
    "prepare_model",
    "selected_kind",
    "_supervised_ancestry",
    "_backbone",
    "_validate_backbone",
    "required_sources",
    "_Trajectory.checkpoint",
    "_Trajectory.restore",
    "_Trajectory.inference_artifact",
]
for name in same:
    if previous[name] != current[name]:
        raise ValueError("Recipe/model/ancestry implementation changed: " + name)
files = {
    p: {
        "sha256": hashlib.sha256((BUILDER / p).read_bytes()).hexdigest(),
        "text": (BUILDER / p).read_text(encoding="utf-8"),
    }
    for p in before["files"]
}
diff = "".join(
    "".join(
        difflib.unified_diff(
            before["files"][p]["text"].splitlines(keepends=True),
            record["text"].splitlines(keepends=True),
            fromfile="main/" + p,
            tofile="builder/" + p,
        )
    )
    for p, record in files.items()
)
result = {
    "kind": "native_prefix_native_configuration_source_proof_v3",
    "protected_main_sha256": protected,
    "closed_v2_evidence_unchanged": True,
    "closed_v2_evidence_files": len(before["closed_v2_evidence_sha256"]),
    "before_source_sha256": {p: r["sha256"] for p, r in before["files"].items()},
    "authored_sha256": {p: r["sha256"] for p, r in files.items()},
    "unchanged_recipe_ast": same,
    "public_numeric_access": "NOT_RUN",
}
mode = sys.argv[1] if len(sys.argv) > 1 else "proof"
if mode == "proof":
    print(json.dumps(result, indent=2))
elif mode == "diff":
    print(diff, end="")
elif mode == "final_sources":
    print(json.dumps(files))
else:
    raise ValueError("Explicit bounded source proof operation required.")
