"""Read-only dependency capture and exact static science proof; no data codecs."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

from test_support import MAIN, prefix_module
from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as suffix

prefix = prefix_module()
HERE = Path(__file__).resolve().parent
AUTHOR = [
    "src/marine_echo/training/native_prefix_matched_transfer_v4.py",
    "src/marine_echo/evaluation/native_prefix_suffix_assessment_v1.py",
    "tests/unit/test_native_prefix_matched_transfer_v4.py",
    "tests/integration/test_native_prefix_matched_transfer_v4.py",
    "tests/unit/test_native_prefix_suffix_assessment_v1.py",
    "tests/integration/test_native_prefix_suffix_assessment_v1.py",
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def exclusive_json(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


KIND_MAP = {
    "native_prefix_matched_transfer_resume_v4": "native_prefix_transfer_resume_v1",
    "native_prefix_matched_transfer_inference_v4": "native_prefix_transfer_inference_v1",
    "native_prefix_matched_transfer_completion_v4": "native_prefix_transfer_completion_v1",
}


def nodes(raw, normalize_kinds=False):
    tree = ast.parse(raw)
    if normalize_kinds:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = KIND_MAP.get(node.value, node.value)
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            values[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        values[node.name + "." + method.name] = ast.dump(method, include_attributes=False)
    return values


before = (HERE / "prefix-baseline-source.txt").read_bytes()
original = MAIN / "src/marine_echo/training/native_prefix_desktop_transfer_v3.py"
assert before == original.read_bytes(), "Immutable original prefix changed since clone"
current = Path(prefix.__file__).read_bytes()
old, new = nodes(before), nodes(current)
initial = json.loads((HERE / "prefix-copy-proof.json").read_text())
for name in initial["unchanged_ast"]:
    assert old[name] == new[name], "Original science AST changed: " + name
for name in ("_Trajectory.advance", "_Trajectory.__init__", "PrefixModel.forecast"):
    assert old[name] == new[name], "Numerical trajectory changed: " + name
normalized = nodes(current, normalize_kinds=True)
typed_only = ("_Trajectory.restore", "_Trajectory.checkpoint", "_Trajectory.inference_artifact")
for name in typed_only:
    assert old[name] == normalized[name], "More than typed artifact kind changed: " + name
exclusive_json("scientific-ast-proof-final-v1.json", {
    "original_path": str(original), "original_before_sha256": digest(before),
    "original_current_sha256": digest(original.read_bytes()), "new_sha256": digest(current),
    "exact_equal_ast": sorted(k for k in old.keys() & new.keys() if old[k] == new[k]),
    "changed_ast": sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
    "added_definitions": sorted(new.keys() - old.keys()),
    "unchanged_science_assertions": initial["unchanged_ast"] + ["_Trajectory.advance", "_Trajectory.__init__", "PrefixModel.forecast"],
    "artifact_kind_only_ast_equivalence": list(typed_only), "exact_kind_map": KIND_MAP,
    "changed_scope": "typed CF controls/scratch and source/ancestry/config admission; V4 kinds; additive source-pinned conventional policy/replay/resume; no original numerical trajectory modification",
    "runtime_source_rewriting": False, "original_module_global_patching": False,
})
with (HERE / "prefix-bounded-diff-final-v1.patch").open("x", encoding="utf-8") as stream:
    stream.writelines(difflib.unified_diff(before.decode().splitlines(True), current.decode().splitlines(True), fromfile=str(original), tofile=str(Path(prefix.__file__))))
dependencies = set(suffix.required_sources())
dependencies.update(MAIN / name for name in (
    "docs/NATIVE_SSL_COORDINATOR_CODE_REVIEW_V2.md", "docs/adr/0021-native-prefix-transfer-assessment.md",
    "docs/adr/0025-cf-matched-controls-continuation.md", "src/marine_echo/evaluation/native_assessment_replication.py",
    "evidence/ssl-prefix-native-config-builder-v3/prefix_test_support.py",
))
authored = {str(Path(prefix.__file__).parents[3] / name) for name in AUTHOR}
protected = sorted(p for p in dependencies if str(p) not in authored)
first = {str(p): digest(p.read_bytes()) for p in protected}
for index, p in enumerate(protected):
    with (HERE / ("protected-source-%02d%s" % (index, p.suffix or ".txt"))).open("xb") as stream:
        stream.write(p.read_bytes())
second = {str(p): digest(p.read_bytes()) for p in protected}
assert first == second
exclusive_json("protected-source-proof-final-v1.json", {
    "captured_baseline": first, "current": second, "protected_sources_unchanged": first == second,
    "original_prefix_clone_baseline_unchanged": before == original.read_bytes(),
    "snapshot_index": {str(p): "protected-source-%02d%s" % (index, p.suffix or ".txt") for index, p in enumerate(protected)},
    "scope": "byte-exact clone-start prefix proof and final double-read static dependency capture; no corpus/weights/forecasts decoded",
})
exclusive_json("authored-source-proof-final-v1.json", {name: digest((HERE.parents[1] / name).read_bytes()) for name in AUTHOR})
print(json.dumps({"exit": 0, "authored": len(AUTHOR), "protected": len(protected), "equal_AST_definitions": sum(old[k] == new[k] for k in old.keys() & new.keys())}))
