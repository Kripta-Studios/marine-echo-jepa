"""Stdlib byte/AST proof plus truthful imported source closure; no numerical data."""

from __future__ import annotations

import ast
import base64
import copy
import difflib
import hashlib
import importlib
import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
HERE = Path(__file__).resolve().parent
AUTHORED = [
    "src/marine_echo/training/native_desktop_runtime_v3.py",
    "src/marine_echo/training/native_prefix_desktop_transfer_v3.py",
    "tools/execute_native_prefix_desktop_job_v3.py",
    "tools/execute_native_prefix_desktop_worker_v3.py",
    "tests/unit/test_native_prefix_desktop_execution_v3.py",
    "tests/integration/test_native_prefix_desktop_transfer_v3.py",
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def definitions(path):
    return {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def same(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


class ResourceRoute(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if node.attr == "Resources" and isinstance(node.value, ast.Name) and node.value.id == "desktop_runtime":
            node.value.id = "core"
        return node


class SourceRoutes(ast.NodeTransformer):
    def visit_Constant(self, node):
        replacements = {
            "tools/execute_native_prefix_desktop_worker_v3.py": "tools/execute_native_prefix_worker.py",
            "src/marine_echo/training/native_prefix_desktop_transfer_v3.py": "src/marine_echo/training/native_prefix_transfer.py",
        }
        if isinstance(node.value, str) and node.value in replacements:
            node.value = replacements[node.value]
        return node

    def visit_ImportFrom(self, node):
        if node.module == "marine_echo.training.native_prefix_desktop_transfer_v3":
            node.module = "marine_echo.training.native_prefix_transfer"
        return node


baseline = json.loads((HERE / "protected-baseline-v3.json").read_bytes())
protected = {}
for i, (name, snapshot) in enumerate(baseline.items()):
    path = Path(name)
    raw = path.read_bytes()
    assert digest(raw) == snapshot["sha256"], f"Protected source changed: {path}"
    assert base64.b64decode(snapshot["bytes_base64"]) == raw
    protected[name] = {"baseline_sha256": snapshot["sha256"], "current_sha256": digest(raw), "unchanged": True}
    with (HERE / f"protected-source-{i:02d}-{path.name}").open("xb") as stream:
        stream.write(raw)

authored = {}
for i, name in enumerate(AUTHORED):
    raw = (BUILDER / name).read_bytes()
    authored[name] = {"sha256": digest(raw), "bytes": len(raw), "bytes_base64": base64.b64encode(raw).decode()}
    with (HERE / f"authored-source-{i:02d}-{Path(name).name}").open("xb") as stream:
        stream.write(raw)

old = definitions(MAIN / "src/marine_echo/training/native_prefix_transfer.py")
new = definitions(BUILDER / AUTHORED[1])
assert old.keys() == new.keys()
equivalence = {}
for name in old:
    if name == "required_sources":
        equivalence[name] = "CHANGED_OPERATIONAL_SOURCE_CLOSURE_ONLY"
        continue
    after = ResourceRoute().visit(copy.deepcopy(new[name]))
    assert same(old[name], after), name
    equivalence[name] = "EXACT_AST" if same(old[name], new[name]) else "EXACT_AST_AFTER_RESOURCES_ROUTE_ONLY"
assert same(
    definitions(MAIN / "src/marine_echo/training/native_ssl.py")["Resources"],
    definitions(BUILDER / AUTHORED[0])["Resources"],
)
worker_old = ast.parse((MAIN / "tools/execute_native_prefix_worker.py").read_bytes())
worker_new = SourceRoutes().visit(ast.parse((BUILDER / AUTHORED[3]).read_bytes()))
assert same(worker_old, worker_new), "Worker differs beyond module route"

wrapper_old = definitions(MAIN / "tools/execute_native_prefix_job_v2.py")
wrapper_new = definitions(BUILDER / AUTHORED[2])
for name in wrapper_old:
    if name != "main":
        assert same(wrapper_old[name], wrapper_new[name]), name
normalized = SourceRoutes().visit(copy.deepcopy(wrapper_new["main"]))
removed = []
for node in list(normalized.body):
    if isinstance(node, ast.ImportFrom) and node.module == "marine_echo.training.native_desktop_runtime_v3":
        removed.append("new_bound_runtime_import")
        normalized.body.remove(node)
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
        if node.value.func.id in {"validate_operational_authority", "validate_idle_desktop_ledger"}:
            removed.append(node.value.func.id)
            normalized.body.remove(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in node.targets):
        before = next(n for n in wrapper_old["main"].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in n.targets))
        assert len(node.value.elts) == len(before.value.elts) + 11
        assert same(ast.List(elts=node.value.elts[:len(before.value.elts)], ctx=ast.Load()), before.value)
        removed.append("11_explicit_operational_source_bindings")
        node.value = copy.deepcopy(before.value)
assert set(removed) == {"new_bound_runtime_import", "validate_operational_authority", "validate_idle_desktop_ledger", "11_explicit_operational_source_bindings"}
assert same(wrapper_old["main"], normalized), "Unadmitted wrapper main change"

diffs = []
for source, target in (
    ("src/marine_echo/training/native_prefix_transfer.py", AUTHORED[1]),
    ("tools/execute_native_prefix_job_v2.py", AUTHORED[2]),
    ("tools/execute_native_prefix_worker.py", AUTHORED[3]),
):
    diffs.extend(difflib.unified_diff((MAIN/source).read_text(encoding="utf-8").splitlines(True), (BUILDER/target).read_text(encoding="utf-8").splitlines(True), fromfile="MAIN/"+source, tofile="BUILDER/"+target))
with (HERE / "bounded-original-to-desktop-v3.diff").open("x", encoding="utf-8") as stream:
    stream.writelines(diffs)

fixture_dependencies = {}
for relative in (
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
    "evidence/ssl-prefix-native-config-builder-v3/prefix_test_support.py",
):
    raw = (MAIN / relative).read_bytes()
    fixture_dependencies[relative] = digest(raw)
    with (HERE / ("fixture-dependency-" + Path(relative).name)).open("xb") as stream:
        stream.write(raw)
save("root-fixture-dependencies-v3.json", fixture_dependencies)

sys.path.insert(0, str(MAIN / "src"))
package = importlib.import_module("marine_echo.training")
package.__path__.append(str(BUILDER / "src/marine_echo/training"))
candidate = importlib.import_module("marine_echo.training.native_prefix_desktop_transfer_v3")
closure = {str(path): digest(path.read_bytes()) for path in candidate.required_sources()}
save("source-proof-final-v3.json", {
    "evidence_kind": "SOURCE_AND_SYNTHETIC_CORRECTNESS_ONLY",
    "authored": authored,
    "protected": protected,
    "protected_count": len(protected),
    "imported_source_closure": closure,
    "actual_original_core_file": candidate.core.__file__,
    "actual_new_prefix_file": candidate.__file__,
    "actual_operational_classifier_file": candidate.desktop_runtime.native_resources.__file__,
    "prefix_definition_ast_equivalence": equivalence,
    "resources_class_ast": "EXACT_ORIGINAL_NATIVE_SSL_RESOURCES_AST",
    "worker_ast": "EXACT_AFTER_MODULE_ROUTE_ONLY",
    "wrapper_old_functions_ast": "EXACT_EXCEPT_MAIN",
    "wrapper_main_ast": "EXACT_AFTER_DOCUMENTED_OPERATIONAL_ROUTE_BINDINGS_AND_IDLE_GUARDS_ONLY",
    "wrapper_main_documented_changes": removed,
    "numerical_science_changes": False,
    "public_numeric_or_weights_access": False,
})
print(json.dumps({"status": "BYTE_AND_AST_PROOF_PASSED", "protected": len(protected), "authored": len(authored), "source_closure": len(closure)}))
