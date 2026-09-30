"""Close proof after a protected snapshot-name collision; never overwrite it."""

import ast
import base64
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = [
    "src/marine_echo/training/native_desktop_runtime_v3.py",
    "src/marine_echo/training/native_prefix_desktop_transfer_v3.py",
    "tools/execute_native_prefix_desktop_job_v3.py",
    "tools/execute_native_prefix_desktop_worker_v3.py",
    "tests/unit/test_native_prefix_desktop_execution_v3.py",
    "tests/integration/test_native_prefix_desktop_transfer_v3.py",
]


def sha(raw):
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
        routes = {
            "tools/execute_native_prefix_desktop_worker_v3.py": "tools/execute_native_prefix_worker.py",
            "src/marine_echo/training/native_prefix_desktop_transfer_v3.py": "src/marine_echo/training/native_prefix_transfer.py",
        }
        if isinstance(node.value, str) and node.value in routes:
            node.value = routes[node.value]
        return node

    def visit_ImportFrom(self, node):
        if node.module == "marine_echo.training.native_prefix_desktop_transfer_v3":
            node.module = "marine_echo.training.native_prefix_transfer"
        return node


protected = {}
for i, (name, prior) in enumerate(json.loads((HERE / "protected-baseline-v3.json").read_bytes()).items()):
    path = Path(name)
    raw = path.read_bytes()
    assert sha(raw) == prior["sha256"]
    assert raw == base64.b64decode(prior["bytes_base64"])
    assert raw == (HERE / f"protected-source-{i:02d}-{path.name}").read_bytes()
    protected[name] = {"baseline_sha256": prior["sha256"], "current_sha256": sha(raw), "unchanged": True}
authored = {}
for i, name in enumerate(AUTHORED):
    raw = (BUILDER / name).read_bytes()
    assert raw == (HERE / f"authored-source-{i:02d}-{Path(name).name}").read_bytes()
    authored[name] = {"sha256": sha(raw), "bytes": len(raw), "bytes_base64": base64.b64encode(raw).decode()}

old = definitions(MAIN / "src/marine_echo/training/native_prefix_transfer.py")
new = definitions(BUILDER / AUTHORED[1])
assert old.keys() == new.keys()
equivalence = {}
for name in old:
    if name == "required_sources":
        equivalence[name] = "CHANGED_OPERATIONAL_SOURCE_CLOSURE_ONLY"
        continue
    assert same(old[name], ResourceRoute().visit(copy.deepcopy(new[name]))), name
    equivalence[name] = "EXACT_AST" if same(old[name], new[name]) else "EXACT_AST_AFTER_RESOURCES_ROUTE_ONLY"
assert same(definitions(MAIN / "src/marine_echo/training/native_ssl.py")["Resources"], definitions(BUILDER / AUTHORED[0])["Resources"])
assert same(ast.parse((MAIN / "tools/execute_native_prefix_worker.py").read_bytes()), SourceRoutes().visit(ast.parse((BUILDER / AUTHORED[3]).read_bytes())))
before = definitions(MAIN / "tools/execute_native_prefix_job_v2.py")
after = definitions(BUILDER / AUTHORED[2])
for name in before:
    if name != "main":
        assert same(before[name], after[name]), name
normalized = SourceRoutes().visit(copy.deepcopy(after["main"]))
removed = []
for node in list(normalized.body):
    if isinstance(node, ast.ImportFrom) and node.module == "marine_echo.training.native_desktop_runtime_v3":
        normalized.body.remove(node)
        removed.append("bound_runtime_import")
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id in {"validate_operational_authority", "validate_idle_desktop_ledger"}:
        normalized.body.remove(node)
        removed.append(node.value.func.id)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in node.targets):
        old_assignment = next(n for n in before["main"].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in n.targets))
        assert len(node.value.elts) == len(old_assignment.value.elts) + 11
        assert same(ast.List(elts=node.value.elts[:len(old_assignment.value.elts)], ctx=ast.Load()), old_assignment.value)
        node.value = copy.deepcopy(old_assignment.value)
        removed.append("11_explicit_operational_bindings")
assert set(removed) == {"bound_runtime_import", "validate_operational_authority", "validate_idle_desktop_ledger", "11_explicit_operational_bindings"}
assert same(before["main"], normalized)

fixture_dependencies = {}
for label, relative in (
    ("unit", "tests/unit/test_native_prefix_transfer.py"),
    ("integration", "tests/integration/test_native_prefix_transfer.py"),
    ("support", "evidence/ssl-prefix-native-config-builder-v3/prefix_test_support.py"),
):
    raw = (MAIN / relative).read_bytes()
    fixture_dependencies[relative] = sha(raw)
    if label == "unit":
        assert raw == (HERE / "fixture-dependency-test_native_prefix_transfer.py").read_bytes()
    else:
        with (HERE / f"fixture-{label}-source.py").open("xb") as stream:
            stream.write(raw)
save("root-fixture-dependencies-v3.json", fixture_dependencies)

sys.path.insert(0, str(MAIN / "src"))
package = importlib.import_module("marine_echo.training")
package.__path__.append(str(BUILDER / "src/marine_echo/training"))
candidate = importlib.import_module("marine_echo.training.native_prefix_desktop_transfer_v3")
closure = {str(p): sha(p.read_bytes()) for p in candidate.required_sources()}
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
    "wrapper_main_ast": "EXACT_AFTER_DOCUMENTED_OPERATIONAL_ROUTES_BINDINGS_AND_IDLE_GUARDS_ONLY",
    "wrapper_main_documented_changes": removed,
    "numerical_science_changes": False,
    "public_numeric_or_weights_access": False,
    "preserved_snapshot_collision_log": "source-proof-final-v3.log",
})
print(json.dumps({"status": "BYTE_AND_AST_PROOF_PASSED", "protected": len(protected), "authored": len(authored), "source_closure": len(closure)}))
