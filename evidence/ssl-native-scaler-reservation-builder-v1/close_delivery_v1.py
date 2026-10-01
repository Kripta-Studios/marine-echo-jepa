"""Close metadata-only hashes/AST proof; never decode models or acoustic arrays."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
AUTHORED = [
    "tools/native_train_scaler_reservation_v1.py",
    "tools/prepare_native_development_comparison_v5.py",
    "tools/prepare_completed_native_inventory_v6.py",
    "tools/execute_completed_native_inventory_owned_v6.py",
    "tests/unit/test_native_train_scaler_reservation_v1.py",
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("xb") as stream:
        stream.write(value)


def write_json(path, value):
    save(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("_v6", "_v4").replace("-v6", "-v4").replace("_v5", "_v4").replace("-v5", "-v4")
        return node

    def visit_Name(self, node):
        if node.id == "ROOT":
            node.id = "root"
        return node


def encoded(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


old_inventory = ast.parse((HERE / "baseline-prepare_completed_native_inventory_v4.py").read_bytes())
new_inventory = ast.parse((BUILDER / AUTHORED[2]).read_bytes())
old_owned = ast.parse((HERE / "baseline-execute_completed_native_inventory_owned_v4.py").read_bytes())
new_owned = ast.parse((BUILDER / AUTHORED[3]).read_bytes())
old_dev = ast.parse((HERE / "baseline-prepare_native_development_comparison_v4.py").read_bytes())
new_dev = ast.parse((BUILDER / AUTHORED[1]).read_bytes())
unchanged = ["digest", "regular", "json_bytes", "document", "fresh", "idle_ledger", "reservation_path", "declared_kind"]
for name in unchanged:
    assert encoded(function(old_inventory, name)) == encoded(function(new_inventory, name)), name
owned = copy.deepcopy(function(new_owned, "execute"))
required = next(n for n in owned.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "required" for t in n.targets))
removed = [n for n in required.value.elts if "native_train_scaler_reservation_v1.py" in ast.unparse(n)]
assert len(removed) == 1
required.value.elts = [n for n in required.value.elts if n not in removed]
assert encoded(owned) == encoded(function(old_owned, "execute")), "Owned lifecycle changed outside explicit helper binding/version names"
guards = {}
for label, old, new in (("inventory", function(old_inventory, "prepare"), function(new_inventory, "prepare")), ("development", function(old_dev, "main"), function(new_dev, "prepare"))):
    nodes = [n for n in ast.walk(old) if isinstance(n, ast.If)]
    new_nodes = {encoded(n) for n in ast.walk(new) if isinstance(n, ast.If)}
    excluded = [n for n in nodes if "Only original strong endpoints may reuse original TRAIN scalers" in ast.unparse(n)]
    assert len(excluded) == (2 if label == "inventory" else 0)
    admitted = [n for n in nodes if n not in excluded]
    assert all(encoded(n) in new_nodes for n in admitted), label + " original admission guard changed"
    guards[label] = {"original_if_guards_preserved": len(admitted), "replaced_old_scaler_restriction_if_nodes": len(excluded)}
baseline = json.loads((HERE / "protected-baseline-v1.json").read_bytes())
current = {path: sha(path) for path in baseline["files"]}
assert current == baseline["files"], "Protected scientific/source/archive dependency changed"
hashes = {path: sha(BUILDER / path) for path in AUTHORED}
for path in AUTHORED:
    snapshot = HERE / ("closed-" + Path(path).name)
    save(snapshot, (BUILDER / path).read_bytes())
    assert sha(snapshot) == hashes[path]
diffs = []
for old, new in (("prepare_native_development_comparison_v4.py", "prepare_native_development_comparison_v5.py"), ("prepare_completed_native_inventory_v4.py", "prepare_completed_native_inventory_v6.py"), ("execute_completed_native_inventory_owned_v4.py", "execute_completed_native_inventory_owned_v6.py")):
    diffs.extend(difflib.unified_diff((MAIN / "tools" / old).read_text(encoding="utf-8").splitlines(True), (BUILDER / "tools" / new).read_text(encoding="utf-8").splitlines(True), fromfile="MAIN/tools/" + old, tofile="BUILDER/tools/" + new))
save(HERE / "bounded-version-diff-v1.patch", "".join(diffs).encode())

# Metadata byte identities only; no parsing public model/scaler/corpus values.
metadata_pins = {
    str(MAIN / "outputs/native_acoustic_ssl_v1/shared_ssl_seed7_h96_cuda0/scalers.json"): "5783edf9c57f0d5b402bcd1ed72317e049e19550beebd38708a554994395d9ed",
    str(MAIN / "outputs/native_acoustic_ssl_v1/band_direct_end_to_end_seed23_h96_replication_v2/run.json"): "26da50474786a4c6ea4c9d593575dc6608e811872f6022fa6101dbe3f0ff4e55",
}
assert all(sha(path) == value for path, value in metadata_pins.items()), "Owner-named metadata byte identity changed"
proof = {
    "kind": "native_train_scaler_reservation_source_proof_v1",
    "authored": hashes,
    "protected_sources_unchanged": True,
    "protected_baseline_sha256": sha(HERE / "protected-baseline-v1.json"),
    "protected_current_dependencies": current,
    "owner_named_metadata_pins_checked_by_hash_only": metadata_pins,
    "unaltered_inventory_functions_exact_ast": unchanged,
    "owned_execute_exact_ast_except_version_names_and_required_helper_binding": True,
    "original_admission_guard_ast": guards,
    "intended_semantic_changes": ["Shared explicit TRAIN scaler reservation for typed completed downstream endpoints", "Pinned own/aliased TRAIN JSON identity and both saved tensor bindings", "V5/V6 fresh metadata names and current source closure", "Development named-reference/role/idle/safe-path guards"],
    "scientific_sources_modified": [],
    "tensor_or_corpus_decoding": False,
    "embedded_scaler_state_equality": "NOT_ESTABLISHED_EXISTING_ROOT_BOUNDED_TENSOR_AUDIT_REQUIRED",
    "approval": False,
}
write_json(HERE / "source-proof-final-v1.json", proof)

base = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest tests\unit\test_native_train_scaler_reservation_v1.py -q --import-mode=importlib -p no:cacheprovider"
paths_cli = " ".join(path.replace("/", "\\") for path in AUTHORED)
ruff = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff "
checks = [
    {"command": base, "exit_code": 1, "witness": "90ba8d", "log": "red-missing-band-scalers-01.log", "result": "1 genuine failure at the preserved original/Band scaler restriction"},
    {"command": base, "exit_code": 1, "witness": "49b6d7", "log": "green-band-reservation-01.log", "result": "Implementation accepted alias; private test used POSIX path suffix on Windows, corrected to Path.parts"},
    {"command": base, "exit_code": 0, "witness": "6185fd", "log": "green-helper-and-inventory-02.log"},
    {"command": base, "exit_code": 0, "witness": "b142a6", "log": "green-all-routes-03.log"},
    {"command": base, "exit_code": 0, "witness": "ba9b8a", "log": "green-complete-contract-04.log", "result": "58 passed in 7.59s"},
    {"command": base, "exit_code": 0, "witness": "17385d", "log": "green-final-05.log", "result": "58 passed in 7.33s"},
    {"command": ruff + "check --no-cache " + paths_cli, "exit_code": 1, "witness": "88e4cf", "log": "lint-check-01.log", "result": "Import formatting and combined-with style; corrected"},
    {"command": ruff + "format --no-cache " + paths_cli, "exit_code": 0, "witness": "26ff3f", "log": "format-01.log"},
    {"command": ruff + "check --no-cache " + paths_cli, "exit_code": 0, "witness": "f3b8a1", "log": "lint-check-02.log", "result": "All checks passed"},
    {"command": ruff + "format --check --no-cache " + paths_cli, "exit_code": 0, "witness": "f66458", "log": "lint-format-final-02.log"},
]
for check in checks:
    check["log_sha256"] = sha(HERE / check["log"])
write_json(HERE / "executed-checks-v1.json", {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "checks": checks, "physical_codecs": "Private JSON/opaque-byte fixtures only", "virtual_owned_wrapper_tests": "Mock child/supervisor, not actual processes/resources/cleanup", "actual_production_execution": "NOT_RUN"})
report = """# TRAIN scaler reservation metadata tooling V1

Closed five new source/test files. Historical outputs and V4 tools are unchanged.
The shared helper reserves regular safe scaler bytes and validates completed,
typed original/Band V1/Band V2 method/seed/mode/source identities. Core artifacts
must have their own JSON. Downstream artifacts without that duplicate may only
alias the pinned original Shared7 TRAIN JSON. Own and aliased bytes must match
the owner pin. No duplicate scaler file is written to historical outputs.

Receipts explicitly identify the proposed alias, producer, method/seed/mode,
endpoint and both selected/inference byte bindings. They confer no equality,
access or approval. The unchanged current ancestry decoder reads scalers_path
and must compare both embedded saved scaler states during a later bounded audit.
The helper does not load tensors, scaler values or corpora.

Development V5 preserves all 28 original bindings and reserves all 19 exact Band
endpoints: 47 methods/43 neural plus four named references. Inventory V6 retains
all completed review/config/source/parent guards and the immutable ledger capture.
Proposed scaler provenance lives in admission/reservation receipts, preserving
the existing audit manifest/entry schema. References remain a separate schema,
with no neural loading or clean-local Chronos ancestry claim. No40 substitution.
Missing three outcomes still prevents production preparation.

The V6 owned executor retains the original supervisor/lifecycle AST except its
versioned names and mandatory helper hash binding: CPU, <=600 seconds, tree RAM
below22GiB, hard live-idle/owner-lock/pending/reconciliation gates, fresh outputs,
actual exit/cleanup/report checks and failure receipts. No generic retry added.

Actual builder verification: focused red exit1 reproduced the Band missing-file
restriction; final pytest exit0, 58 passed in7.33s; Ruff check/format check exit0.
Tests use explicitly labeled private synthetic opaque artifacts and metadata;
they actually call both preparers/shared helper and the virtual owned wrapper.
They exercise all three producer families and direct/frozen/full aliases,
core refusal, wrong bytes/paths/kinds/config identities, both tensor bindings,
unsafe path simulation without links, original28/four-reference and43/47 guards,
missing3, immutable capture, parent/ownership and source-binding refusals.
Virtual process/resource evidence is not actual process execution.

The AST proof retains original inventory helper functions and admission guards,
replacing only the two old scaler-restriction nodes. Existing owned execution
matches exactly after removing its one new helper binding and normalizing version
names. Forty protected source/archive dependencies match their baseline bytes.
Owner-named Shared7 scaler and corrected direct23 report hashes were checked as
metadata bytes only; no model tensors, forecasts or acoustic arrays were decoded.
No scientific numerical function/model/objective/config was changed.

Production43/47 preparation, actual bounded tensor audit/embedded scaler equality,
owned process/resource checks and distinct review are NOT_RUN here. Root later
integrates closed bytes and executes only when all fixed endpoints exist and
scientific ownership is idle. No fit/selection/final-test access/approval, model
quality or safety-policy claim is established. No root files/ledger were written.
Private fixture directories are excluded; integrate top-level listed evidence
only. Use ROOT src on PYTHONPATH for the current immutable ancestry decoder.
"""
save(HERE / "NATIVE_TRAIN_SCALER_RESERVATION_V1.md", report.encode())
evidence = {p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.name != "close-delivery-01.log"}
handoff = {
    "status": "CLOSED_ENGINEERING_DELIVERY",
    "kind": "native_train_scaler_reservation_builder_handoff_v1",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
    "files": hashes,
    "dependencies": current,
    "owner_named_metadata_pins": metadata_pins,
    "protected_sources_unchanged": True,
    "evidence_files": evidence,
    "executed_checks": checks,
    "production_status": "NOT_RUN",
    "root_commands_cwd": str(MAIN),
    "root_environment": {"PYTHONPATH": str(MAIN / "src")},
    "root_check_commands": [
        r".venv\Scripts\python.exe -B -m pytest tests\unit\test_native_train_scaler_reservation_v1.py -q --import-mode=importlib -p no:cacheprovider",
        r".venv\Scripts\python.exe -B tools\prepare_native_development_comparison_v5.py",
        r".venv\Scripts\python.exe -B tools\prepare_completed_native_inventory_v6.py",
        r".venv\Scripts\python.exe -B tools\execute_completed_native_inventory_owned_v6.py",
    ],
    "root_execution_preconditions": ["Metadata preparation commands are prospective only, NOT authorized by this delivery", "All47 fixed outcomes/43 neural endpoints completed and present", "Live scientific tree idle, no ownership lock/pending journal/reconciliation", "Exact source closure integrated and fresh V5/V6 destinations", "Subsequent bounded tensor audit must compare embedded selected/inference scalers", "Distinct review remains separate from synthetic software checks"],
    "limitations": ["No tensor/scaler-state/corpus/forecast numerical decoding in builder", "Metadata alias does not establish embedded equality or approval/access authority", "No40/43 substitution or missing-outcome fabrication", "No historical scalers.json created", "No real owned audit process/resources executed", "Reference inventory remains separate; external Chronos ancestry UNKNOWN", "No scientific/final-selection/SOTA or programme completion established"],
    "excluded_from_integration": ["SYNTHETIC_CORRECTNESS_ONLY-* private fixture trees", "Active close stdout is external witness, excluded from immutable self-hash list"],
    "scientific_fitting": False,
    "gpu_initialization": False,
    "main_writes": False,
    "approval": False,
}
write_json(HERE / "handoff-v1.json", handoff)
assert all(sha(HERE / name) == value for name, value in evidence.items())
assert all(sha(BUILDER / path) == value for path, value in hashes.items())
print(json.dumps({"status": handoff["status"], "files": hashes, "handoff_sha256": sha(HERE / "handoff-v1.json"), "report_sha256": sha(HERE / "NATIVE_TRAIN_SCALER_RESERVATION_V1.md"), "source_proof_sha256": sha(HERE / "source-proof-final-v1.json"), "protected_sources_unchanged": True, "protected_dependencies": len(current)}, allow_nan=False))
