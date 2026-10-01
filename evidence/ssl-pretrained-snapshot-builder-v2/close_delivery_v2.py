"""Close source-only proof and explicit synthetic witnesses; no model decoding."""

import ast
import hashlib
import json
import sys
from pathlib import Path

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILDER / "tools"))
import package_native_pretrained_snapshot_v2 as package


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def save_json(path, value):
    save(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


authored = [
    "tools/package_native_pretrained_snapshot_v2.py",
    "tools/check_native_pretrained_snapshot_worker_v3.py",
    "tools/check_native_pretrained_snapshot_owned_v3.py",
    "tests/unit/test_native_pretrained_snapshot_v2_contract.py",
]
baseline = json.loads((HERE / "protected-baseline-v2.json").read_bytes())
current = {path: sha(path) for path in baseline["files"]}
assert current == baseline["files"], "Protected ROOT dependency changed; stop closeout"
closure = {str(path): sha(path) for path in package.source_closure(MAIN)}
assert set(closure) <= set(current), "An unbaselined inference source requires explicit proof"
source_hashes = {path: sha(BUILDER / path) for path in authored}
snapshot_files = {}
for path in authored:
    destination = HERE / ("closed-" + Path(path).name)
    save(destination, (BUILDER / path).read_bytes())
    assert sha(destination) == source_hashes[path]
    snapshot_files[destination.name] = sha(destination)

old_ast = ast.parse((MAIN / "tools/check_native_pretrained_snapshot_worker_v2.py").read_bytes())
new_ast = ast.parse((BUILDER / authored[1]).read_bytes())
old_nodes = {ast.dump(node, include_attributes=False) for node in ast.walk(old_ast)}
new_nodes = {ast.dump(node, include_attributes=False) for node in ast.walk(new_ast)}
expressions = [
    "encoder.encode(x, observed, metadata)",
    "predictor.encode(x, observed, metadata)",
    "predictor.predict_cf_zones(x, observed, metadata)",
    "predictor.predict_latents(x, observed, metadata, query)",
    "np.array_equal(embedding, predictor.encode(x, observed, metadata))",
    "np.isfinite(embedding).all()",
    "np.isfinite(predicted).all()",
    "torch.equal(state, torch.get_rng_state())",
    "torch.cuda.is_initialized()",
    "200 / 250",
]
replay_ast = {}
for expression in expressions:
    encoded = ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False)
    assert encoded in old_nodes and encoded in new_nodes, expression
    replay_ast[expression] = "EXACT_AST_EXPRESSION_PRESERVED"
proof = {
    "kind": "native_pretrained_snapshot_source_proof_v2",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
    "main_read_only": True,
    "protected_sources_unchanged": True,
    "protected_baseline_sha256": sha(HERE / "protected-baseline-v2.json"),
    "protected_current": current,
    "static_local_inference_closure": closure,
    "authored": source_hashes,
    "closed_snapshots": snapshot_files,
    "old_worker_replay_expressions": replay_ast,
    "scientific_sources_changed": [],
    "qa_changes": ["Seven fixed typed endpoints", "Band V1/V2 loader routing", "Copied byte hashes", "Namespace __file__ AND __path__ containment", "Fresh external CPU receipts and original owned supervisor"],
    "public_tensor_or_corpus_decoding": "NOT_RUN",
    "v1_zip_owner_pin": baseline["v1_archive_owner_pin"],
    "v1_zip_actual_byte_verification": "ROOT_PACKAGE_ONLY_NOT_RUN_BUILDER",
    "ast_limit": "Expression preservation and dependency byte identity; no new scientific approval or actual seven-model replay asserted.",
}
save_json(HERE / "source-proof-final-v2.json", proof)

pytest_base = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B -m pytest tests\unit\test_native_pretrained_snapshot_v2_contract.py -q --import-mode=importlib -p no:cacheprovider"
files_cli = " ".join(path.replace("/", "\\") for path in authored)
ruff_base = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B -m ruff "
checks = [
    {"command": pytest_base + " -k band_v1_routes", "exit_code": 1, "witness": "5de8d6", "log": "red-band-v1-route-01.log", "result": "1 meaningful failure: Band V1 routed to legacy; 48 deselected"},
    {"command": pytest_base, "exit_code": 1, "witness": "3a9d95", "log": "green-contract-01.log", "result": "48 passed; 1 fixture assertion matched an earlier byte guard instead of membership identity guard; corrected private fixture only"},
    {"command": pytest_base, "exit_code": 0, "witness": "c514e2", "log": "green-contract-02.log", "result": "54 passed in 15.48s"},
    {"command": pytest_base, "exit_code": 0, "witness": "f94e5d", "log": "green-contract-final-03.log", "result": "54 passed in 15.82s; includes added immutable V1 tooling bindings"},
    {"command": ruff_base + "check --no-cache " + files_cli, "exit_code": 1, "witness": "93c478", "log": "lint-check-01.log", "result": "RUF012 in synthetic adapter; fixed instance metadata"},
    {"command": ruff_base + "format --no-cache " + files_cli, "exit_code": 0, "witness": "50d7eb", "log": "format-final-03.log"},
    {"command": ruff_base + "check --no-cache " + files_cli, "exit_code": 0, "witness": "139a3d", "log": "lint-check-final-03.log", "result": "All checks passed"},
    {"command": ruff_base + "format --check --no-cache " + files_cli, "exit_code": 0, "witness": "9bdeb4", "log": "lint-format-final-03.log", "result": "4 files already formatted"},
]
for check in checks:
    check["log_sha256"] = sha(HERE / check["log"])
save_json(HERE / "executed-checks-v2.json", {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "checks": checks, "virtual_process_checks": "Test doubles, NOT actual supervised child execution", "unicode_codec": "Physical ZIP/JSON opaque synthetic byte-copy roundtrip; no trained tensor decoding"})

report = """# Pretrained snapshot V2 engineering delivery

Closed four new files. This extends the existing working snapshot/CPU checker to
exactly Shared7, CF7/13/23 and Band Shared7/13/23. There is no best-seed choice.
All model, training and inference source bytes remain unchanged. The static local
inference closure and 33 protected ROOT dependencies match the recorded baseline.
The old CPU check's encoding, latent calls, finite checks, query rejection and RNG
expressions are preserved by AST; new routing uses the existing typed APIs.

The packager checks completed reports, inference/membership bytes, original TRAIN
scalers, full bound configs, the original four replay records, the pinned V1 ZIP
and Band23 run identity. It copies only five model leaves per endpoint, static
source, configs, existing licenses/provenance and necessary historical evidence.
Missing historical code remains an explicit unrecovered version; recovered code
is never compatibility approval. No license is added. The manifest and card say
INCOMPLETE_RESEARCH_SNAPSHOT with QA pending a separate external receipt.

The worker retains selected-versus-latent equality, finite outputs, frozen CPU
loading, RNG preservation and no CUDA initialization. Band V1 and V2 use their
actual typed loaders and reject a 230m query relabelled 200m. CF ONLINE ordinal
zones remain distinct from selected EMA encodings; full H96 zone inference is
outside sampled training-crop support. Source-calendar timezone is unknown.
Namespace files and every package __path__ must stay within copied source.

The owned QA wrapper uses the unchanged supervisor with a 600-second full attempt
deadline and tree RAM below 22GiB. Scientific ledger, owner lock and pending
journal gates apply before copying/launching. Every destination is fresh;
failures retain receipts and parent exceptions require reconciliation. The
immutable manifest is not rewritten after QA.

Actual builder checks: meaningful red exit1 for Band V1 routing; final focused
pytest exit0, 54 passed in 15.82s; Ruff check and format check both exit0. The
physical Unicode ZIP fixture copied opaque synthetic bytes. Supervisor fixtures
are virtual policy tests, NOT actual process cleanup/resource measurements.
All commands, intermediate failures and exit witnesses remain in executed-checks.
Final formatting only changed layout after the last passing checks.

ROOT packaging, real copied seven-model CPU replay/RNG/resource verification,
actual V1 ZIP pin verification and independent package review are NOT_RUN here.
Run the handoff commands only after integration and scientific ownership is idle.
Do not integrate synthetic-* fixture trees: they contain intentionally fake
reports/opaque non-tensors, are not evidence of real completed training, and are
excluded from the closed handoff file list. No public tensors/corpora/forecasts
were decoded, no fit/GPU/process-supervisor scientific execution occurred, no
root ledger/source was written and no denied operation was retried.

Strong endpoints and controls remain separate study artifacts. Short selection
probes do not establish transfer. No scientific, prefit, held-out, sealed-site,
final-selection, performance/SOTA, packaging/release or publication authority is
conferred. ROOT integrates exact closed files and executes/reviews separately.
"""
save(HERE / "NATIVE_PRETRAINED_SNAPSHOT_V2.md", report.encode())
evidence = {path.name: sha(path) for path in HERE.iterdir() if path.is_file()}
handoff = {
    "kind": "native_pretrained_snapshot_builder_handoff_v2",
    "status": "CLOSED_ENGINEERING_DELIVERY",
    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
    "files": source_hashes,
    "evidence_files": evidence,
    "dependencies": current,
    "static_local_inference_closure": closure,
    "protected_sources_unchanged": True,
    "models": [{"id": name, "method": method, "seed": seed, "inference_kind": kind} for name, method, seed, kind in package.SPECS],
    "root_commands_cwd": str(MAIN),
    "root_commands": [
        r".venv\Scripts\python.exe -B -m pytest tests\unit\test_native_pretrained_snapshot_v2_contract.py -q --import-mode=importlib -p no:cacheprovider",
        r".venv\Scripts\python.exe -B tools\package_native_pretrained_snapshot_v2.py",
        r".venv\Scripts\python.exe -B tools\check_native_pretrained_snapshot_owned_v3.py",
    ],
    "root_required": ["Idle original scientific ledger/owner lock/pending journals", "Exact closed bytes integrated", "Fresh snapshot directory/ZIP/package receipt and separate fresh QA directory", "Actual copied seven-model CPU replay, preserved Torch RNG, no CUDA initialization and namespace isolation", "Original supervisor actual exit/cleanup/tree RSS/full-time receipt", "Distinct independent package/compatibility review before later packaging use"],
    "root_only_not_run": ["Production package", "Actual trained tensor loading/replay", "Physical owned child lifecycle", "V1 public ZIP byte validation", "Independent review"],
    "builder_checks": checks,
    "excluded_from_integration": ["synthetic-* private fixture trees"],
    "limitations": ["INCOMPLETE_RESEARCH_SNAPSHOT", "No transfer-value or SOTA established", "No new scientific/prefit/final-selection/held-out authority", "Historical source preservation is not compatibility approval", "No new license or public app/release", "Strong downstream controls/readouts remain separate", "CF online ordinal zones are not horizons1/3/6 acoustic forecasts", "Full-H96 CF zones exceed sampled crop view; source clock UNKNOWN UTC"],
    "public_numeric_access": False,
    "scientific_fitting": False,
    "gpu_execution": False,
    "self_approval": False,
}
save_json(HERE / "handoff-v2.json", handoff)
print(json.dumps({"status": handoff["status"], "files": source_hashes, "protected_dependencies": len(current), "closure_sources": len(closure), "source_proof_sha256": sha(HERE / "source-proof-final-v2.json"), "report_sha256": sha(HERE / "NATIVE_PRETRAINED_SNAPSHOT_V2.md"), "handoff_sha256": sha(HERE / "handoff-v2.json")}, allow_nan=False))
