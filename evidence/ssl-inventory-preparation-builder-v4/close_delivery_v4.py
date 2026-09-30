"""Close hashes, retained test evidence and metadata-only ROOT instructions."""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
sys.path.insert(0, str(BUILDER / "tools"))

import prepare_completed_native_inventory_v4 as preparer
from marine_echo.evaluation import native_ancestry_inventory as inventory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


baseline = json.loads((HERE / "dependency-baseline-v4.json").read_bytes())
protected = {}
for name, expected in baseline.items():
    actual = digest(Path(name))
    assert actual == expected, f"Protected assigned input changed: {name}"
    protected[name] = {"before_sha256": expected, "current_sha256": actual, "unchanged": True}
authored = {}
for name in (
    "tools/prepare_completed_native_inventory_v4.py",
    "tools/execute_completed_native_inventory_owned_v4.py",
    "tests/unit/test_completed_native_inventory_v4_preparation.py",
):
    path = BUILDER / name
    authored[name] = digest(path)
    with (HERE / ("closed-source-" + path.name)).open("xb") as stream:
        stream.write(path.read_bytes())
closure = {str(p): digest(p) for p in preparer.dependencies(MAIN)}
closure.update({name: item["current_sha256"] for name, item in protected.items()})
proof = {
    "kind": "native_completed_inventory_preparation_source_proof_v4",
    "authored": authored,
    "protected_dependencies": protected,
    "protected_sources_unchanged": True,
    "protected_baseline_count": len(protected),
    "source_closure_current_hashes": closure,
    "actual_inventory_helper_file": inventory.__file__,
    "actual_preparer_file": preparer.__file__,
    "actual_audit_source_root": str(inventory.source_root()),
    "root_scientific_ledger_read_or_written_by_builder": False,
    "public_corpus_tensor_or_forecast_decoded": False,
    "production_source_writes": list(authored),
}
proof_path = HERE / "source-proof-closed-v4.json"
save(proof_path, proof)

python = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B"
test = python + r" -m pytest -q --import-mode=importlib -p no:cacheprovider tests\unit\test_completed_native_inventory_v4_preparation.py"
source_args = " ".join(name.replace("/", "\\") for name in authored)
checks = [
    {"command": test + " -k snapshot_binding", "exit_code": 1, "log": "red-snapshot-binding-v4.log", "result": "Fixture setup error while extracting nonliteral source default; not behavioral red evidence", "witness": "c40c21"},
    {"command": test + " -k snapshot_binding", "exit_code": 1, "log": "red-snapshot-binding-focused-v4.log", "result": "Meaningful red: exact43 fixture still binds mutable live ledger", "witness": "a37b6a"},
    {"command": test, "exit_code": 1, "log": "green-first-v4.log", "result": "39 passed, 1 failed in180.29s; Windows test path assertion corrected", "witness": "c5cc6f"},
    {"command": test, "exit_code": 0, "log": "green-final-v4.log", "result": "48 passed in191.11s before narrow original direct-scaler compatibility follow-up", "witness": "040ff1"},
    {"command": test + " -k snapshot_binding", "exit_code": 0, "log": "green-original-downstream-scalers-v4.log", "result": "1 passed, 47 deselected in20.98s on closed bytes; whole43 reservation includes original frozen and direct endpoints without duplicate scaler files", "witness": "be2a28"},
    {"command": python + " -m ruff check --no-cache " + source_args, "exit_code": 1, "log": "lint-initial-v4.log", "result": "Import order and invalid-type exception lint; corrected", "witness": "e0a550"},
    {"command": python + " -m ruff check --no-cache " + source_args, "exit_code": 0, "log": "lint-closed-v4.log", "result": "All checks passed", "witness": "f44d0d"},
    {"command": python + " -m ruff format --check --no-cache " + source_args, "exit_code": 0, "log": "format-check-closed-v4.log", "result": "3 files already formatted", "witness": "f11e26"},
]
for item in checks:
    item["log_sha256"] = digest(HERE / item["log"])
save(HERE / "executed-checks-v4.json", checks)

commands = [
    r".venv\Scripts\python.exe -B -m pytest -q --import-mode=importlib -p no:cacheprovider tests\unit\test_completed_native_inventory_v4_preparation.py",
    r".venv\Scripts\python.exe -B tools\prepare_completed_native_inventory_v4.py --matrix orchestration\native_development_comparison_v4.json --manifest orchestration\native_completed_inventory_v4.json --ledger-snapshot evidence\ssl-research-v1\native-completed-inventory-ledger-v4.json --output evidence\native-completed-inventory-v4",
    r".venv\Scripts\python.exe -B tools\execute_completed_native_inventory_owned_v4.py --manifest orchestration\native_completed_inventory_v4.json --ledger-snapshot evidence\ssl-research-v1\native-completed-inventory-ledger-v4.json --output evidence\native-completed-inventory-v4 --receipt evidence\ssl-research-v1\native-completed-inventory-owned-v4",
]
report = HERE / "NATIVE_COMPLETED_INVENTORY_PREPARATION_V4.md"
text = [
    "# Completed-neural inventory V4 engineering delivery",
    "",
    "The preparer admits only an existing development reservation with exactly47 explicit paths:43 neural endpoints and persistence, seasonal24, lightgbm, chronos2_zero_shot. It neither enumerates ignored outputs nor chooses seeds. Missing/incomplete endpoints, duplicate identities, unresolved original reviews/configs or absent fitted parents fail closed; no partial inventory is published.",
    "",
    "The original35-endpoint manifest and all historical receipts were left untouched and were not executed. V4 saves live idle-ledger bytes exclusively to a fresh snapshot. Exact completed review lookup uses the captured ledger; only the snapshot enters the audit bindings. Live ownership is reread during preparation, after capture and before publishing its preparation receipt. A changed capture retains any partial fresh files and prevents a completed reservation; there is no deletion or retry. A later idle ledger change does not invalidate the immutable snapshot, while live active/reconciliation records, the actual GPU owner lock and all pending journal forms still block preparation and launch.",
    "",
    "Exact complete config files, embedded core-config metadata through run.json, original reviews, selected/inference/membership/scaler byte hashes, declared TRAIN corpus/report/cohort/split and all three historical source catalogs remain bound. Kind is derived from bound producer source's declared saved schema and architecture, preserving V1/V2 distinctions; the existing safe decoder must verify actual tensor/artifact schema and embedded scalers at ROOT audit. Original downstream frozen/full/direct endpoints without duplicate scalers.json may use only the original shared_ssl_seed7_h96_cuda0 TRAIN scaler file. Band endpoints cannot acquire that fallback. Every non-null fitted run+encoder ancestor resolves to another explicitly named endpoint; fresh direct retains null ancestors and supervised labels.",
    "",
    "The separately bound snapshot reservation receipt records all43 neural identities, four reference names/paths and exact47 cardinality. The audit manifest deliberately has references={} because reference-specific details are not audited here. Chronos ancestry remains unknown and receives no local-neural ancestry guarantee. The preparation is METADATA_RESERVED_NOT_APPROVAL, never final selection, a fit review or numerical-access authorization.",
    "",
    "The additive executor checks the immutable reservation/hash bindings and live idle gates before output mutations or launch. It runs only the existing CPU metadata audit CLI under original supervise_owned, at600seconds and strictly below22GiB owned-tree peak RSS. Success requires actual child0, no stop, verified cleanup, bounded measurements and an exact43-model inventory.json completion with the manifest hash. Failed attempts keep logs/resources; parent exceptions retain child identity and a reconciliation receipt. No shared scientific ledger or unknown lock is modified.",
    "",
    "Executed SYNTHETIC_CORRECTNESS_ONLY metadata checks: meaningful red exit1 for live-ledger binding; full48 passed exit0 in191.11s; then a narrow source-only compatibility follow-up and its whole43-endpoint positive test passed exit0 in20.98s. Closed Ruff check/format-check exited0. Earlier setup, Windows path assertion and lint failures are preserved, not counted as passes. Fixtures use physically serialized private Unicode metadata and opaque noncodec byte placeholders; they never decode real corpora, forecasts or tensors. Owned-process policy fixtures use injected virtual callables and fake PID/resource values, not actual process measurements or scientific execution.",
    "",
    f"All{len(protected)} explicitly snapshotted assigned ROOT dependency files remain byte-identical. Current transitive source hashes and actual MAIN/helper versus builder/new-source paths are in source-proof-closed-v4.json. No original source, historical inventory, root ledger, lock, Git index or commit was changed. Private synthetic fixture trees are retained locally and intentionally excluded from the delivery list; only small top-level proof/log/handoff files are handed off.",
    "",
    "## ROOT follow-up — NOT_RUN by builder",
    "",
    "Integrate checked bytes, run the full closed suite, then prepare only after the final47 reservation exists and the genuine scientific tree is idle. All listed destinations must be absent; an earlier attempt requires separately chosen fresh versioned paths, never overwrite/retry. The reservation sidecar is automatically named native-completed-inventory-ledger-v4.reservation.json. The existing decoder performs actual safe weight/embedded-scaler/state/whole-TRAIN ancestry checks under the owned process. If an original immutable review itself binds a mutable ledger, this implementation refuses it and requires an independently preserved original receipt rather than rewriting ancestry or old approvals.",
    "",
    "```cmd", *commands, "```",
    "",
    "Real completed-artifact decoding, actual child lifecycle/resource measurements and final47 preparation/audit are NOT_RUN here. Missing final47 matrix is an expected blocker, never a reason to skip endpoints. Independent inventory/ancestry review remains required before final freeze or held-out use. Scientific quality, final numeric authority, programme completion and SOTA are NOT_ESTABLISHED.",
    "",
    "## Closed authored hashes", "",
    *[f"- `{name}`: `{value}`" for name, value in authored.items()],
]
with report.open("x", encoding="utf-8") as stream:
    stream.write("\n".join(text) + "\n")

evidence = {str(p.relative_to(BUILDER)): digest(p) for p in HERE.iterdir() if p.is_file()}
handoff = {
    "kind": "native_completed_inventory_preparation_builder_handoff_v4",
    "status": "CLOSED_ENGINEERING_DELIVERY",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
    "files": authored,
    "dependencies": closure,
    "protected_sources_unchanged": True,
    "protected_sources_baseline_count": len(protected),
    "protected_sources": protected,
    "source_proof": {"path": str(proof_path.relative_to(BUILDER)), "sha256": digest(proof_path)},
    "report": {"path": str(report.relative_to(BUILDER)), "sha256": digest(report)},
    "red_green_lint": checks,
    "expected_cardinality": {"neural": 43, "references": 4, "comparison": 47},
    "reference_names": list(preparer.REFERENCES),
    "limitations": [
        "Actual completed weights/corpora/forecasts were not decoded in builder.",
        "Actual ROOT owned process and600s/22GiB resource audit NOT_RUN; policy tests are virtual.",
        "Final47 public matrix preparation and audit NOT_RUN; every endpoint is mandatory.",
        "References remain empty in neural audit manifest; four paths/names reserved separately; Chronos ancestry UNKNOWN.",
        "Actual tensor kinds, embedded scaler equality and TRAIN membership/state ancestry remain existing decoder's ROOT audit obligations.",
        "Original reviews that themselves hash mutable ledger bytes require an original immutable receipt; no original approval is rewritten.",
        "Full48 suite passed before final original direct-scaler compatibility extension; focused whole43 case passed afterward. ROOT repeats full closed suite.",
        "Distinct ancestry/inventory review needed before final freeze or held-out use; no scientific authority granted.",
    ],
    "root_required_commands": commands,
    "historical_manifest_changes": False,
    "root_ledger_or_lock_modifications": False,
    "public_numeric_decode_or_gpu_or_fitting": False,
    "scientific_review_authority": False,
    "git_index_or_commit": False,
    "top_level_evidence": evidence,
    "synthetic_fixture_tree_delivery": "EXCLUDED; private, no actual process/scientific evidence",
}
target = HERE / "handoff-v4.json"
save(target, handoff)
print(json.dumps({"files": authored, "protected_sources_unchanged": True,
                  "source_proof_sha256": digest(proof_path), "report_sha256": digest(report),
                  "handoff_sha256": digest(target)}, indent=2))
