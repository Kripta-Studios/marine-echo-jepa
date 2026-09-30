"""Exclusive engineering closeout; no public numerical decoding or approval."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]
proof_path = HERE / "source-proof-final-v3.json"
proof = json.loads(proof_path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, item in proof["authored"].items():
    assert sha(BUILDER / name) == item["sha256"], name
for name, item in proof["protected"].items():
    assert sha(Path(name)) == item["current_sha256"], name
for name, expected in proof["imported_source_closure"].items():
    assert sha(Path(name)) == expected, name

python = r"..\marine-echo-jepa\.venv\Scripts\python.exe -B"
paths = " ".join(name.replace("/", "\\") for name in proof["authored"])
fixture = r"evidence\ssl-prefix-desktop-builder-v3\root_fixture.py"
pytest = python + r" -m pytest --import-mode=importlib -p no:cacheprovider tests\unit\test_native_prefix_desktop_execution_v3.py tests\integration\test_native_prefix_desktop_transfer_v3.py"
checks = [
    {"command": pytest, "exit_code": 0, "log": "red-initial-v3.log", "result": "34 passed, 1 skipped; initial green despite historical log basename"},
    {"command": python + r" -m pytest --import-mode=importlib -p no:cacheprovider tests\unit\test_native_prefix_desktop_execution_v3.py -k baseline_operational", "exit_code": 1, "log": "red-original-vlc-policy-v3.log", "result": "1 meaningful failure: original classifier blocks exact WDDM VLC"},
    {"command": python + " -m ruff check --no-cache " + paths + " " + fixture, "exit_code": 1, "log": "lint-final-v3.log", "result": "one import-order error; corrected"},
    {"command": pytest, "exit_code": 0, "log": "green-final-v3.log", "result": "35 passed, 1 skipped in 23.54 seconds; before final import-order correction"},
    {"command": pytest, "exit_code": 0, "log": "green-closed-v3.log", "result": "35 passed, 1 skipped in 22.45 seconds; closed authored bytes", "exit_witness_chunk": "a8e012"},
    {"command": python + r" evidence\ssl-prefix-desktop-builder-v3\close_source_proof.py", "exit_code": 1, "log": "source-proof-final-v3.log", "result": "snapshot basename collision; no existing snapshot overwritten; preserved failure"},
    {"command": python + r" evidence\ssl-prefix-desktop-builder-v3\complete_source_proof_v3.py", "exit_code": 0, "log": "source-proof-completed-v3.log", "result": "36 protected, 6 authored, 32 source bindings verified; byte and bounded AST proof", "exit_witness_chunk": "2d50be"},
    {"command": python + " -m ruff check --no-cache " + paths + " " + fixture, "exit_code": 0, "log": "lint-closed-v3.log", "result": "All checks passed", "exit_witness_chunk": "b31aff"},
    {"command": python + " -m ruff format --check --no-cache " + paths + " " + fixture, "exit_code": 0, "log": "format-check-closed-v3.log", "result": "7 files already formatted", "exit_witness_chunk": "2433ea"},
]
for check in checks:
    check["log_sha256"] = sha(HERE / check["log"])
with (HERE / "executed-checks-v3.json").open("x", encoding="utf-8") as stream:
    json.dump(checks, stream, indent=2, allow_nan=False)
    stream.write("\n")

resource_receipts = []
for path in HERE.glob("SYNTHETIC_CORRECTNESS_ONLY-*/resources.json"):
    content = json.loads(path.read_bytes())
    assert content["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY" and content["fitting"] is False
    resource_receipts.append({"path": str(path.relative_to(BUILDER)), "sha256": sha(path), "record": content})

root_commands = [
    r".venv\Scripts\python.exe -B -m pytest --import-mode=importlib -p no:cacheprovider tests\unit\test_native_prefix_desktop_execution_v3.py tests\integration\test_native_prefix_desktop_transfer_v3.py",
    r'.venv\Scripts\python.exe -B evidence\ssl-prefix-desktop-builder-v3\root_fixture.py --output "evidence\ssl-prefix-desktop-builder-v3\SYNTHETIC_CORRECTNESS_ONLY-Á-Replay-NEW"',
    r".venv\Scripts\python.exe -B tools\execute_native_prefix_desktop_job_v3.py --manifest ROOT_APPROVED_MANIFEST --review DISTINCT_NEW_PREFIT_REVIEW --output outputs\NEW_PREFIX_CELL --receipt evidence\NEW_OWNED_ATTEMPT",
]

lines = [
    "# Desktop-compatible prefix execution V3 — engineering closeout",
    "",
    "Four additive production files and two tests are closed. Original prefix, training, model, inference, worker, supervisor, owner authority and prior deliveries were not edited. No public arrays, trained weights, GPU ownership query, scientific process, or scientific fitting was executed.",
    "",
    "The new runtime copies the original native_ssl.Resources class with exact AST equality. Its operational alias imports ROOT's unchanged native_desktop_resources_v2 through its actual file path. The pinned owner resolution is 18e29745c7595a9ea50bccf23e3124b06c57106929ac20c40a8f04776ff598e4. The classifier admits only the owner-authorized exact VLC executable under WDDM; unknown and other ML processes remain blocking. VLC need not be absent or present. This operational exception grants no scientific prefit or numeric access.",
    "",
    "All prefix definitions retain the original science AST, except operational required_sources and fit's Resources route. The worker differs only by its module route. Every existing wrapper function except main retains exact AST; main normalizes exactly to the original after the documented source/worker routes, eleven operational bindings, authority validation and additional idle-ledger guard. New guards reject unknown accounting, active/reconciliation records, owner locks and pending journals before output creation or launch. Original full-owned Band12 inside aggregate96 accounting, 22GiB tree supervision, strict <10GiB CUDA caps and owned cleanup remain intact. No source-global Resources swapping or production private-root hook was added.",
    "",
    "The proof verifies 36 protected MAIN files against byte snapshots (including the preceding 28-source baseline) and records 32 actual source-closure bindings. ROOT's original prefix SHA256 remains 844159d11a1d9a410f7cb7f1a83c782360427e04aae965c19cb94a74d5a28d38. In this builder, original helpers/classifier resolve to MAIN while the four new modules resolve to the builder. Integration must preserve and rebind the actual integrated paths; no __file__ is falsified.",
    "",
    "The focused red check exited1 because the original classifier blocked WDDM VLC. Final synthetic CPU checks exited0: 35 passed, one explicitly ROOT-only skipped in22.45s. Ruff check and format-check exited0. The initial 34-pass log was named red-initial-v3.log but is initial green evidence. A lint import-order failure and a snapshot basename collision are retained; no existing snapshot was overwritten. Exact commands, exits, logs and witnesses are in executed-checks-v3.json.",
    "",
    "Actual CPU Resources checks wrote small Unicode SYNTHETIC_CORRECTNESS_ONLY receipts and prohibited CUDA initialization. These measure the builder process, not an executed training tree. Physical optimizer/save/resume/frozen-parent/portable-inference replay is NOT_RUN here: the known builder optimizer/cache denial was not retried. ROOT's supplied fixture uses private original synthetic inputs, a fresh Unicode output, a real child, the immutable supervisor, a 600-second deadline and22GiB owned-tree cap; it rejects real active ownership/journals before mkdir/Popen. It writes safe physical checkpoints and checks exact replay for frozen and scratch modes. This prospective fixture has been linted and source-inspected, not executed by the builder.",
    "",
    "## Authored files (SHA256)",
    "",
]
lines.extend(f"- `{name}`: `{item['sha256']}`" for name, item in proof["authored"].items())
lines.extend([
    "",
    "## ROOT checks after byte-checked integration",
    "",
    "Run the first two commands only after actual scientific ownership/journals close. The final command is a public entrypoint template; ROOT must supply an independently approved exact new source/runtime/data/ancestor/output manifest and review before any scientific execution. These are not authorized or executed by this handoff.",
    "",
    "```cmd",
    *root_commands,
    "```",
    "",
    "Original optimizer/selection/sampling/scaler roles, typed artifact/seed controls, true-deployment native configuration and gap proofs, native230 geometry and suffix-leak exclusions are unchanged. No prefit, held-out access, final selection, scientific quality or SOTA approval/result is established by software tests. No commit or Git index operation was attempted.",
])
report = HERE / "NATIVE_PREFIX_DESKTOP_IMPLEMENTATION_V3.md"
with report.open("x", encoding="utf-8") as stream:
    stream.write("\n".join(lines) + "\n")

evidence = {str(p.relative_to(BUILDER)): sha(p) for p in HERE.rglob("*") if p.is_file()}
handoff = {
    "kind": "native_prefix_desktop_builder_handoff_v3",
    "status": "CLOSED_ENGINEERING_DELIVERY",
    "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
    "evidence_kind": "SOURCE_AND_SYNTHETIC_CORRECTNESS_ONLY",
    "authored": {name: item["sha256"] for name, item in proof["authored"].items()},
    "support": {"evidence/ssl-prefix-desktop-builder-v3/root_fixture.py": sha(HERE / "root_fixture.py")},
    "source_proof": {"path": str(proof_path.relative_to(BUILDER)), "sha256": sha(proof_path)},
    "report": {"path": str(report.relative_to(BUILDER)), "sha256": sha(report)},
    "protected_sources": proof["protected"],
    "source_closure": proof["imported_source_closure"],
    "executed_checks": checks,
    "actual_synthetic_cpu_resource_receipts": resource_receipts,
    "root_required": {
        "status": "NOT_RUN_BY_BUILDER",
        "commands": root_commands,
        "optimizer_cache_denial_retry": False,
        "actual_owned_process_lifecycle": "ROOT_ONLY_NOT_RUN",
        "physical_optimizer_save_resume_and_portable_replay": "ROOT_ONLY_NOT_RUN",
        "actual_WDDM_CUDA_ownership_and_cap_checks": "ROOT_ONLY_NOT_RUN_REQUIRES_DISTINCT_PREFIT",
        "integration_requires_exact_bytes_and_current_source_bindings": True,
    },
    "preserved_failures": ["red-original-vlc-policy-v3.log", "lint-final-v3.log", "source-proof-final-v3.log"],
    "evidence_files": evidence,
    "original_sources_modified": False,
    "scientific_numerical_functions_modified": False,
    "public_numeric_or_selected_weights_decoded": False,
    "cuda_initialized_or_gpu_execution": False,
    "scientific_fitting_or_prefit_approval": False,
    "independent_review_required": True,
    "git_index_or_commit_attempted": False,
}
path = HERE / "handoff-v3.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(handoff, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"authored": handoff["authored"], "report_sha256": sha(report), "handoff_sha256": sha(path), "source_proof_sha256": sha(proof_path)}, indent=2))
