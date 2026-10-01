"""Close an immutable, small handoff from actual source and check witnesses."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


proof = json.loads((HERE / "source-proof-final-v1.json").read_bytes())
for name, expected in proof["authored"].items():
    assert sha(BUILDER / name) == expected
for name, expected in proof["protected_baseline"].items():
    assert sha(name) == expected
checks = []
for path in sorted(HERE.glob("*.json")):
    value = json.loads(path.read_bytes())
    if "command" in value and "exit_code" in value:
        checks.append({"witness": path.name, **value})
evidence = {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
runner = ".venv/Scripts/python.exe -m pytest --import-mode=importlib -q -p no:cacheprovider tests/unit/test_native_cf_controls.py tests/integration/test_native_cf_controls.py"
value = {"status": "CLOSED_ENGINEERING_DELIVERY", "implementer_session_id": "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1",
    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "files_sha256": proof["authored"],
    "dependencies_sha256": proof["protected_baseline"], "runtime_source_closure_sha256": proof["runtime_static_source_closure"],
    "protected_sources_unchanged": True, "source_proof": "source-proof-final-v1.json", "evidence_sha256": evidence,
    "executed_checks": checks, "final_tests": {"passed": 65, "skipped_root_only": 4, "exit_code": 0, "seconds": 11.80, "witness": "focused-green-08.json"},
    "behavioral_red": {"exit_code": 1, "witness": "red-frozen-bn-02.json", "invariant": "frozen encoder BatchNorm buffers"},
    "final_lint": {"check_exit": 0, "format_exit": 0, "witnesses": ["lint-check-final-03.json", "lint-format-final-03.json"]},
    "parameters": proof["parameters"],
    "root_check_commands": ["set NATIVE_CF_CONTROLS_ROOT_RUNNER_CHECKS=1", runner,
       ".venv/Scripts/python.exe -m ruff check --no-cache src/marine_echo/training/native_cf_controls.py src/marine_echo/inference/native_cf_controls.py tests/unit/test_native_cf_controls.py tests/integration/test_native_cf_controls.py"],
    "root_only_checks": {"status": "NOT_RUN", "cases": ["random-frozen AdamW exact resume/physical forecast replay", "direct AdamW encoder/moments exact resume/physical forecast replay", "sampler resume tamper", "frozen tensor resume tamper"], "reason": "Retained builder optimizer/cache restriction; no retry"},
    "training_loss_matching": proof["training_loss"], "selection_matching": proof["selection"],
    "limits": ["No distinct source/prefit review or approval", "No public numerical/tensor/prediction access; no CUDA initialization", "No real fitting or held-out evaluation", "ROOT guarded executor/configs/budget receipt are separate prospective work", "Existing Resources and its ownership classifier remain unchanged", "Existing SSL/assessment/prefix loaders reject new control artifact kinds", "Root optimizer/resume fixtures remain NOT_RUN", "No scientific quality/transfer/SOTA claim"],
    "real_control_fits": "NOT_RUN", "independent_source_prefit_review": "NOT_RUN", "held_out_evaluation": "NOT_RUN",
    "handoff_policy": "Four source/tests plus top-level small evidence only; exclude private fixture directories/bytecode"}
with (HERE / "handoff-v1.json").open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
print(json.dumps({"status": value["status"], "handoff_sha256": sha(HERE / "handoff-v1.json"), "report_sha256": sha(HERE / "NATIVE_CF_CONTROLS_IMPLEMENTATION_V1.md"), "files_sha256": proof["authored"]}))
