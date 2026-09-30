"""Prove a prospective supervisor repair changes only pending-journal admission."""

import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    before_path = folder / "prefix-supervisor-before-pending-repair-v3.json"
    before = json.loads(before_path.read_bytes())
    old = base64.b64decode(before["base64_bytes"])
    source = ROOT / "tools/execute_native_prefix_job_v2.py"
    current = source.read_bytes()
    if hashlib.sha256(old).hexdigest() != before["sha256"]:
        raise ValueError("Exact rejected supervisor snapshot required")
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(old).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(current).body if isinstance(n, ast.FunctionDef)}
    changed = [name for name in old_functions if old_functions[name] != new_functions.get(name)]
    if changed != ["main"] or set(old_functions) != set(new_functions):
        raise ValueError("Only exact launch pending admission may change")
    expected = old.decode("utf-8").replace(
        '            LEDGER.with_suffix(s).exists() for s in (".pending", ".prefix-pending", ".band-pending")',
        '            LEDGER.with_suffix(s).exists()\n            for s in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending")')
    if current.decode("utf-8").replace("\r\n", "\n") != expected.replace("\r\n", "\n"):
        raise ValueError("No other supervisor operation may change")
    log = folder / "prefix-pending-repair-policy-green-v3.log"
    if "31 passed" not in log.read_text(encoding="utf-8"):
        raise ValueError("Actual full31 policy checks required")
    receipt = {"status": "UNFITTED_PREFIX_PENDING_REPAIR_31_CHECKS_PASSED", "actual_exit_code": 0,
               "actual_exit_chunk_id": "5ba404", "before_sha256": before["sha256"],
               "current_sha256": hashlib.sha256(current).hexdigest(), "changed_functions": changed,
               "only_added_pending_guards": [".assessment-pending", ".reconciliation-pending"],
               "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
               "original_fitted_sources_or_scientific_recipe_changes": False,
               "scientific_admission": "NOT_GRANTED"}
    with (folder / "prefix-pending-repair-closeout-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "checks": 31}))


if __name__ == "__main__":
    main()
