"""Create a new prefix supervisor version with additive closed-history accounting only."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "tools/execute_native_prefix_job.py"
    target = ROOT / "tools/execute_native_prefix_job_v2.py"
    original = source.read_text(encoding="utf-8")
    revised = original.replace("from execute_native_band_job import budget_totals",
                               "from native_band_budget_history_v2 import budget_totals")
    marker = '            ROOT / "tools/execute_native_band_job.py",\n'
    if original.count(marker) != 1 or original.count("from execute_native_band_job import budget_totals") != 1:
        raise ValueError("Exact original supervisor source required")
    revised = revised.replace(marker, marker + '            ROOT / "tools/native_band_budget_history_v2.py",\n')
    before = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(revised).body if isinstance(n, ast.FunctionDef)}
    unchanged = [name for name in before if before[name] == after[name]]
    if set(before) - set(unchanged) != {"resource_budget", "main"}:
        raise ValueError("Only source binding and versioned budget import may change")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(revised)
    with (ROOT / "evidence/ssl-research-v1/prefix-budget-wrapper-versioning-v2.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "UNFITTED_PREFIX_SUPERVISOR_VERSIONED_NOT_APPROVAL", "original_path": str(source),
                   "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "original_source_text": original,
                   "new_path": str(target), "new_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                   "unchanged_functions": unchanged, "changed_functions": ["resource_budget", "main"],
                   "only_changes": "Import additive closed-prefix-history budget helper and bind its exact source before use",
                   "scientific_recipe_or_ledger_changes": False, "fitting": "NOT_RUN"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "UNFITTED_PREFIX_SUPERVISOR_VERSIONED_NOT_APPROVAL"}))


if __name__ == "__main__":
    main()
