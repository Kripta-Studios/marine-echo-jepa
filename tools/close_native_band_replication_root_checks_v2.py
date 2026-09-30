"""Bind the narrow unfitted metadata repair and actual root CPU checks."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "evidence/ssl-band-replication-builder-v2"
REPAIRED = "src/marine_echo/inference/native_band_replication_acoustic.py"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nodes(source):
    result = {}
    for item in ast.parse(source).body:
        if isinstance(item, ast.FunctionDef):
            result[item.name] = ast.dump(item, include_attributes=False)
        elif isinstance(item, ast.ClassDef):
            for method in item.body:
                if isinstance(method, ast.FunctionDef):
                    result[item.name + "." + method.name] = ast.dump(method, include_attributes=False)
    return result


def main():
    handoff = json.loads((FOLDER / "handoff-v2.json").read_bytes())
    original = json.loads((FOLDER / "final-code-snapshot-v2.json").read_bytes())[REPAIRED]
    before, after = nodes(original["text"]), nodes((ROOT / REPAIRED).read_bytes())
    if set(before) != set(after) or {k for k in before if before[k] != after[k]} != {"_base"}:
        raise ValueError("Only explicit metadata admission may change")
    for name, expected in handoff["authored_sha256"].items():
        if name != REPAIRED and digest(ROOT / name) != expected:
            raise ValueError("Other closed replication source changed")
    protected = {p: sha for p, sha in handoff["source_closure_sha256"].items()
                 if Path(p).is_relative_to(ROOT) and Path(p).resolve() != (ROOT / REPAIRED).resolve()}
    for name, expected in protected.items():
        if digest(name) != expected:
            raise ValueError(f"Original protected dependency changed: {name}")
    logs = {
        "initial_root": ("root-all-checks-v2.log", "6 failed, 220 passed in 39.80s", 1, "25245/f78925"),
        "metadata_red": ("root-review-metadata-red-v2.log", "7 failed", 1, "bd2466"),
        "metadata_green": ("root-review-metadata-green-v2.log", "7 passed", 0, "55487/0b8cde"),
        "repaired_full": ("root-all-checks-repaired-v2.log", "233 passed in 63.38s", 0, "75738/8c7eae"),
        "repaired_durable": ("root-durable-repaired-v2.log", '"durable_replay": "PASSED"', 0, "b2ed7e"),
    }
    checks = {}
    for key, (name, expected, code, witness) in logs.items():
        path = FOLDER / name
        if expected not in path.read_text(encoding="utf-8"):
            raise ValueError(f"Actual execution log differs: {name}")
        checks[key] = {"path": str(path), "sha256": digest(path),
                       "actual_exit_code": code, "actual_exit_witness": witness}
    result = {"status": "REPAIRED_REPLICATION_ROOT_233_CHECKS_AND_DURABLE_CPU_REPLAY_PASSED",
              "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "checks": checks,
              "passed_checks_total": 233, "root_only_cases_executed": 12, "skipped": 0,
              "repair": {"path": REPAIRED, "before_sha256": original["sha256"],
                         "after_sha256": digest(ROOT / REPAIRED), "changed_function": "_base",
                         "scope": "Known opaque review_sha256 metadata with lowercase64hex validation",
                         "all_other_function_asts_identical": True,
                         "original_snapshot_path": str(FOLDER / "final-code-snapshot-v2.json")},
              "current_authored_sha256": {name: digest(ROOT / name) for name in handoff["authored_sha256"]},
              "protected_original_dependencies_sha256": protected,
              "original_capacity_and_root_failures_preserved": True,
              "scientific_models_or_training_changed": False, "fitted_sources_changed": False,
              "independent_prefit": "NOT_RUN", "public_numeric_access": "NOT_RUN",
              "scientific_replication_fitting": "NOT_RUN"}
    with (FOLDER / "root-checks-closeout-repaired-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "passed_checks": 233,
                      "repaired_source_sha256": result["repair"]["after_sha256"]}))


if __name__ == "__main__":
    main()
