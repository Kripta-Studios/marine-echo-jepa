"""Record actual non-scientific platform verification without replacing failures."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(path.read_text())
    if any(r.get("id") == "cf_deterministic_cuda_correctness_v1" for r in ledger["runs"]):
        raise ValueError("Actual correctness invocation already journalled.")
    check = json.loads(
        (ROOT / "evidence/ssl-research-v1/cf-deterministic-cuda-green.log").read_text()
    )
    assert check["status"] == "PASSED" and check["optimizer_updates"] == 0
    ledger["runs"].append(
        {
            "id": "cf_deterministic_cuda_correctness_v1",
            **check,
            "exit_code": 0,
            "full_tool_owned_wall_seconds": 6.0730095,
        }
    )
    ledger["gpu_hours_spent_owned_scientific_jobs"] += 6.0730095 / 3600
    for run in ledger["runs"]:
        if run.get("id") == "lightgbm_seed7_h96":
            run["resource_measurement_limit"] = (
                "INCOMPLETE_LAUNCHER_ONLY_PEAK; actual interpreter descendant was not monitored in this preserved failed attempt"
            )
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("Recorded actual CUDA correctness and preserved comparator measurement limitation.")


if __name__ == "__main__":
    main()
