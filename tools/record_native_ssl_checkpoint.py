"""Update dynamic research ledger from actual review, checkpoint and reference records."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    for record in ledger["runs"]:
        if record["id"] == "model_prefit_review":
            record.update(
                status="APPROVED_EXACT_RENEWED_BINDINGS",
                review="evidence/ssl-research-v1/prefit-review-final-02.json",
                verification="evidence/ssl-research-v1/prefit-bindings-verification-02.json",
            )
        if record["id"] == "shared_ssl_seed7_history96":
            record.update(
                status="INITIAL_PLAN_EXECUTING_AS_BOUND_CUDA_JOB",
                execution_id="shared_ssl_seed7_h96_cuda0",
            )
        if record["id"] == "shared_ssl_seed7_h96_cuda0":
            record["durable_progress"] = json.loads(
                (ROOT / "evidence/ssl-research-v1/shared-first-job-progress-1500.json").read_text()
            )
    for method in ("persistence", "seasonal24"):
        identifier = method + "_h96"
        if any(r["id"] == identifier for r in ledger["runs"]):
            continue
        result_path = ROOT / "outputs/native_acoustic_ssl_v1" / identifier / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        ledger["runs"].append(
            {
                "id": identifier,
                "status": result["status"],
                "exit_code": 0,
                "output": str(result_path.relative_to(ROOT)),
                "daily_dev_pinball_db": result["metrics"]["primary_pinball_db"],
                "elapsed_seconds": result["elapsed_seconds"],
                "fitted_parameters": 0,
                "test_access": "NOT_RUN",
                "gpu_hours": 0,
            }
        )
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
