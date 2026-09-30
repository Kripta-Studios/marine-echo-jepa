"""Retain a real failed direct CF attempt with explicitly conservative accounting."""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    identifier = "cf_jepa_seed7_h96_cuda0"
    if any(record.get("id") == identifier for record in ledger["runs"]):
        raise ValueError("Attempt already recorded; do not double charge.")
    log = ROOT / "evidence/ssl-research-v1/cf-jepa-first-job-cuda0.log"
    if "adaptive_avg_pool2d_backward_cuda" not in log.read_text(errors="replace"):
        raise ValueError("Expected preserved actual CUDA failure evidence is absent.")
    charged = time.time() - log.stat().st_ctime
    record = {
        "id": identifier,
        "status": "FAILED_REAL_CUDA_ATTEMPT",
        "exit_code": 1,
        "tool_session_id": 24368,
        "method": "cf_jepa",
        "seed": 7,
        "history": 96,
        "output": "outputs/native_acoustic_ssl_v1/" + identifier,
        "review": "evidence/ssl-research-v1/prefit-review-final-02.json",
        "console": str(log.relative_to(ROOT)),
        "optimizer_steps_completed": 0,
        "failure": "First backward rejected non-deterministic adaptive average pool CUDA kernel",
        "elapsed_owned_seconds": None,
        "budget_charged_seconds": charged,
        "accounting": "Conservative observation window from console creation through this journal; actual process time was not measured",
        "final_test_access": "NOT_RUN",
    }
    ledger["runs"].append(record)
    ledger["gpu_hours_spent_owned_scientific_jobs"] += charged / 3600
    ledger["status"] = "PLATFORM_REPAIRS_PENDING_RENEWED_PREFIT"
    temporary = ledger_path.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ledger_path)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
