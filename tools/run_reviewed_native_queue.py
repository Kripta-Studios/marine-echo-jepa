"""Serialize a fixed reviewed command list and inspect every actual result."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "tools/execute_native_ssl_job.py",
    "tools/execute_native_downstream_job.py",
    "tools/execute_bounded_native_comparator.py",
}
STATUSES = {"COMPLETED", "COMPLETED_DEVELOPMENT_REFERENCE", "COMPLETED_ZERO_SHOT_DEVELOPMENT"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw = args.queue.read_bytes()
    queue = json.loads(raw)
    if queue.get("kind") != "reviewed_native_serial_queue_v1" or args.receipt.exists():
        raise ValueError("Require a fixed queue and a new coordinator receipt.")
    jobs = queue["jobs"]
    if not jobs or any(job["entrypoint"] not in ENTRYPOINTS for job in jobs):
        raise ValueError("Only distinctly reviewed supervising entrypoints are admitted.")
    args.receipt.mkdir(parents=True)
    results = {
        "queue_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "RUNNING",
        "jobs": [],
        "app_release_work": "FROZEN",
    }
    receipt = args.receipt / "queue.json"
    receipt.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for job in jobs:
        # Each required wrapper independently verifies all approvals, bindings,
        # ledger ownership and the remaining finite budget immediately at launch.
        command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", job["entrypoint"], *job["args"]]
        print(json.dumps({"event": "LAUNCH", "id": job["id"], "command": command}), flush=True)
        with (args.receipt / (job["id"] + ".log")).open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        report_path = ROOT / job["report"]
        result = {"id": job["id"], "exit_code": completed.returncode, "report": job["report"]}
        valid = completed.returncode == 0 and report_path.is_file()
        if valid:
            report_raw = report_path.read_bytes()
            report = json.loads(report_raw)
            method = report.get("method", report.get("config", {}).get("method"))
            if (
                report.get("status") == "COMPLETED_ZERO_SHOT_DEVELOPMENT"
                and report.get("weights_fitted_in_this_study") is False
                and report.get("cross_learning") is False
            ):
                method = "chronos2"
            valid = report.get("status") in STATUSES and method == job["method"]
            if job.get("mode"):
                valid = valid and report.get("mode") == job["mode"]
            result.update(
                report_sha256=hashlib.sha256(report_raw).hexdigest(),
                report_status=report.get("status"),
                development_pinball_db=report.get(
                    "selected_daily_dev_pinball",
                    report.get("metrics", {}).get("primary_pinball_db"),
                ),
                scientific_claim="NOT_ESTABLISHED",
            )
        result["status"] = "VERIFIED_COMPLETED" if valid else "FAILED_QUEUE_STOPPED"
        results["jobs"].append(result)
        results["status"] = "RUNNING" if valid else "STOPPED_AFTER_FAILURE"
        receipt.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)
        if not valid:
            sys.exit(1)
    results["status"] = "COMPLETED_REVIEWED_SERIAL_QUEUE"
    receipt.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
