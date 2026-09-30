"""Serialize exact reviewed band commands using the real supervised executor."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    if (
        queue.get("kind") != "reviewed_native_band_serial_queue_v1"
        or len(queue.get("jobs", [])) != 5
        or any(j["entrypoint"] != "tools/execute_native_band_job.py" for j in queue["jobs"])
    ):
        raise ValueError("Exactly five previously approved supervising commands required")
    ledger = json.loads(
        (ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8")
    )
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Wait for actual prior scientific completion")
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("Preserve existing scientific owner")
    target = args.receipt.resolve()
    if not target.is_relative_to(ROOT / "evidence"):
        raise ValueError("Root-owned queue evidence required")
    target.mkdir()
    result = {
        "status": "RUNNING",
        "queue_sha256": hashlib.sha256(args.queue.read_bytes()).hexdigest(),
        "jobs": [],
    }
    destination = target / "queue.json"
    for job in queue["jobs"]:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", job["entrypoint"], *job["args"]]
        print(json.dumps({"event": "LAUNCH", "id": job["id"], "command": command}), flush=True)
        with (target / (job["id"] + ".log")).open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        report_path = Path(job["report"])
        valid = completed.returncode == 0 and report_path.exists()
        entry = {"id": job["id"], "exit_code": completed.returncode}
        if valid:
            raw = report_path.read_bytes()
            report = json.loads(raw)
            config = report.get("config", {})
            valid = (
                report.get("status") == "COMPLETED"
                and report.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
                and report.get("architecture") == "nonlinear_frequency_conditioned_v1"
                and config.get("method") == job["method"]
                and config.get("seed") == 7
            )
            entry.update(
                run_sha256=hashlib.sha256(raw).hexdigest(),
                development_pinball_db=report.get("selected_daily_dev_pinball"),
            )
        entry["status"] = "VERIFIED_COMPLETED" if valid else "FAILED_QUEUE_STOPPED"
        result["jobs"].append(entry)
        result["status"] = "RUNNING" if valid else "STOPPED_AFTER_FAILURE"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if not valid:
            sys.exit(1)
    result["status"] = "COMPLETED_REVIEWED_BAND_SERIAL_QUEUE"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
