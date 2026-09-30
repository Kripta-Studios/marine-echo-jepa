"""Execute one reviewed CUDA screen and journal its actual exit and GPU budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_ledger(ledger):
    temporary = LEDGER.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT":
        raise ValueError("A distinct exact prefit approval is required.")
    for path, expected in review["bindings"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"A reviewed binding changed: {path}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["method"] not in review.get("allowed_methods", []):
        raise ValueError("Method is outside prefit approval scope.")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if any(run.get("status") == "RUNNING_CUDA" for run in ledger["runs"]):
        raise ValueError("Another journalled CUDA trajectory is active.")
    spent = float(ledger.get("gpu_hours_spent_owned_scientific_jobs", 0))
    remaining_seconds = (float(ledger["gpu_limit_hours"]) - spent) * 3600
    if remaining_seconds <= 0:
        raise ValueError("Aggregate scientific GPU time budget exhausted.")
    args.output.mkdir(parents=True, exist_ok=True)
    record = {
        "id": args.output.name,
        "status": "RUNNING_CUDA",
        "config": str(args.config),
        "config_sha256": digest(args.config),
        "review": str(args.review),
        "review_sha256": digest(args.review),
        "output": str(args.output),
        "method": config["method"],
        "seed": config["seed"],
        "history": config["history"],
        "gpu_owner": "root",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "resume": str(args.resume) if args.resume else None,
    }
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "-m",
        "marine_echo.training.native_ssl",
        "--train",
        "data/processed/native_ssl_v1/train.npz",
        "--dev",
        "data/processed/native_ssl_v1/development.npz",
        "--split",
        "configs/native_ssl_split_v1.json",
        "--protocol",
        "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        "--config",
        str(args.config.resolve()),
        "--review",
        str(args.review.resolve()),
        "--output",
        str(args.output.resolve()),
        "--device",
        "cuda",
    ]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    log = args.output / ("resume-console.log" if args.resume else "console.log")
    with log.open("x", encoding="utf-8") as stream:
        started = time.monotonic()
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        record["command"] = command
        ledger["runs"].append(record)
        ledger["status"] = "REAL_SELF_SUPERVISED_TRAINING_RUNNING"
        write_ledger(ledger)
        print(
            json.dumps({"status": record["status"], "pid": child.pid, "output": str(args.output)}),
            flush=True,
        )
        timed_out = False
        while child.poll() is None:
            if time.monotonic() - started >= remaining_seconds:
                child.terminate()  # Only the explicitly owned child, never user applications.
                timed_out = True
                break
            time.sleep(2)
        exit_code = child.wait()
        elapsed = time.monotonic() - started
    record.update(exit_code=exit_code, elapsed_owned_seconds=elapsed, budget_exceeded=timed_out)
    report = args.output / "run.json"
    if exit_code == 0 and report.exists():
        result = json.loads(report.read_text(encoding="utf-8"))
        if (
            result["status"] != "COMPLETED"
            or result["evidence_kind"] != "REAL_TRAIN_DEVELOPMENT_FIT"
        ):
            raise ValueError("Child exited successfully without a completed real-fit artifact.")
        record.update(
            status="COMPLETED_REAL_CUDA",
            run_sha256=digest(report),
            resources=result["resources"],
            selected_pretrain_step=result["selected_pretrain_step"],
            selected_daily_dev_pinball=result["selected_daily_dev_pinball"],
        )
        ledger["scientific_fits_completed"] = ledger.get("scientific_fits_completed", 0) + 1
        ledger["gpu_hours_completed_scientific_training"] = (
            ledger.get("gpu_hours_completed_scientific_training", 0)
            + result["resources"]["elapsed_gpu_seconds"] / 3600
        )
    else:
        record["status"] = "FAILED_REAL_CUDA_ATTEMPT"
    ledger["gpu_hours_spent_owned_scientific_jobs"] = spent + elapsed / 3600
    ledger["status"] = (
        "REAL_SSL_SCREEN_COMPLETED" if exit_code == 0 else "REAL_SSL_SCREEN_ATTEMPT_FAILED"
    )
    write_ledger(ledger)
    print(json.dumps(record), flush=True)
    sys.exit(exit_code or (1 if timed_out else 0))


if __name__ == "__main__":
    main()
