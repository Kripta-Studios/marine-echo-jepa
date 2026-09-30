"""Execute one distinctly reviewed transfer endpoint with an owned tree supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

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
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "review", "output", "receipt"):
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ("encoder", "ancestor-review", "ancestor-config", "resume"):
        parser.add_argument("--" + flag, type=Path)
    args = parser.parse_args()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_DOWNSTREAM_PREFIT":
        raise ValueError("Separate downstream prefit approval required.")
    if not review.get("reviewer_session_id") or review["reviewer_session_id"] in (
        review.get("implementer_session_id"),
        review.get("coordinator_session_id"),
    ):
        raise ValueError("Downstream reviewer must be distinct from every author.")
    if config["method"] not in review.get("allowed_methods", []) or config[
        "mode"
    ] not in review.get("allowed_modes", []):
        raise ValueError("Transfer endpoint is outside review scope.")
    for path, expected in review["bindings"].items():
        if digest(path) != expected:
            raise ValueError(f"Changed downstream prefit binding: {path}")
    for path in (
        Path(__file__).resolve(),
        ROOT / "tools/native_reference_supervisor.py",
        args.config.resolve(),
    ):
        if review["bindings"].get(str(path)) != digest(path):
            raise ValueError("Transfer supervisor/config bindings absent.")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Another scientific process is active.")
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists():
        raise ValueError("Existing GPU owner must remain undisturbed.")
    spent = float(ledger.get("gpu_hours_spent_owned_scientific_jobs", 0))
    remaining = (float(ledger["gpu_limit_hours"]) - spent) * 3600
    if remaining <= 0 or (args.output.exists() and args.resume is None) or args.receipt.exists():
        raise ValueError("No budget or output already exists; preserve earlier attempts.")
    args.receipt.mkdir(parents=True)
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "-m",
        "marine_echo.training.native_downstream",
    ]
    inputs = {
        "train": "data/processed/native_ssl_v1/train.npz",
        "dev": "data/processed/native_ssl_v1/development.npz",
        "train-cohort": "data/processed/native_ssl_v1/train.json",
        "dev-cohort": "data/processed/native_ssl_v1/development.json",
        "split": "configs/native_ssl_split_v1.json",
        "adr0016": "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
        "protocol": "docs/adr/0018-native-acoustic-downstream-transfer.md",
        "config": args.config.resolve(),
        "review": args.review.resolve(),
        "output": args.output.resolve(),
        "device": "cuda:0",
    }
    for flag in ("encoder", "ancestor-review", "ancestor-config", "resume"):
        value = getattr(args, flag.replace("-", "_"))
        if value is not None:
            inputs[flag] = value.resolve()
    for flag, value in inputs.items():
        command += ["--" + flag, str(value)]
    record = {
        "id": args.output.name,
        "status": "RUNNING_CUDA",
        "method": config["method"],
        "mode": config["mode"],
        "seed": config["seed"],
        "history": config["history"],
        "output": str(args.output),
        "receipt": str(args.receipt),
        "review": str(args.review),
        "review_sha256": digest(args.review),
        "config_sha256": digest(args.config),
        "command": command,
        "gpu_owner": "root",
    }
    with (args.receipt / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        ledger["runs"].append(record)
        ledger["status"] = "REAL_DOWNSTREAM_TRANSFER_RUNNING"
        write_ledger(ledger)
        print(json.dumps(record), flush=True)
        resources = supervise_owned(
            child, started=started, deadline_seconds=remaining, rss_limit_bytes=22 * 2**30
        )
    record["resources_full_attempt"] = resources
    if resources["stopped_for"] and lock.exists():
        owner = json.loads(lock.read_text(encoding="utf-8"))
        if (
            resources["owned_tree_cleanup_verified"]
            and owner.get("pid") in resources["owned_process_pids"]
            and owner.get("output") == str(args.output.resolve())
        ):
            lock.unlink()
            record["owned_terminated_tree_lock_cleanup"] = True
        else:
            record["unknown_lock_preserved"] = True
    report_path = args.output / "run.json"
    completed = (
        resources["exit_code"] == 0 and resources["stopped_for"] is None and report_path.exists()
    )
    if completed:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        completed = (
            report.get("status") == "COMPLETED"
            and report.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
            and report.get("mode") == config["mode"]
        )
    record["status"] = (
        "COMPLETED_REAL_DOWNSTREAM" if completed else "FAILED_REAL_DOWNSTREAM_ATTEMPT"
    )
    ledger["gpu_hours_spent_owned_scientific_jobs"] = (
        spent + resources["elapsed_full_attempt_seconds"] / 3600
    )
    if completed:
        record.update(
            run_sha256=digest(report_path),
            selected_daily_dev_pinball=report["selected_daily_dev_pinball"],
            selected_supervised_step=report["selected_supervised_step"],
            supervised_updates=report["supervised_updates"],
            resources_additional_downstream=report["resources_additional_downstream"],
        )
        ledger["scientific_fits_completed"] += 1
        ledger["gpu_hours_completed_scientific_training"] += (
            report["resources_additional_downstream"]["elapsed_gpu_seconds"] / 3600
        )
    ledger["status"] = "DOWNSTREAM_TRANSFER_COMPLETED" if completed else "DOWNSTREAM_ATTEMPT_FAILED"
    write_ledger(ledger)
    print(json.dumps(record), flush=True)
    sys.exit(0 if completed else 1)


if __name__ == "__main__":
    main()
