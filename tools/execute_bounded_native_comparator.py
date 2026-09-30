"""Execute one reviewed comparator with full owned-time/RAM enforcement and receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()  # Includes review validation, preparation and final assessment.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lightgbm", "chronos2"), required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--history", type=int, choices=(24, 96), default=96)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    import hashlib

    review = json.loads(args.review.read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT" or args.method not in review.get(
        "allowed_methods", []
    ):
        raise ValueError("Exact distinct comparator approval required.")
    if review["reviewer_session_id"] in (
        review["implementer_session_id"],
        review.get("builder_session_id"),
    ):
        raise ValueError("Comparator review must be a distinct session.")
    for raw, expected in review["bindings"].items():
        with Path(raw).open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"Reviewed comparator binding changed: {raw}")
    for path in (Path(__file__).resolve(), ROOT / "tools/native_reference_supervisor.py"):
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if review["bindings"].get(str(path)) != observed:
            raise ValueError("Independent comparator supervisor bindings required.")
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Serialize fitting/comparator jobs after the active training trajectory.")
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("A GPU training owner is active; do not launch another fitting process.")
    if args.receipt.exists():
        raise ValueError("Comparator receipt already exists; preserve earlier attempts.")
    args.receipt.mkdir(parents=True)
    maximum_seconds = 2 * 3600 if args.method == "chronos2" else 3 * 3600
    if args.method == "chronos2":
        previous_slot_seconds = sum(
            float(r.get("resources_full_attempt", {}).get("elapsed_full_attempt_seconds", 0))
            for r in ledger["runs"]
            if r.get("method") == "chronos2"
        )
        maximum_seconds -= previous_slot_seconds
        remaining = (
            ledger["gpu_limit_hours"] - ledger.get("gpu_hours_spent_owned_scientific_jobs", 0)
        ) * 3600
        maximum_seconds = min(maximum_seconds, remaining)
        if maximum_seconds <= 0 or args.snapshot is None:
            raise ValueError("Chronos requires a snapshot and remaining aggregate GPU budget.")
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-u",
            "tools/execute_native_chronos.py",
            "--dev",
            "data/processed/native_ssl_v1/development.npz",
            "--output",
            str(args.output.resolve()),
            "--ownership-output",
            str((args.receipt / "gpu-ownership").resolve()),
            "--review",
            str(args.review.resolve()),
            "--snapshot",
            str(args.snapshot.resolve()),
            "--history",
            str(args.history),
        ]
    else:
        command = [
            str(ROOT / ".venv/Scripts/python.exe"),
            "-u",
            "-m",
            "marine_echo.training.native_references",
            "--train",
            "data/processed/native_ssl_v1/train.npz",
            "--dev",
            "data/processed/native_ssl_v1/development.npz",
            "--output",
            str(args.output.resolve()),
            "--review",
            str(args.review.resolve()),
            "--method",
            "lightgbm",
            "--history",
            str(args.history),
            "--seed",
            "7",
        ]
    record = {
        "id": args.output.name,
        "status": "RUNNING_CUDA" if args.method == "chronos2" else "RUNNING_CPU_FIT",
        "method": args.method,
        "history": args.history,
        "command": command,
        "output": str(args.output),
        "receipt": str(args.receipt),
        "maximum_full_attempt_seconds": maximum_seconds,
        "rss_limit_bytes": 22 * 2**30,
        "gpu_owner": "root" if args.method == "chronos2" else None,
    }
    with (args.receipt / "console.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        ledger["runs"].append(record)
        temporary = ledger_path.with_suffix(".pending")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
        temporary.replace(ledger_path)
        resources = supervise_owned(
            child, started=started, deadline_seconds=maximum_seconds, rss_limit_bytes=22 * 2**30
        )
    record["resources_full_attempt"] = resources
    # A terminated owned child cannot run its context-manager exit. Only this
    # attempt's proven lock may be cleaned after exit; unknown owners stay intact.
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    if args.method == "chronos2" and resources["stopped_for"] and lock.exists():
        owner = json.loads(lock.read_text(encoding="utf-8"))
        expected_owner_output = str((args.receipt / "gpu-ownership").resolve())
        if (
            resources["owned_tree_cleanup_verified"]
            and owner.get("pid") in resources["owned_process_pids"]
            and owner.get("output") == expected_owner_output
        ):
            lock.unlink()
            record["owned_terminated_child_lock_cleanup"] = True
        else:
            record["unknown_lock_preserved"] = True
    result_path = args.output / "result.json"
    completed = (
        resources["exit_code"] == 0 and resources["stopped_for"] is None and result_path.exists()
    )
    if completed:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_status = (
            "COMPLETED_ZERO_SHOT_DEVELOPMENT"
            if args.method == "chronos2"
            else "COMPLETED_DEVELOPMENT_REFERENCE"
        )
        completed = result.get("status") == expected_status
    record["status"] = "COMPLETED_REVIEWED_COMPARATOR" if completed else "FAILED_COMPARATOR_ATTEMPT"
    if completed:
        record["daily_dev_pinball_db"] = result["metrics"]["primary_pinball_db"]
    if args.method == "chronos2":
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            ledger.get("gpu_hours_spent_owned_scientific_jobs", 0)
            + resources["elapsed_full_attempt_seconds"] / 3600
        )
    temporary = ledger_path.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(ledger_path)
    (args.receipt / "execution.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record))
    sys.exit(0 if completed else 1)


if __name__ == "__main__":
    main()
