"""Journal an approved direct CLI invocation and its actual owned process time."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--method", default="shared_ssl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--console", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--tool-session", type=int, default=41703)
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    identifier = args.output.name
    if args.pid:
        if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
            raise ValueError("Another journalled training process is active.")
        review_path = ROOT / "evidence/ssl-research-v1/prefit-review-final-02.json"
        started = psutil.Process(args.pid).create_time()
        record = {
            "id": identifier,
            "status": "RUNNING_CUDA",
            "pid": args.pid,
            "output": str(args.output),
            "device": "cuda:0",
            "method": args.method,
            "seed": 7,
            "history": 96,
            "process_created_epoch": started,
            "started_utc": datetime.fromtimestamp(started, UTC).isoformat(),
            "config": str(args.config or "configs/native_ssl_shared_ssl_seed7_h96_v1.json"),
            "review": str(review_path.relative_to(ROOT)),
            "review_sha256": digest(review_path),
            "binding_verification": str(
                args.verification or "evidence/ssl-research-v1/prefit-bindings-verification-02.json"
            ),
            "console": str(args.console or "evidence/ssl-research-v1/shared-first-job-cuda0.log"),
            "executor": "approved native_ssl CLI with explicit indexed CUDA device",
            "tool_session_id": args.tool_session,
            "gpu_owner": "root",
        }
        ledger["runs"].append(record)
        ledger["status"] = "REAL_SELF_SUPERVISED_TRAINING_RUNNING"
    else:
        if args.exit_code is None:
            raise ValueError("Actual completed process exit code is required.")
        record = next(r for r in ledger["runs"] if r["id"] == identifier)
        if record["status"] != "RUNNING_CUDA":
            raise ValueError("Only an active owned record can be finalized once.")
        elapsed = datetime.now(UTC).timestamp() - record["process_created_epoch"]
        record.update(exit_code=args.exit_code, elapsed_owned_seconds=elapsed)
        report_path = args.output / "run.json"
        if args.exit_code == 0 and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report["status"] != "COMPLETED"
                or report["evidence_kind"] != "REAL_TRAIN_DEVELOPMENT_FIT"
            ):
                raise ValueError("Missing completed real fit, never infer success from exit alone.")
            record.update(
                status="COMPLETED_REAL_CUDA",
                run_sha256=digest(report_path),
                resources=report["resources"],
                selected_pretrain_step=report["selected_pretrain_step"],
                selected_daily_dev_pinball=report["selected_daily_dev_pinball"],
            )
            ledger["scientific_fits_completed"] += 1
            ledger["gpu_hours_completed_scientific_training"] += (
                report["resources"]["elapsed_gpu_seconds"] / 3600
            )
            ledger["status"] = "REAL_NATIVE_SSL_SCREEN_COMPLETED"
        else:
            record["status"] = "FAILED_REAL_CUDA_ATTEMPT"
            ledger["status"] = "REAL_SSL_ATTEMPT_FAILED"
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            ledger.get("gpu_hours_spent_owned_scientific_jobs", 0) + elapsed / 3600
        )
    temporary = LEDGER.with_suffix(".pending")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEDGER)
    print(json.dumps({"id": identifier, "status": record["status"]}))


if __name__ == "__main__":
    main()
