"""Preserve one completed replica and one ownership-guard failure, without fitting."""

import hashlib
import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    queue_path = folder / "band-fixed-replication-queue-attempt-01/queue.json"
    queue = json.loads(queue_path.read_bytes())
    if queue.get("status") != "STOPPED_AFTER_FAILURE" or len(queue.get("jobs", [])) != 2:
        raise ValueError("Actual closed two-attempt partial queue required")
    completed, failed = queue["jobs"]
    if completed.get("status") != "VERIFIED_COMPLETED" or failed.get("actual_exit_code") != 1:
        raise ValueError("Actual completion and failed exit required")
    attempt_path = folder / (failed["id"] + "-attempt-01") / "attempt.json"
    attempt = json.loads(attempt_path.read_bytes())
    resources = attempt["resources_full_attempt"]
    output = ROOT / "outputs/native_acoustic_ssl_v1" / failed["id"]
    ownership_path = output / "gpu-ownership.json"
    ownership = json.loads(ownership_path.read_bytes())
    console = attempt_path.parent / "console.log"
    if (resources.get("exit_code") != 1 or resources.get("stopped_for") is not None
            or resources.get("owned_tree_cleanup_verified") is not True
            or attempt.get("requires_reconciliation") is not False
            or attempt.get("verified_completion") is not False
            or ownership.get("blocking_pids") != [12168]
            or not any(p == {"pid": 12168, "executable": "vlc.exe"} for p in ownership["observed_processes"])
            or "Another GPU runtime or unknown process is active; serialize execution." not in console.read_text(encoding="utf-8", errors="replace")
            or {p.name for p in output.iterdir()} != {"gpu-ownership.json"}
            or any(psutil.pid_exists(pid) for pid in resources["owned_process_pids"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()):
        raise ValueError("Exact closed ownership-guard failure and no checkpoint required")
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    entries = [run for run in ledger["runs"] if run.get("id") == failed["id"] + "-attempt-01"]
    if (len(entries) != 1 or entries[0].get("elapsed_owned_seconds") != resources["elapsed_full_attempt_seconds"]
            or entries[0].get("verified_completion") is not False
            or any(str(run.get("status", "")).startswith("RUNNING_") for run in ledger["runs"])):
        raise ValueError("Closed charged failure must remain in ledger")
    receipt = {
        "status": "ONE_REAL_REPLICATION_COMPLETED_DIRECT13_OWNERSHIP_GUARD_STOPPED",
        "actual_cli_exit_code": 1, "session_id": 88883, "actual_exit_chunk_id": "895b96",
        "queue_sha256": digest(queue_path), "completed": completed,
        "failure_attempt_sha256": digest(attempt_path), "ownership_sha256": digest(ownership_path),
        "console_sha256": digest(console), "failed_full_owned_seconds_retained": resources["elapsed_full_attempt_seconds"],
        "owned_pids_verified_absent": resources["owned_process_pids"],
        "failure_reason": "GPU_OWNERSHIP_GUARD_BLOCKED_VLC_BEFORE_FIRST_UPDATE",
        "foreign_process_terminated": False, "gpu_guard_changed": False,
        "remaining_original_jobs": "SSL23 and direct23 NOT_RUN; direct13 needs separately reviewed exact-recipe fresh-output retry",
        "charged_band_hours_snapshot": ledger["native_band_gpu_hours_spent_full_owned"],
        "charged_aggregate_hours_snapshot": ledger["gpu_hours_spent_owned_scientific_jobs"],
        "independent_numeric_results": "NOT_RUN", "scientific_programme_complete": False,
    }
    with (folder / "band-fixed-replication-partial-closeout-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
