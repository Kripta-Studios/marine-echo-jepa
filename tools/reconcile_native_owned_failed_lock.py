"""Remove only an exact closed failed job's orphaned local ownership lock."""

import argparse
import hashlib
import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if receipt.exists() or not receipt.is_relative_to(ROOT / "evidence"):
        raise ValueError("New root-owned evidence receipt required")
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8"))
    matches = [r for r in ledger["runs"] if r.get("id") == args.id]
    if len(matches) != 1 or matches[0].get("status") != "FAILED_REAL_DOWNSTREAM_ATTEMPT":
        raise ValueError("Exactly one closed failed supervised job required")
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Do not reconcile while a scientific job is active")
    run = matches[0]
    resources = run["resources_full_attempt"]
    pids = resources["owned_process_pids"]
    if not resources.get("owned_tree_cleanup_verified") or not pids or any(psutil.pid_exists(p) for p in pids):
        raise ValueError("Captured owned processes must all be gone; preserve reused/unknown PIDs")
    lock = (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").resolve()
    raw = lock.read_bytes()
    owner = json.loads(raw)
    output = (ROOT / run["output"]).resolve()
    if owner.get("pid") not in pids or owner.get("output") != str(output):
        raise ValueError("Unknown ownership lock must be preserved")
    command = run["command"]
    if "marine_echo.training.native_downstream" not in command or str(output) not in command:
        raise ValueError("Recorded command does not identify this exact owned output")
    if lock.read_bytes() != raw:
        raise ValueError("Ownership changed; preserve lock")
    lock.unlink()
    record = {"status": "EXACT_CLOSED_FAILED_OWNER_LOCK_RECONCILED", "id": args.id,
              "preserved_lock_bytes": raw.decode(), "lock_sha256": hashlib.sha256(raw).hexdigest(),
              "captured_owned_pids_all_gone": pids, "actual_child_exit_code": resources["exit_code"],
              "actual_child_exit_hex": hex(resources["exit_code"] & 0xffffffff),
              "charged_owned_seconds": resources["elapsed_full_attempt_seconds"],
              "failure_cause": "UNKNOWN_NATIVE_PROCESS_EXIT_NO_TRACEBACK",
              "failure_artifacts_preserved": True, "scientific_result": "NOT_RUN",
              "ledger_modified": False, "junction_operations": False}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
