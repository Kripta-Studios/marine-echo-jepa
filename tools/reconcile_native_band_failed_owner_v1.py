"""Reconcile only the exact proven dead random-control owner; retain full failure charges."""

import hashlib
import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "band_random_frozen_frozen_readout_seed7_h96-attempt-01"
OWNED = {33356, 60212, 61568, 62784}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    witness = json.loads((folder / "band-downstream-queue-exit-witness-v1.json").read_bytes())
    if witness.get("session_id") != 5351 or witness.get("actual_cli_exit_code") != 1:
        raise ValueError("Actual closed failed queue required")
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    raw_ledger = ledger_path.read_bytes()
    ledger = json.loads(raw_ledger)
    if any(str(run.get("status", "")).startswith("RUNNING_") for run in ledger["runs"]):
        raise ValueError("No active scientific process may be reconciled")
    matches = [run for run in ledger["runs"] if run.get("id") == IDENTIFIER]
    if len(matches) != 1:
        raise ValueError("Exact recorded failure required")
    run = matches[0]
    resources = run["resources_full_attempt"]
    attempt_path = Path(run["receipt"]) / "attempt.json"
    attempt_raw = attempt_path.read_bytes()
    attempt = json.loads(attempt_raw)
    if (run.get("status") != "FAILED_REAL_BAND_CUDA_ATTEMPT" or not run.get("requires_reconciliation")
            or run.get("budget_family") != "native_band_v1" or run.get("verified_completion") is not False
            or resources.get("exit_code") != 3221227274 or resources.get("owned_tree_cleanup_verified") is not True
            or set(resources.get("owned_process_pids", [])) != OWNED
            or attempt.get("resources_full_attempt") != resources
            or any(psutil.pid_exists(pid) for pid in OWNED)):
        raise ValueError("All exact captured processes must be dead; preserve reused/unknown PIDs")
    lock = (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").resolve()
    if not lock.is_relative_to(ROOT / "evidence"):
        raise ValueError("Ownership target escaped intended evidence directory")
    raw_lock = lock.read_bytes()
    output = (ROOT / "outputs/native_acoustic_ssl_v1/band_random_frozen_frozen_readout_seed7_h96").resolve()
    if (json.loads(raw_lock) != {"pid": 61568, "output": str(output)}
            or Path(run["output"]).resolve() != output
            or "marine_echo.training.native_band_downstream" not in run["command"]
            or str(output) not in run["command"]):
        raise ValueError("Only this exact dead owned file may be removed")
    receipt = folder / "band-random-failed-owner-reconciliation-v1.json"
    pending = ledger_path.with_suffix(".reconciliation-pending")
    if receipt.exists() or any(ledger_path.with_suffix(suffix).exists() for suffix in
                               (".pending", ".prefix-pending", ".band-pending", ".reconciliation-pending")):
        raise FileExistsError("Preserve unknown pending accounting or prior reconciliation")
    record = {"status": "EXACT_DEAD_BAND_OWNER_RECONCILED_FAILURE_CHARGES_PRESERVED", "id": IDENTIFIER,
              "preserved_lock_bytes": raw_lock.decode("utf-8"), "lock_sha256": digest(raw_lock),
              "owned_pids_verified_absent": sorted(OWNED), "actual_child_exit_code": resources["exit_code"],
              "charged_full_owned_seconds": resources["elapsed_full_attempt_seconds"],
              "original_ledger_sha256": digest(raw_ledger), "original_attempt_sha256": digest(attempt_raw),
              "original_failed_ledger_record": json.loads(json.dumps(run)), "failure_cause": "UNKNOWN_NATIVE_PROCESS_EXIT",
              "no_process_termination": True, "junction_operations": False, "scientific_result": "NOT_RUN"}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    run["requires_reconciliation"] = False
    run["ownership_reconciliation"] = {"path": str(receipt), "sha256": digest(receipt.read_bytes()),
                                        "owned_pids_verified_absent": sorted(OWNED)}
    ledger["status"] = "FAILED_BAND_OWNER_RECONCILED_NO_SCIENTIFIC_COMPLETION"
    if (ledger_path.read_bytes() != raw_ledger or lock.read_bytes() != raw_lock
            or attempt_path.read_bytes() != attempt_raw or any(psutil.pid_exists(pid) for pid in OWNED)):
        raise ValueError("Ownership/accounting changed; preserve unknown state")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(ledger, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lock.unlink()
    pending.replace(ledger_path)
    print(json.dumps({"status": record["status"], "charged_full_owned_seconds": record["charged_full_owned_seconds"]}))


if __name__ == "__main__":
    main()
