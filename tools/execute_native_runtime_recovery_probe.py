"""Own and charge the existing tiny synthetic CUDA preflight after a native exit."""

import json
import subprocess
import sys
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"


def save(ledger):
    pending = LEDGER.with_suffix(".pending")
    pending.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    pending.replace(LEDGER)


def main():
    started = time.monotonic()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Another scientific operation is active")
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("Preserve existing scientific owner")
    spent = ledger["gpu_hours_spent_owned_scientific_jobs"]
    remaining = (ledger["gpu_limit_hours"] - spent) * 3600
    if remaining <= 0:
        raise ValueError("Aggregate allowance exhausted")
    folder = ROOT / "evidence/ssl-research-v1/cf-native-runtime-recovery-probe-01"
    folder.mkdir()
    report = folder / "runtime.json"
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "tools/runtime_preflight.py", "--torch", "--output", str(report)]
    record = {"id": folder.name, "status": "RUNNING_CUDA", "command": command,
              "operation": "EXISTING_TINY_SYNTHETIC_CUDA_KERNEL_PREFLIGHT", "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
              "public_data": False, "gpu_owner": "root", "resources": "full-owned120sec/22GiB"}
    ledger["runs"].append(record)
    save(ledger)
    with (folder / "console.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        save(ledger)
        resources = supervise_owned(child, started=started, deadline_seconds=min(remaining, 120), rss_limit_bytes=22 * 2**30)
    valid = resources["exit_code"] == 0 and resources["stopped_for"] is None and report.exists()
    if valid:
        probe = json.loads(report.read_text(encoding="utf-8"))
        valid = probe["torch"].get("cuda_training_verified") is True and probe["torch"].get("peak_allocated_gib", 10) < 10
    record.update(status="COMPLETED_SYNTHETIC_GPU_PREFLIGHT" if valid else "FAILED_SYNTHETIC_GPU_PREFLIGHT",
                  resources_full_attempt=resources, native_failure_cause="UNKNOWN; probe only checks current tiny kernels")
    ledger["gpu_hours_spent_owned_scientific_jobs"] = spent + resources["elapsed_full_attempt_seconds"] / 3600
    save(ledger)
    with (folder / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "resources": resources, "public_fit": False}))
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
