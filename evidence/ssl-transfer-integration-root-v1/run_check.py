"""Run bounded CPU commands and retain exact exits without changing research state."""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent


def main():
    name, *arguments = sys.argv[1:]
    if not name or Path(name).name != name:
        raise ValueError("One receipt basename required")
    command = [sys.executable, "-B", *arguments]
    ledger = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()
    started = datetime.now(UTC).isoformat()
    environment = dict(os.environ)
    environment["NATIVE_CF_CONTROLS_ROOT_RUNNER_CHECKS"] = "1"
    environment["NATIVE_PREFIX_MATCHED_ROOT_CHECKS"] = "1"
    sys.path.insert(0, str(ROOT / "tools"))
    import psutil
    from native_reference_supervisor import supervise_owned

    parent = psutil.Process().memory_info()
    parent_peak = max(parent.rss, getattr(parent, "peak_wset", 0))
    remaining_ram = 22 * 2**30 - parent_peak
    if remaining_ram <= 0:
        raise RuntimeError("Owned test launcher already reaches RAM cap")
    with (DEST / (name + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        resources = supervise_owned(
            process, started=time.monotonic(), deadline_seconds=600, rss_limit_bytes=remaining_ram
        )
    exit_code = resources["exit_code"] if resources["stopped_for"] is None else 124
    after = hashlib.sha256(ledger.read_bytes()).hexdigest()
    receipt = {
        "command": command,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "exit_code": exit_code,
        "resources": resources,
        "launcher_peak_rss_bytes": parent_peak,
        "evidence_kind": "CPU_ENGINEERING_ONLY",
        "ledger_sha256_before": before,
        "ledger_sha256_after": after,
    }
    with (DEST / (name + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if before != after:
        raise RuntimeError("Scientific ledger changed")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
