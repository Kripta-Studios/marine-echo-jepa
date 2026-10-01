"""Run bounded CPU commands and retain exact exits without changing research state."""

import hashlib
import json
import os
import subprocess
import sys
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
    with (DEST / (name + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    after = hashlib.sha256(ledger.read_bytes()).hexdigest()
    receipt = {
        "command": command,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "exit_code": process.returncode,
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
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
