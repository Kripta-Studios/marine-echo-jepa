"""Record bounded CPU review checks without touching the scientific ledger."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = digest(LEDGER)
    environment = dict(os.environ)
    environment["NATIVE_CF_CONTROLS_ROOT_RUNNER_CHECKS"] = "1"
    suites = {
        "existing-control-checks": [
            "tests/unit/test_native_cf_controls.py",
            "tests/integration/test_native_cf_controls.py",
            "tests/unit/test_native_cf_control_budget_v1.py",
            "tests/unit/test_native_cf_control_execution_v1.py",
            "tests/unit/test_native_seed_summary_v1.py",
            "tests/unit/test_native_cf_control_registration_v1.py",
            "tests/unit/test_native_cf_control_prefit_v1.py",
        ],
        "transfer-integration-gaps": [
            "evidence/ssl-coordinator-review-v2/test_transfer_integration_gaps.py"
        ],
    }
    for name, paths in suites.items():
        command = [
            sys.executable, "-B", "-m", "pytest", "--import-mode=importlib",
            "-q", "-p", "no:cacheprovider", *paths, "--tb=short",
        ]
        start = datetime.now(timezone.utc).isoformat()
        with (DEST / (name + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=180, check=False,
            )
        receipt = {
            "evidence_kind": "COORDINATOR_CPU_CHECKS_NOT_INDEPENDENT_APPROVAL",
            "started_utc": start,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "exit_code": process.returncode,
            "ledger_sha256_before": before,
            "ledger_sha256_after": digest(LEDGER),
            "log_sha256": digest(DEST / (name + ".log")),
            "new_real_fits": 0,
            "reserved_numerical_access": False,
        }
        with (DEST / (name + ".json")).open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"suite": name, "exit_code": process.returncode}), flush=True)
    if digest(LEDGER) != before:
        raise RuntimeError("Scientific ledger unexpectedly changed")


if __name__ == "__main__":
    main()
