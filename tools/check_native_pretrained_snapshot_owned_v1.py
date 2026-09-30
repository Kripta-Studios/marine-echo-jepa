"""Supervise isolated package CPU correctness within existing local RAM limits."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.monotonic()
    folder = ROOT / "evidence/ssl-research-v1"
    snapshot = json.loads((folder / "native-pretrained-model-snapshot-v1.json").read_bytes())
    bundle = Path(snapshot["directory"])
    for name, sha in snapshot["files"].items():
        if digest(bundle / name) != sha:
            raise ValueError("Copied package bytes changed")
    destination = folder / "native-pretrained-snapshot-isolated-cpu-v1"
    if destination.exists():
        raise FileExistsError("Preserve actual package correctness attempts")
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    if (any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Prior scientific owner and all journals must be closed")
    destination.mkdir()
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-I", "-B", str(ROOT / "tools/check_native_pretrained_snapshot_worker_v1.py"),
               str(bundle), str(destination / "completion.json")]
    with (destination / "stdout.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT)
        resources = supervise_owned(child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 1024**3)
    receipt = {"command": command, "resources": resources, "fitting": False, "device": "cpu",
               "snapshot_sha256": digest(folder / "native-pretrained-model-snapshot-v1.json"),
               "evidence_kind": "SYNTHETIC_INPUT_CORRECTNESS_ONLY_REAL_PRETRAINED_WEIGHTS"}
    with (destination / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"actual_exit_code": resources["exit_code"], "stopped_for": resources["stopped_for"],
                      "owned_tree_cleanup_verified": resources["owned_tree_cleanup_verified"]}))
    return 0 if resources["exit_code"] == 0 and resources["stopped_for"] is None and resources["owned_tree_cleanup_verified"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
