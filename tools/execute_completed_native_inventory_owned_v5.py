"""Own the fixed historical 35-endpoint CPU audit after scientific closure."""

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
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    if (ledger.get("requires_reconciliation")
            or any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Scientific ownership and every journal must be closed")
    manifest = ROOT / "orchestration/native_completed_inventory_v5.json"
    document = json.loads(manifest.read_bytes())
    output = ROOT / "evidence/native-completed-inventory-v5"
    receipt = ROOT / "evidence/ssl-research-v1/native-completed-inventory-owned-v5"
    if document.get("output_path") != str(output) or len(document.get("endpoints", {})) != 40 or output.exists() or receipt.exists():
        raise ValueError("Completed40 metadata inventory and fresh evidence required")
    if str(ledger_path) in document["bindings"]:
        raise ValueError("Bind only the immutable ledger capture, never the mutable ledger")
    for path, expected in document["bindings"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Immutable input changed: {path}")
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-B", str(ROOT / "tools/prepare_native_research_inventory.py"),
               "--manifest", str(manifest), "--output", str(output)]
    receipt.mkdir()
    with (receipt / "stdout.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        resources = supervise_owned(child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 1024**3)
    record = {"status": "CLOSED_CPU_COMPLETED40_ARTIFACT_AUDIT", "command": command, "resources": resources,
              "manifest_sha256": digest(manifest), "device": "cpu", "fitting": False, "scientific_prefit": False,
              "complete_programme_freeze": False,
              "source_bindings": {str(path): digest(path) for path in (Path(__file__).resolve(), ROOT / "tools/native_reference_supervisor.py")}}
    with (receipt / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    completed = resources.get("exit_code") == 0 and resources.get("stopped_for") is None and resources.get("owned_tree_cleanup_verified") is True
    print(json.dumps({"status": "COMPLETED40_ARTIFACT_AUDIT_PASSED" if completed else "ARTIFACT_AUDIT_FAILED",
                      "actual_exit_code": resources["exit_code"], "stopped_for": resources["stopped_for"]}))
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
