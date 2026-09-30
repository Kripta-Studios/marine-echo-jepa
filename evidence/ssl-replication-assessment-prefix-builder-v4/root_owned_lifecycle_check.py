"""ROOT-only actual owned supervisor check; SYNTHETIC_CORRECTNESS_ONLY.

This is a tiny CPU process check, not an assessment worker/model/data replay.
Builder execution is explicitly refused. ROOT must be idle before invocation.
"""

import hashlib
import importlib.util
import json
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError("NOT_RUN builder: actual owned-process checks are ROOT-only.")
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists() or any(
        str(r.get("status", "")).startswith("RUNNING_") or r.get("requires_reconciliation")
        for r in ledger["runs"]
    ):
        raise RuntimeError("ROOT must be idle; no competing process or owner action.")
    supervisor_path = ROOT / "tools/native_reference_supervisor.py"
    proof = json.loads(Path(__file__).with_name("source-proof-closed-v4.json").read_bytes())
    expected = proof["source_closure"][str(supervisor_path)]
    if hashlib.sha256(supervisor_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Supervisor differs from the delivered source closure.")
    spec = importlib.util.spec_from_file_location("root_actual_owned_supervisor", supervisor_path)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    folder = Path(__file__).parent / ("SYNTHETIC_CORRECTNESS_ONLY_owned_" + uuid.uuid4().hex)
    folder.mkdir(exist_ok=False)
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-B",
        "-u",
        "-c",
        "import time; print('SYNTHETIC_CORRECTNESS_ONLY', flush=True); time.sleep(0.15)",
    ]
    started = time.monotonic()
    with (folder / "console.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = supervisor.supervise_owned(
            child, started=started, deadline_seconds=10, rss_limit_bytes=22 * 2**30
        )
    if result["exit_code"] != 0 or result["owned_tree_cleanup_verified"] is not True:
        raise RuntimeError("Actual owned synthetic CPU process check failed.")
    with (folder / "resources.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "command": command,
                "resources": result,
                "not_worker_or_scientific_evidence": True,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "SYNTHETIC_OWNED_SUPERVISOR_CHECK_PASSED",
                "receipt": str(folder),
                "exit_code": result["exit_code"],
            }
        )
    )


if __name__ == "__main__":
    main()
