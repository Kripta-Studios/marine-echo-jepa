"""Interrupt only the exact root-launched builder tree; retain all files/evidence."""

import datetime
import json
from pathlib import Path

import psutil

SESSION = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt_path = ROOT / "evidence/ssl-research-v1/builder-owned-interruption-01.json"
    if receipt_path.exists():
        raise ValueError("Preserve earlier interruption evidence")
    parent = psutil.Process(4488)
    if parent.name().lower() != "codex.exe" or SESSION not in parent.cmdline():
        raise ValueError("Exact owned builder identity no longer matches")
    processes = [parent, *parent.children(recursive=True)]
    identities = [(process.pid, process.create_time(), process.name()) for process in processes]
    if any(name.lower() in ("python.exe", "pythonw.exe") for _, _, name in identities):
        raise ValueError("Unexpected Python child; preserve tree without interruption")
    receipt = {"status": "INTERRUPTING_EXACT_OWNED_AGENT_TREE", "session_id": SESSION,
               "utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "reason": "Frozen source proof retained; builder event stream remains partial and final handoff absent",
               "cause_of_stall": "UNKNOWN", "owned_process_identities": identities,
               "scientific_queue_modified": False, "files_deleted": False, "termination_errors": []}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for pid, created, _ in reversed(identities):
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                raise ValueError("PID identity changed; do not touch replacement process")
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except Exception as error:
            receipt["termination_errors"].append({"pid": pid, "error": str(error)})
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            raise
    _, alive = psutil.wait_procs(processes, timeout=10)
    receipt["remaining_owned_pids"] = [process.pid for process in alive]
    receipt["status"] = "OWNED_AGENT_TREE_INTERRUPTED" if not alive else "OWNED_AGENT_EXIT_PENDING"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
