"""Close an idle exact owned delivery agent; never touch a scientific process."""

import datetime
import json
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SESSION = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"


def main():
    receipt_path = ROOT / "evidence/ssl-research-v1/assessment-builder-owned-interruption-01.json"
    events = ROOT / "evidence/ssl-research-v1/heldout-executor-builder-events.jsonl"
    if receipt_path.exists() or time.time() - events.stat().st_mtime < 1200:
        raise ValueError("Preserve prior evidence and require twenty-minute idle stream")
    parent = psutil.Process(58036)
    command = parent.cmdline()
    if parent.name().lower() != "codex.exe" or SESSION not in command or not any(
        "heldout-executor-builder-final.txt" in argument for argument in command
    ):
        raise ValueError("Exact root-launched assessment-builder identity differs")
    processes = [parent, *parent.children(recursive=True)]
    identities = [(p.pid, p.create_time(), p.name()) for p in processes]
    if any(name.lower() in ("python.exe", "pythonw.exe") for _, _, name in identities):
        raise ValueError("Unexpected Python child; preserve the tree")
    receipt = {"status": "INTERRUPTING_EXACT_OWNED_AGENT_TREE", "session_id": SESSION,
               "utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "reason": "Completed source proof and 92 root checks retained; delivery stream idle over twenty minutes, no final handoff",
               "cause_of_stall": "UNKNOWN", "owned_process_identities": identities,
               "scientific_queue_modified": False, "files_deleted": False}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    for pid, created, _ in reversed(identities):
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                raise ValueError("PID reused; never touch replacement")
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    receipt["remaining_owned_pids"] = [p.pid for p in alive]
    receipt["status"] = "OWNED_AGENT_TREE_INTERRUPTED" if not alive else "OWNED_AGENT_EXIT_PENDING"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "owned_processes": len(identities),
                      "scientific_queue_modified": False, "remaining_owned_pids": receipt["remaining_owned_pids"]}))


if __name__ == "__main__":
    main()
