"""Close only the exact stalled root-launched engineering agent tree."""

import datetime
import json
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SESSION = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    destination = folder / "native-transfer-builder-owned-interruption-v1.json"
    events = folder / "native-transfer-corpus-builder-events-v1b.jsonl"
    if destination.exists() or time.time() - events.stat().st_mtime < 1200:
        raise ValueError("Preserve evidence and require twenty-minute idle stream")
    parent = psutil.Process(34108)
    command = parent.cmdline()
    if (parent.name().lower() != "codex.exe" or SESSION not in command
            or not any("native-transfer-corpus-builder-final-v1b.json" in arg for arg in command)
            or "workspace-write" not in command):
        raise ValueError("Exact root-launched engineering agent identity differs")
    processes = [parent, *parent.children(recursive=True)]
    identities = [(p.pid, p.create_time(), p.name()) for p in processes]
    if any(name.lower() in ("python.exe", "pythonw.exe", "ruff.exe") for _, _, name in identities):
        raise ValueError("Unexpected active implementation command; preserve the tree")
    receipt = {
        "status": "INTERRUPTING_EXACT_OWNED_ENGINEERING_AGENT_TREE",
        "session_id": SESSION, "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "reason": "Format-check tool has not returned for twenty minutes; no active Python or Ruff child",
        "cause_of_stall": "UNKNOWN", "owned_process_identities": identities,
        "scientific_queue_modified": False, "files_deleted": False,
        "source_delivery": "NOT_CLOSED", "pending_format_check": "NO_ACTUAL_EXIT_WITNESS",
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    for pid, created, _ in reversed(identities):
        try:
            process = psutil.Process(pid)
            if process.create_time() != created:
                raise ValueError("PID reused; preserve replacement")
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=10)
    receipt["remaining_owned_pids"] = [p.pid for p in alive]
    receipt["status"] = "OWNED_ENGINEERING_AGENT_INTERRUPTED" if not alive else "OWNED_AGENT_EXIT_PENDING"
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
