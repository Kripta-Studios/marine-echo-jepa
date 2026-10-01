"""Capture an actual ROOT correctness command; not a scientific review."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

folder = Path(__file__).resolve().parent
name, *args = sys.argv[1:]
if not name or Path(name).name != name:
    raise ValueError("Simple fresh evidence name required")
log = folder / (name + ".log")
receipt = folder / (name + ".json")
if log.exists() or receipt.exists():
    raise FileExistsError("Preserve previous verification")
started = datetime.now(timezone.utc).isoformat()
with log.open("xb") as stream:
    result = subprocess.run([sys.executable, "-B", *args], stdout=stream,
                            stderr=subprocess.STDOUT, cwd=folder.parents[1])
with receipt.open("x", encoding="utf-8", newline="\n") as stream:
    json.dump({"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, "-B", *args], "exit_code": result.returncode,
               "log": str(log), "evidence_kind": "CPU_SOFTWARE_CORRECTNESS_NOT_SCIENTIFIC_REVIEW"},
              stream, indent=2)
    stream.write("\n")
print(json.dumps({"name": name, "exit_code": result.returncode}))
raise SystemExit(result.returncode)
