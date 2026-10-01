"""Run a supplied bounded local check and preserve its actual exit witness."""
import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
name, *arguments = sys.argv[1:]
command = [sys.executable, *arguments]
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
with (here / (name + ".log")).open("x", encoding="utf-8") as stream:
    stream.write(result.stdout + result.stderr)
with (here / (name + ".json")).open("x", encoding="utf-8") as stream:
    json.dump({"command": command, "exit_code": result.returncode, "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"}, stream, indent=2)
print(result.stdout + result.stderr)
print(json.dumps({"exit_code": result.returncode, "log": name + ".log"}))
sys.exit(result.returncode)
