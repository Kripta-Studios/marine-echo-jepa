"""Bounded actual synthetic check witnesses; no production launch."""
import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
here = Path(__file__).resolve().parent
name, *args = sys.argv[1:]
command = [sys.executable, *args]
result = subprocess.run(command, capture_output=True, timeout=600)
with (here / (name + ".log")).open("xb") as stream:
    stream.write(result.stdout + result.stderr)
with (here / (name + ".json")).open("x", encoding="utf-8") as stream:
    json.dump({"command": command, "exit": result.returncode, "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"}, stream, indent=2)
print((result.stdout + result.stderr).decode("utf-8", errors="replace"))
sys.exit(result.returncode)
