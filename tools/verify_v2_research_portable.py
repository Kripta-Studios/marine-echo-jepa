"""Verify a relocated v2 research archive with offline install and local API startup."""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import psutil

archive, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if output.exists():
    raise FileExistsError("Relocation test requires a new destination.")
output.mkdir(parents=True)
with zipfile.ZipFile(archive) as packed:
    if packed.testzip() is not None:
        raise ValueError("Archive CRC verification failed.")
    for info in packed.infolist():
        candidate = output / info.filename
        if not candidate.resolve().is_relative_to(output) or ".." in Path(info.filename).parts:
            raise ValueError("Unsafe package path.")
    packed.extractall(output)
env = dict(os.environ)
for name in ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME"):
    env.pop(name, None)
env.update(UV_OFFLINE="1", PYTHONUTF8="1")
report = {
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "relocated": str(output),
    "network_policy": "uv offline/no-index; browser test denies non-loopback requests",
    "steps": [],
}
logs = archive.parent / (archive.stem + "-validation")
logs.mkdir(exist_ok=True)


def verify() -> int:
    return subprocess.run(
        [sys.executable, str(output / "Verify-Release.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ).returncode


assert verify() == 0
sample = output / "research/artifacts/raw-development.json"
original = sample.read_bytes()
sample.write_bytes(original + b"corruption")
try:
    assert verify() == 2
finally:
    sample.write_bytes(original)
assert verify() == 0
report["integrity_clean_corrupt_restored"] = [0, 2, 0]
port = 8772
for phase in ("first_offline_install_and_start", "cold_start_existing_environment"):
    started = time.perf_counter()
    with (logs / f"{phase}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(output / "Run-V2-Research.ps1"),
                "-Port",
                str(port),
            ],
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            health = None
            while time.perf_counter() - started < 60:
                if process.poll() is not None:
                    raise RuntimeError(f"Launcher exited {process.returncode}; see {phase}.log")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=0.5
                    ) as response:
                        health = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.05)
            if health is None:
                raise TimeoutError("Local research service did not start in 60 seconds.")
            elapsed = time.perf_counter() - started
            assert health["release_class"] == "OFFLINE_RESEARCH_ENGINEERING_ONLY"
            if phase == "cold_start_existing_environment":
                assert elapsed < 30
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/experiments") as response:
                registry = json.load(response)
            assert len(registry) == 25
            assert all(row["status"] == "BLOCKED" and row["updates"] == 0 for row in registry)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/v1/evidence/raw-development"
            ) as response:
                development = json.load(response)
            assert len(development["rows"]) == 212
            assert development["final_evaluation"] is False
            report["steps"].append(
                {
                    "phase": phase,
                    "seconds": elapsed,
                    "health": health,
                    "raw_development_rows": len(development["rows"]),
                    "historical_blocked_registry_rows": len(registry),
                }
            )
        finally:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in reversed(children):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            psutil.wait_procs(children, timeout=10)
report["status"] = "PASS_OFFLINE_RESEARCH_ENGINEERING_ONLY"
(logs / "portable.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
