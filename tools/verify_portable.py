"""Extract our package into a new location and test integrity and offline startup."""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import psutil

archive, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if output.exists():
    raise FileExistsError("Relocation test requires a new empty destination.")
output.mkdir(parents=True)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for info in z.infolist():
        candidate = output / info.filename
        if not candidate.resolve().is_relative_to(output) or ".." in Path(info.filename).parts:
            raise ValueError("Unsafe package path.")
    z.extractall(output)
env = dict(os.environ)
env.pop("PYTHONHOME", None)
env.pop("UV_INTERNAL__PYTHONHOME", None)
env.pop("PYTHONPATH", None)
env["UV_OFFLINE"] = "1"
env["PYTHONUTF8"] = "1"
report = {"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(), "relocated":str(output), "network_policy":"uv offline/no-index; browser separately denies all non-loopback requests", "steps":[]}
logs = archive.parent / (archive.stem + "-validation")
logs.mkdir(exist_ok=True)

def verify() -> int:
    result = subprocess.run([sys.executable, str(output / "Verify-Release.py")],env=env,capture_output=True,text=True)
    print(result.stdout)
    return result.returncode

assert verify() == 0
sample = output / "demo/artifacts/catalog.json"
original = sample.read_bytes()
sample.write_bytes(original+b"corruption")
try:
    assert verify() == 2
finally:
    sample.write_bytes(original)
assert verify() == 0
report["integrity_clean_corrupt_restored"] = [0,2,0]
for phase in ["first_offline_install_and_start", "cold_start_existing_environment"]:
    start = time.perf_counter()
    with (logs / (phase+".log")).open("w",encoding="utf-8") as log:
        process = subprocess.Popen(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(output / "Run-Demo.ps1"),"-Port","8768"],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            health = None
            while time.perf_counter()-start < 60:
                if process.poll() is not None:
                    raise RuntimeError(f"Launcher exited {process.returncode}; inspect {phase}.log")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8768/health",timeout=.5) as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(.05)
            if health is None:
                raise TimeoutError("Local service did not become healthy within60seconds.")
            elapsed=time.perf_counter()-start
            if phase == "cold_start_existing_environment":
                assert elapsed < 30
            report["steps"].append({"phase":phase,"seconds":elapsed,"health":health})
            with urllib.request.urlopen("http://127.0.0.1:8768/api/v1/experiments") as response:
                assert len(json.load(response))==25
        finally:
            # Only this test's owned process tree, never unrelated listeners.
            parent=psutil.Process(process.pid)
            children=parent.children(recursive=True)
            for child in reversed(children):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            if process.poll() is None: process.terminate()
            process.wait(timeout=10)
            psutil.wait_procs(children,timeout=10)
report["status"]="PASS_ENGINEERING_DIAGNOSTIC_ONLY"
(logs / "portable.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
