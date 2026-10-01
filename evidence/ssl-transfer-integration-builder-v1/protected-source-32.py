"""Conservative desktop-aware ownership checks; not hardware exclusivity proof.

NVIDIA WDDM reports N/A per-process memory and this installed driver also
reports desktop GUI applications in query-compute-apps. A fixed explicit list
admits familiar desktop graphics only. Unknown executables and other Python/ML
runtimes block; the exclusive local training lock remains mandatory.
Source: https://docs.nvidia.com/deploy/nvidia-smi/
"""

from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil

DESKTOP_GRAPHICS_NAMES = frozenset(
    {
        "dwm.exe",
        "logonui.exe",
        "explorer.exe",
        "windowsterminal.exe",
        "textinputhost.exe",
        "startmenuexperiencehost.exe",
        "searchhost.exe",
        "powertoys.exe",
        "powertoys.quickaccess.exe",
        "amdrssrcext.exe",
        "firefox.exe",
        "msedgewebview2.exe",
        "ms-teams.exe",
        "jabra-direct.exe",
        "chatgpt classic.exe",
        "microsoft.cmdpal.ui.exe",
        "docker desktop.exe",
        "radeonsoftware.exe",
        "lockapp.exe",
        "shellexperiencehost.exe",
        "applicationframehost.exe",
        "systemsettings.exe",
        "code.exe",
        "notion calendar.exe",
        "discord.exe",
        "notepad.exe",
        "shellhost.exe",
        "powertoys.colorpickerui.exe",
        "powertoys.powerlauncher.exe",
        "powertoys.fancyzones.exe",
    }
)


def classify_gpu_processes(processes, *, driver_model, owner_pid):
    if driver_model not in ("WDDM", "TCC", "N/A") or owner_pid <= 0:
        raise ValueError("Unknown driver model or invalid owner PID.")
    blocked, desktop = [], []
    for pid, name in processes:
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("Invalid GPU process identity.")
        if pid == owner_pid:
            continue
        if driver_model == "WDDM" and str(name).casefold() in DESKTOP_GRAPHICS_NAMES:
            desktop.append(pid)
        else:
            blocked.append(pid)
    return {
        "driver_model": driver_model,
        "blocking_pids": blocked,
        "desktop_graphics_pids": desktop,
        "hardware_exclusivity": "UNVERIFIABLE_WDDM"
        if driver_model == "WDDM"
        else "PROCESS_QUERY_ONLY",
        "policy": "exclusive_local_training_lock_plus_gpu_process_runtime_check",
        "desktop_allowlist": sorted(DESKTOP_GRAPHICS_NAMES),
    }


def inspect_gpu_ownership(output):
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True)
    root = ET.fromstring(result.stdout)
    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise RuntimeError("This local research contract requires exactly one visible NVIDIA GPU.")
    gpu = gpus[0]
    driver = gpu.findtext("driver_model/current_dm", "N/A")
    processes = []
    for item in gpu.findall("processes/process_info"):
        pid = int(item.findtext("pid", "0"))
        # Process executable name only; never collect command lines or credentials.
        try:
            name = psutil.Process(pid).name()
        except psutil.NoSuchProcess:
            continue  # GPU entry exited between snapshots.
        except psutil.AccessDenied:
            name = ""  # Unknown process blocks rather than gaining new permissions.
        processes.append((pid, name))
    checked = classify_gpu_processes(processes, driver_model=driver, owner_pid=os.getpid())
    checked["observed_processes"] = [{"pid": pid, "executable": name} for pid, name in processes]
    path = Path(output) / "gpu-ownership.json"
    path.write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    if checked["blocking_pids"]:
        raise RuntimeError("Another GPU runtime or unknown process is active; serialize execution.")
    return checked
