"""Exact owner-authorized VLC desktop exception; retain unknown-runtime guards."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil

from marine_echo.training import native_resources as original

ROOT = Path(__file__).resolve().parents[3]
OWNER_PATH = ROOT / "orchestration/native_vlc_desktop_owner_resolution_v1.json"
OWNER_SHA256 = "18e29745c7595a9ea50bccf23e3124b06c57106929ac20c40a8f04776ff598e4"


def validate_owner_resolution(resolution):
    required = {
        "kind": "native_desktop_process_owner_resolution_v1",
        "status": "OWNER_AUTHORIZED_OPERATIONAL_EXCEPTION_PREFIT_REQUIRED",
        "allowed_additional_desktop_executables": ["vlc.exe"],
        "allowed_driver_model": "WDDM", "single_scientific_training_process_tree": True,
        "unknown_or_other_ml_runtimes_blocked": True,
        "gpu_allocated_reserved_bytes_strictly_less_than": 10 * 1024**3,
        "owned_tree_ram_bytes_strictly_less_than": 22 * 1024**3,
        "band_full_owned_gpu_hours": 12, "aggregate_full_owned_gpu_hours": 96,
        "hardware_exclusivity": "UNVERIFIABLE_WDDM",
        "foreign_process_termination_authorized": False,
        "historical_fitted_source_changes_authorized": False,
        "independent_prefit": "REQUIRED", "final_numeric_access_or_paid_resources_authorized": False,
    }
    if not isinstance(resolution, dict):
        raise TypeError("Explicit owner resolution required")
    for key, value in required.items():
        actual = resolution.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Owner desktop exception cannot change {key}")


def classify_gpu_processes(processes, *, driver_model, owner_pid, owner_resolution):
    validate_owner_resolution(owner_resolution)
    result = original.classify_gpu_processes(processes, driver_model=driver_model, owner_pid=owner_pid)
    allowed = [pid for pid, name in processes if pid != owner_pid and str(name).casefold() == "vlc.exe"] if driver_model == "WDDM" else []
    result["blocking_pids"] = [pid for pid in result["blocking_pids"] if pid not in allowed]
    result["desktop_graphics_pids"].extend(allowed)
    result.update(owner_authorized_additional_desktop_pids=allowed,
                  policy="exclusive_local_training_lock_plus_owner_authorized_desktop_exception_v2",
                  owner_resolution_sha256=OWNER_SHA256,
                  additional_desktop_gpu_consumption="UNMEASURED",
                  scientific_process_exclusivity="MANDATORY_SINGLE_OWNED_TRAINING_TREE")
    return result


def inspect_gpu_ownership(output):
    raw = OWNER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA256:
        raise ValueError("Exact bound owner desktop exception changed")
    resolution = json.loads(raw)
    validate_owner_resolution(resolution)
    response = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True)
    gpus = ET.fromstring(response.stdout).findall("gpu")
    if len(gpus) != 1:
        raise RuntimeError("Exactly one visible NVIDIA GPU required")
    gpu, processes = gpus[0], []
    for item in gpu.findall("processes/process_info"):
        pid = int(item.findtext("pid", "0"))
        try:
            name = psutil.Process(pid).name()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            name = ""
        processes.append((pid, name))
    checked = classify_gpu_processes(processes, driver_model=gpu.findtext("driver_model/current_dm", "N/A"),
                                    owner_pid=os.getpid(), owner_resolution=resolution)
    checked["observed_processes"] = [{"pid": pid, "executable": name} for pid, name in processes]
    (Path(output) / "gpu-ownership.json").write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")
    if checked["blocking_pids"]:
        raise RuntimeError("Another GPU runtime or unknown process is active; serialize execution.")
    return checked
