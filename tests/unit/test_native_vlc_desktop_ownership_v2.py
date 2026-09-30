"""Owner-authorized desktop coexistence never permits another ML runtime."""

import copy
import json
from pathlib import Path

import pytest

from marine_echo.training import native_resources as original

try:
    from marine_echo.training import native_desktop_resources_v2 as candidate
except ImportError:
    candidate = None

ROOT = Path(__file__).resolve().parents[2]


def classify(processes, driver_model="WDDM", resolution=None):
    resolution = resolution or json.loads((ROOT / "orchestration/native_vlc_desktop_owner_resolution_v1.json").read_bytes())
    if candidate is None:
        return original.classify_gpu_processes(processes, driver_model=driver_model, owner_pid=900)
    return candidate.classify_gpu_processes(processes, driver_model=driver_model, owner_pid=900,
                                           owner_resolution=resolution)


def test_explicit_owner_vlc_exception_preserves_other_runtime_blocks():
    result = classify([(900, "python.exe"), (1, "vlc.exe"), (2, "python.exe"), (3, "unknown.exe"), (4, "dwm.exe")])
    assert result["blocking_pids"] == [2, 3]
    assert result["desktop_graphics_pids"] == [4, 1]
    assert result["hardware_exclusivity"] == "UNVERIFIABLE_WDDM"


@pytest.mark.parametrize("driver", ["TCC", "N/A"])
def test_vlc_exception_is_wddm_only(driver):
    assert classify([(1, "vlc.exe")], driver)["blocking_pids"] == [1]


@pytest.mark.parametrize("field,value", [
    ("allowed_additional_desktop_executables", ["vlc.exe", "python.exe"]),
    ("allowed_driver_model", "TCC"), ("single_scientific_training_process_tree", False),
    ("unknown_or_other_ml_runtimes_blocked", False), ("foreign_process_termination_authorized", True),
    ("historical_fitted_source_changes_authorized", True), ("independent_prefit", "NOT_REQUIRED"),
    ("band_full_owned_gpu_hours", 13), ("aggregate_full_owned_gpu_hours", 97),
    ("final_numeric_access_or_paid_resources_authorized", True),
    ("gpu_allocated_reserved_bytes_strictly_less_than", True),
])
def test_tampered_authority_cannot_relax_other_limits(field, value):
    resolution = json.loads((ROOT / "orchestration/native_vlc_desktop_owner_resolution_v1.json").read_bytes())
    resolution = copy.deepcopy(resolution)
    resolution[field] = value
    with pytest.raises(ValueError):
        classify([(1, "vlc.exe")], resolution=resolution)


def test_original_classifier_stays_unchanged_and_blocks_vlc():
    assert original.classify_gpu_processes([(1, "vlc.exe")], driver_model="WDDM", owner_pid=900)["blocking_pids"] == [1]
