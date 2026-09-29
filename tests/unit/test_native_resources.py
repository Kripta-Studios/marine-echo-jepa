"""GPU ownership distinguishes Windows desktop graphics from training."""

from __future__ import annotations

import pytest

from marine_echo.training.native_resources import classify_gpu_processes


def test_wddm_desktop_graphics_do_not_count_as_training():
    checked = classify_gpu_processes(
        [(12, "explorer.exe"), (13, "dwm.exe"), (14, "python.exe")],
        driver_model="WDDM",
        owner_pid=14,
    )
    assert checked["blocking_pids"] == []
    assert checked["desktop_graphics_pids"] == [12, 13]
    assert checked["hardware_exclusivity"] == "UNVERIFIABLE_WDDM"


@pytest.mark.parametrize("name", ["python.exe", "pythonw.exe", "ollama.exe", "unknown.exe", ""])
def test_other_runtime_or_unknown_wddm_process_blocks(name):
    checked = classify_gpu_processes([(12, name)], driver_model="WDDM", owner_pid=14)
    assert checked["blocking_pids"] == [12]


def test_non_wddm_compute_entries_block_regardless_of_desktop_name():
    checked = classify_gpu_processes([(12, "explorer.exe")], driver_model="TCC", owner_pid=14)
    assert checked["blocking_pids"] == [12]


def test_unrecognized_driver_and_malformed_pid_fail_closed():
    with pytest.raises(ValueError):
        classify_gpu_processes([(12, "explorer.exe")], driver_model="unknown", owner_pid=14)
    with pytest.raises(ValueError):
        classify_gpu_processes([(0, "explorer.exe")], driver_model="WDDM", owner_pid=14)
