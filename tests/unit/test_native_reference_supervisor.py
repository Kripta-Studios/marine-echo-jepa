"""Real owned child termination, without fitting or GPU use."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import psutil

from tools.native_reference_supervisor import supervise_owned


def test_owned_sleeping_process_is_stopped_by_full_attempt_deadline():
    started = time.monotonic()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    report = supervise_owned(
        child, started=started, deadline_seconds=0.4, rss_limit_bytes=22 * 2**30, poll_seconds=0.02
    )
    assert report["stopped_for"] == "FULL_OWNED_ATTEMPT_DEADLINE"
    assert report["exit_code"] != 0
    assert report["elapsed_full_attempt_seconds"] < 5


def test_owned_process_is_stopped_for_ram_without_killing_another_child():
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        report = supervise_owned(
            child,
            started=time.monotonic(),
            deadline_seconds=10,
            rss_limit_bytes=1,
            poll_seconds=0.02,
        )
        assert report["stopped_for"] == "PROCESS_RAM_LIMIT"
        assert report["exit_code"] != 0
        assert unrelated.poll() is None
    finally:
        if unrelated.poll() is None:
            unrelated.terminate()
        unrelated.wait()


def test_completed_process_keeps_actual_exit_status():
    child = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    report = supervise_owned(
        child,
        started=time.monotonic(),
        deadline_seconds=10,
        rss_limit_bytes=22 * 2**30,
        poll_seconds=0.02,
    )
    assert report["exit_code"] == 7 and report["stopped_for"] is None


def test_redirector_descendant_memory_is_counted_and_owned_tree_stopped(tmp_path):
    marker = tmp_path / "owned-descendant.json"
    script = (
        "import subprocess,sys,time,json; "
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import time; allocation=bytearray(100*2**20); time.sleep(30)']); "
        f"open({str(marker)!r},'w').write(json.dumps({{'pid':p.pid}})); p.wait()"
    )
    child = subprocess.Popen([sys.executable, "-c", script])
    descendants = []
    try:
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            time.sleep(0.02)
        assert marker.exists()
        descendants = psutil.Process(child.pid).children(recursive=True)
        report = supervise_owned(
            child,
            started=time.monotonic(),
            deadline_seconds=3,
            rss_limit_bytes=64 * 2**20,
            poll_seconds=0.02,
        )
        assert report["stopped_for"] == "PROCESS_RAM_LIMIT"
        assert report["peak_process_rss_bytes"] >= 64 * 2**20
        grandchild_pid = json.loads(marker.read_text())["pid"]
        assert not psutil.pid_exists(grandchild_pid)
    finally:
        # Only processes created by this explicit test tree, including on red.
        for process in reversed(descendants):
            if process.is_running():
                process.terminate()
                process.wait(timeout=5)
        if child.poll() is None:
            child.terminate()
        child.wait()


def test_successful_interpreter_child_has_real_memory_and_clean_exit():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; allocation=bytearray(100*2**20); time.sleep(0.2)"]
    )
    report = supervise_owned(
        child,
        started=time.monotonic(),
        deadline_seconds=10,
        rss_limit_bytes=22 * 2**30,
        poll_seconds=0.01,
    )
    assert report["exit_code"] == 0 and report["stopped_for"] is None
    assert report["peak_process_rss_bytes"] >= 100 * 2**20
    assert report["owned_tree_cleanup_verified"]
