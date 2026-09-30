"""Supervise an explicitly owned comparator process under RAM and full-time limits."""

from __future__ import annotations

import subprocess
import time

import psutil


def supervise_owned(
    child: subprocess.Popen,
    *,
    started: float,
    deadline_seconds: float,
    rss_limit_bytes: int,
    poll_seconds: float = 0.1,
) -> dict:
    process = psutil.Process(child.pid)
    owned = {process.pid: process}
    peaks = {}
    peak, reason = 0, None
    while child.poll() is None:
        try:
            # Windows venv python.exe is a redirector. Its descendant owns the
            # arrays/model; measuring only the launcher understates actual RAM.
            for descendant in process.children(recursive=True):
                owned.setdefault(descendant.pid, descendant)
            for pid, member in owned.items():
                try:
                    memory = member.memory_info()
                    peaks[pid] = max(peaks.get(pid, 0), memory.rss, getattr(memory, "peak_wset", 0))
                except psutil.NoSuchProcess:
                    continue
            peak = max(peak, sum(peaks.values()))
        except psutil.NoSuchProcess:
            break
        except psutil.AccessDenied:
            reason = "OWNED_PROCESS_MONITOR_DENIED"
        if peak >= rss_limit_bytes:
            reason = "PROCESS_RAM_LIMIT"
        elif time.monotonic() - started >= deadline_seconds:
            reason = "FULL_OWNED_ATTEMPT_DEADLINE"
        if reason:
            break
        time.sleep(poll_seconds)
    # Stop only captured descendants of this Popen handle. psutil Process objects
    # retain creation-time identity; never terminate unrelated/reused PIDs.
    survivors = [member for member in owned.values() if member.is_running()]
    if reason is None and child.poll() is not None and survivors:
        reason = "OWNED_DESCENDANTS_ALIVE_AFTER_LAUNCHER_EXIT"
    if reason:
        for member in reversed(survivors):
            try:
                member.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(survivors, timeout=5)
        for member in alive:
            try:
                member.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        if alive:
            raise RuntimeError("Owned descendants remain alive; do not retry this attempt.")
    exit_code = child.wait()
    elapsed = time.monotonic() - started
    if elapsed >= deadline_seconds and reason is None:
        reason = "FULL_OWNED_ATTEMPT_DEADLINE_AFTER_EXIT"
    return {
        "exit_code": exit_code,
        "stopped_for": reason,
        "peak_process_rss_bytes": peak,
        "elapsed_full_attempt_seconds": elapsed,
        "poll_seconds": poll_seconds,
        "rss_limit_bytes": rss_limit_bytes,
        "deadline_seconds": deadline_seconds,
        "owned_process_pids": sorted(owned),
        "peak_each_process_rss_bytes": {str(pid): value for pid, value in peaks.items()},
        "peak_definition": "Conservative sum of each owned process OS peak working set; sampled RSS fallback",
        "owned_tree_cleanup_verified": True,
    }
