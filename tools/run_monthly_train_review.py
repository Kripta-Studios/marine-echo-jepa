"""Run only the four preregistered TRAIN review days, sequentially and without promotion."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from continuation_records import write_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    for date in ("2020-02-17", "2020-03-01", "2020-04-01", "2020-05-01"):
        log = ROOT / f"evidence/continuation/monthly-train-{date}.log"
        command = [
            sys.executable,
            str(ROOT / "tools/preprocess_train_candidate.py"),
            "--date",
            date,
            "--variant",
            "monthly-v1",
        ]
        started, peak, stopped = time.monotonic(), 0, False
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                try:
                    owner = psutil.Process(process.pid)
                    peak = max(
                        peak,
                        sum(p.memory_info().rss for p in [owner, *owner.children(recursive=True)]),
                    )
                    if peak > 22 * 1024**3 or time.monotonic() - started > 600:
                        stopped = True
                        for child in reversed(owner.children(recursive=True)):
                            child.terminate()
                        owner.terminate()
                        process.wait(timeout=15)
                except psutil.Error:
                    pass
                time.sleep(0.2)
        result = {
            "date": date,
            "command": command,
            "exit_code": process.returncode,
            "resource_stop": stopped,
            "elapsed_seconds": time.monotonic() - started,
            "peak_sampled_process_tree_rss_gib": peak / 1024**3,
            "log": str(log.relative_to(ROOT)),
        }
        results.append(result)
        write_report(
            ROOT / "evidence/continuation/monthly_train_execution.json",
            {
                "status": "TRAIN_REVIEW_ONLY_NOT_CORPUS_PROMOTION",
                "results": results,
                "benchmark_eligible": False,
            },
        )
        print(json.dumps(result), flush=True)
    return 0 if all(r["exit_code"] == 0 and not r["resource_stop"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
