"""Execute independently approved real train/development native materialization."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from marine_echo.data.native_ssl_corpus import materialize, sha256


def main() -> None:
    review = ROOT / "evidence/ssl-research-v1/numeric-review-final-03.json"
    split = ROOT / "configs/native_ssl_split_v1.json"
    raw_review = json.loads(review.read_text(encoding="utf-8"))
    # Check all exact reviewed files, including the independent adversarial tests.
    for path, expected in raw_review["bindings"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Reviewed materialization input changed: {path}")
    output = ROOT / "data/processed/native_ssl_v1"
    for role in ("train", "development"):
        if (output / f"{role}.npz").exists():
            raise ValueError("Preserve existing corpus; explicitly version a new attempt.")
    started = time.perf_counter()
    running, peak = threading.Event(), [0]
    running.set()

    def monitor() -> None:
        while running.is_set():
            rss = psutil.Process().memory_info().rss
            peak[0] = max(peak[0], rss)
            if rss >= 22 * 1024**3:
                # No extra data/source read is scheduled once the parent sees this.
                running.clear()
                print("RESOURCE_LIMIT_EXCEEDED", flush=True)
                return
            time.sleep(0.25)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    results = []
    try:
        for role in ("train", "development"):
            if not running.is_set():
                raise MemoryError("Native materialization RSS ceiling exceeded.")
            report = materialize(split, review, role, output / f"{role}.npz")
            results.append(report)
            print(
                json.dumps(
                    {
                        "role": role,
                        "issued": report["issued"],
                        "ssl_eligible": report["ssl_eligible"],
                        "npz_sha256": report["npz_sha256"],
                    }
                ),
                flush=True,
            )
    finally:
        running.clear()
        watcher.join(timeout=2)
    evidence = {
        "status": "COMPLETED_REAL_TRAIN_DEVELOPMENT_MATERIALIZATION",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_gib": peak[0] / 1024**3,
        "roles": results,
        "final_test_numeric_access": "NOT_RUN_PROHIBITED",
        "review_sha256": sha256(review),
    }
    (ROOT / "evidence/ssl-research-v1/materialization.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "elapsed_seconds": evidence["elapsed_seconds"],
                "peak_rss_gib": evidence["peak_rss_gib"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
