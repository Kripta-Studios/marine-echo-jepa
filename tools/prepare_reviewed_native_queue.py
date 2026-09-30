"""Freeze already approved serial commands; perform no numerical data access."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "outputs/native_acoustic_ssl_v1/"
EVIDENCE = "evidence/ssl-research-v1/"


def main():
    baseline = EVIDENCE + "baseline-prefit-review-final-03.json"
    neural = EVIDENCE + "prefit-review-final-03.json"
    downstream = EVIDENCE + "downstream-prefit-review-final.json"
    for path, status in (
        (baseline, "APPROVED_PREFIT"),
        (neural, "APPROVED_PREFIT"),
        (downstream, "APPROVED_DOWNSTREAM_PREFIT"),
    ):
        if json.loads((ROOT / path).read_text(encoding="utf-8"))["status"] != status:
            raise ValueError("Queue requires actual distinct approvals.")
    jobs = []

    def add(name, entrypoint, args, method, report="run.json", mode=None):
        jobs.append(
            {
                "id": name,
                "entrypoint": "tools/" + entrypoint,
                "args": args,
                "method": method,
                "mode": mode,
                "report": BASE + name + "/" + report,
            }
        )

    name = "lightgbm_seed7_h96_text"
    add(
        name,
        "execute_bounded_native_comparator.py",
        [
            "--method",
            "lightgbm",
            "--review",
            baseline,
            "--output",
            BASE + name,
            "--receipt",
            EVIDENCE + "lightgbm-attempt-02",
            "--history",
            "96",
        ],
        "lightgbm",
        report="result.json",
    )
    for method, mode in (
        ("shared_ssl", "frozen_readout"),
        ("shared_ssl", "full_finetune"),
        ("direct", "direct_end_to_end"),
    ):
        name = method + "_" + mode + "_seed7_h96"
        config = "configs/native_downstream_" + name + "_v1.json"
        args = [
            "--config",
            config,
            "--review",
            downstream,
            "--output",
            BASE + name,
            "--receipt",
            EVIDENCE + name + "-attempt-01",
        ]
        if method == "shared_ssl":
            args += [
                "--encoder",
                BASE + "shared_ssl_seed7_h96_cuda0/selected_encoder.pt",
                "--ancestor-review",
                EVIDENCE + "prefit-review-final-02.json",
                "--ancestor-config",
                "configs/native_ssl_shared_ssl_seed7_h96_v1.json",
            ]
        add(name, "execute_native_downstream_job.py", args, method, mode=mode)
    for method in ("masked_ssl", "permuted_ssl", "direct", "random_frozen"):
        name = method + "_seed7_h96_reviewed"
        add(
            name,
            "execute_native_ssl_job.py",
            [
                "--config",
                "configs/native_ssl_" + method + "_seed7_h96_v1.json",
                "--review",
                neural,
                "--output",
                BASE + name,
            ],
            method,
        )
    name = "chronos2_h96_local"
    snapshot = (
        "C:/Users/Álvaro Schwiedop/.cache/huggingface/hub/models--amazon--chronos-2/"
        "snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
    )
    add(
        name,
        "execute_bounded_native_comparator.py",
        [
            "--method",
            "chronos2",
            "--review",
            baseline,
            "--output",
            BASE + name,
            "--receipt",
            EVIDENCE + "chronos2-attempt-01",
            "--history",
            "96",
            "--snapshot",
            snapshot,
        ],
        "chronos2",
        report="result.json",
    )
    path = ROOT / "orchestration/native_ssl_approved_queue_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "kind": "reviewed_native_serial_queue_v1",
                "jobs": jobs,
                "authorization": "Owner local mission plus exact existing distinct reviews",
                "execution": "Required approved wrappers only; serial; stop on first failure",
                "final_test_access": "NOT_AUTHORIZED",
                "app_release_work": "FROZEN",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "jobs": len(jobs), "path": str(path)}))


if __name__ == "__main__":
    main()
