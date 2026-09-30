"""Freeze only the five exact independently approved band screen commands."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("shared_ssl", "masked_ssl", "permuted_ssl", "random_frozen", "direct")


def main():
    extraction = ROOT / "evidence/ssl-research-v1/band-screen-approval-extraction-v1.json"
    receipt = json.loads(extraction.read_text(encoding="utf-8"))
    if receipt.get("status") != "FIVE_EXACT_BAND_SCREEN_APPROVALS_VERIFIED_AND_EXTRACTED":
        raise ValueError("Actual complete per-job approvals required")
    jobs = []
    for method in ORDER:
        record = receipt["derived"][method]
        review_path = Path(record["path"])
        if hashlib.sha256(review_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Derived approved bytes changed")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        runtime = review["runtime_arguments"]
        if review["approved_config"]["method"] != method or runtime["review"] != str(review_path):
            raise ValueError("Exact method/runtime review differs")
        kind = "downstream" if method == "direct" else "ssl"
        if runtime["kind"] != kind or any(
            runtime.get(k) is not None
            for k in ("encoder", "ancestor-review", "ancestor-config", "resume")
        ):
            raise ValueError("Only initial scratch/fresh screen arguments required")
        output = Path(runtime["output"])
        attempt = Path(runtime["receipt"])
        if not output.is_relative_to(ROOT / "outputs") or not attempt.is_relative_to(
            ROOT / "evidence"
        ):
            raise ValueError("Only approved root-owned model output/receipt paths")
        args = []
        for flag in ("kind", "config", "review", "output", "receipt"):
            args += ["--" + flag, runtime[flag]]
        jobs.append(
            {
                "id": output.name,
                "method": method,
                "kind": kind,
                "entrypoint": "tools/execute_native_band_job.py",
                "args": args,
                "review_sha256": record["sha256"],
                "report": str(output / "run.json"),
            }
        )
    queue = {
        "kind": "reviewed_native_band_serial_queue_v1",
        "jobs": jobs,
        "recipe_count_including_original": 11,
        "band_full_owned_gpu_limit_hours": 12,
        "aggregate_full_owned_gpu_limit_hours": 96,
        "approval_extraction_sha256": hashlib.sha256(extraction.read_bytes()).hexdigest(),
        "app_release_work": "FROZEN",
        "final_numeric_access": "NOT_AUTHORIZED",
    }
    path = ROOT / "orchestration/native_band_seed7_approved_screen_queue_v1.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(queue, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "FIVE_REVIEWED_COMMANDS_FROZEN_NOT_EXECUTED", "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
