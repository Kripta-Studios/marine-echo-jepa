"""Run exactly four independently approved fixed replications, one owned job at a time."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    extraction_path = ROOT / "evidence/ssl-research-v1/band-fixed-replication-review-extraction-v3.json"
    extraction = json.loads(extraction_path.read_bytes())
    parent = ROOT / "evidence/ssl-research-v1/band-fixed-replication-prefit-compact-final-v3.json"
    if (extraction.get("status") != "FOUR_EXACT_FIXED_REPLICATION_APPROVALS_MATERIALIZED"
            or extraction.get("scientific_fields_unchanged") is not True
            or extraction.get("parent_review_path") != str(parent)
            or extraction.get("parent_review_sha256") != digest(parent)
            or len(extraction.get("derived", [])) != 4):
        raise ValueError("Four exact independently approved records required")
    commands = []
    pairs = []
    for record in extraction["derived"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Extracted review changed")
        review = json.loads(path.read_bytes())
        runtime, config = review["runtime_arguments"], review["approved_config"]
        kind = runtime["kind"]
        if (kind not in ("ssl", "downstream")
                or review["status"] != ("APPROVED_PREFIT" if kind == "ssl" else "APPROVED_DOWNSTREAM_PREFIT")
                or runtime["review"] != str(path) or runtime["trainer-review"] != str(path)
                or any(runtime.get(key) is not None for key in ("resume", "encoder", "ancestor-review", "ancestor-config"))):
            raise ValueError("Exact fresh replication command required")
        pairs.append((config["method"], config["seed"]))
        for name, sha in review["bindings"].items():
            if digest(name) != sha:
                raise ValueError(f"Reviewed dependency changed: {name}")
        if Path(runtime["output"]).exists() or Path(runtime["receipt"]).exists():
            raise FileExistsError("Preserve existing outputs and attempts")
        args = [item for flag in ("kind", "config", "review", "output", "receipt")
                for item in ("--" + flag, runtime[flag])]
        commands.append((record["id"], runtime, [str(ROOT / ".venv/Scripts/python.exe"), "-u",
                                               "tools/execute_native_band_replication_job.py", *args]))
    if pairs != [("shared_ssl", 13), ("direct", 13), ("shared_ssl", 23), ("direct", 23)]:
        raise ValueError("Exact four fixed replications in proposal order required")
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    if (any(str(run.get("status", "")).startswith("RUNNING_") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()):
        raise ValueError("Previous scientific ownership must be closed")
    folder = ROOT / "evidence/ssl-research-v1/band-fixed-replication-queue-attempt-01"
    folder.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "extraction_sha256": digest(extraction_path), "jobs": []}
    destination = folder / "queue.json"
    for identifier, runtime, command in commands:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "LAUNCH", "id": identifier, "command": command}), flush=True)
        with (folder / (identifier + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        report_path = Path(runtime["output"]) / "run.json"
        attempt_path = Path(runtime["receipt"]) / "attempt.json"
        valid = process.returncode == 0 and report_path.exists() and attempt_path.exists()
        entry = {"id": identifier, "actual_exit_code": process.returncode}
        if valid:
            report, attempt = (json.loads(path.read_bytes()) for path in (report_path, attempt_path))
            resources = attempt["resources_full_attempt"]
            valid = (report.get("status") == "COMPLETED"
                     and report.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
                     and report.get("architecture") == "nonlinear_frequency_conditioned_v1"
                     and resources.get("exit_code") == 0 and resources.get("stopped_for") is None
                     and resources.get("owned_tree_cleanup_verified") is True
                     and attempt.get("budget_family") == "native_band_v1")
            entry.update(run_sha256=digest(report_path), attempt_sha256=digest(attempt_path),
                         development_pinball_db=report.get("selected_daily_dev_pinball"))
        entry["status"] = "VERIFIED_COMPLETED" if valid else "FAILED_QUEUE_STOPPED"
        result["jobs"].append(entry)
        result["status"] = "RUNNING" if valid else "STOPPED_AFTER_FAILURE"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if not valid:
            return 1
    result["status"] = "COMPLETED_FOUR_REVIEWED_FIXED_BAND_REPLICATIONS"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
