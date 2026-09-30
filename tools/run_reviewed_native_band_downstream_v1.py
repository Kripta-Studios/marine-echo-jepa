"""Execute six unchanged approved downstream commands, one scientific tree at a time."""

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
    extraction_path = ROOT / "evidence/ssl-research-v1/band-downstream-approval-extraction-v1.json"
    extraction = json.loads(extraction_path.read_bytes())
    if (
        extraction.get("status") != "SIX_EXACT_BAND_DOWNSTREAM_APPROVALS_VERIFIED_AND_EXTRACTED"
        or extraction.get("scientific_fields_unchanged") is not True
        or len(extraction.get("derived", {})) != 6
    ):
        raise ValueError("Six unchanged exact approvals required")
    if (
        digest(ROOT / "evidence/ssl-research-v1/band-downstream-prefit-review-final-v1.json")
        != extraction["parent_review_sha256"]
    ):
        raise ValueError("Combined review changed after extraction")
    commands = []
    for key, record in extraction["derived"].items():
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Exact extracted review changed")
        review = json.loads(path.read_bytes())
        runtime = review["runtime_arguments"]
        method, mode = key.split("/")
        if (
            review["status"] != "APPROVED_DOWNSTREAM_PREFIT"
            or review["approved_config"]["method"] != method
            or review["approved_config"]["mode"] != mode
            or runtime["review"] != str(path)
            or runtime["trainer-review"] != str(path)
            or runtime["kind"] != "downstream"
            or runtime.get("resume") is not None
        ):
            raise ValueError("Exact fresh downstream command required")
        for name, expected in review["bindings"].items():
            if digest(name) != expected:
                raise ValueError(f"Changed reviewed binding: {name}")
        output, receipt = Path(runtime["output"]), Path(runtime["receipt"])
        if output.exists() or receipt.exists():
            raise FileExistsError("Preserve earlier outputs/attempts")
        flags = (
            "kind",
            "config",
            "review",
            "output",
            "receipt",
            "encoder",
            "ancestor-review",
            "ancestor-config",
        )
        arguments = [item for flag in flags for item in ("--" + flag, runtime[flag])]
        commands.append(
            (
                key,
                runtime,
                [
                    str(ROOT / ".venv/Scripts/python.exe"),
                    "-u",
                    "tools/execute_native_band_job.py",
                    *arguments,
                ],
            )
        )
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_bytes())
    if (
        any(str(r.get("status", "")).startswith("RUNNING_") for r in ledger["runs"])
        or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
    ):
        raise ValueError("Actual prior scientific completion required")
    folder = ROOT / "evidence/ssl-research-v1/band-downstream-serial-attempt-01"
    folder.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "extraction_sha256": digest(extraction_path), "jobs": []}
    destination = folder / "queue.json"
    for key, runtime, command in commands:
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "LAUNCH", "id": key, "command": command}), flush=True)
        with (folder / (key.replace("/", "-") + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        report_path = Path(runtime["output"]) / "run.json"
        attempt_path = Path(runtime["receipt"]) / "attempt.json"
        valid = process.returncode == 0 and report_path.exists() and attempt_path.exists()
        entry = {"id": key, "actual_exit_code": process.returncode}
        if valid:
            report = json.loads(report_path.read_bytes())
            attempt = json.loads(attempt_path.read_bytes())
            resources = attempt["resources_full_attempt"]
            valid = (
                report.get("status") == "COMPLETED"
                and report.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
                and report.get("architecture") == "nonlinear_frequency_conditioned_v1"
                and report["config"]["seed"] == 7
                and key == report["config"]["method"] + "/" + report["mode"]
                and resources.get("exit_code") == 0
                and resources.get("stopped_for") is None
                and resources.get("owned_tree_cleanup_verified") is True
                and attempt.get("budget_family") == "native_band_v1"
            )
            entry.update(
                run_sha256=digest(report_path),
                attempt_sha256=digest(attempt_path),
                development_pinball_db=report.get("selected_daily_dev_pinball"),
            )
        entry["status"] = "VERIFIED_COMPLETED" if valid else "FAILED_QUEUE_STOPPED"
        result["jobs"].append(entry)
        result["status"] = "RUNNING" if valid else "STOPPED_AFTER_FAILURE"
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if not valid:
            return 1
    result["status"] = "COMPLETED_SIX_REVIEWED_BAND_DOWNSTREAM_JOBS"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
