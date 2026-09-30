"""Serialize three previously approved original-guard downstream jobs after closure."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bindings(document):
    for group in ("bindings", "proof_bindings"):
        for path, expected in document.get(group, {}).items():
            if digest(path) != expected:
                raise ValueError(f"Previously reviewed bytes changed: {path}")


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    prior_path = folder / "band-fixed-replication-remaining-queue-attempt-01/queue.json"
    prior = json.loads(prior_path.read_bytes())
    if prior.get("status") != "COMPLETED_TWO_ORIGINAL_SEED23_REPLICATIONS" or len(prior.get("jobs", [])) != 2:
        raise ValueError("Wait for successful closure of both original seed23 jobs")
    if any(job.get("status") != "VERIFIED_COMPLETED" or job.get("actual_exit_code") != 0 for job in prior["jobs"]):
        raise ValueError("Actual successful original queue exits required")
    extraction_path = folder / "band-seed13-strong-review-extraction-v3.json"
    extraction = json.loads(extraction_path.read_bytes())
    parent_path = Path(extraction["parent_review_path"])
    if digest(parent_path) != extraction["parent_review_sha256"] or extraction["actual_exit_witness"]["actual_cli_exit_code"] != 0:
        raise ValueError("Actual distinct seed13 completed-parent approval required")
    parent_review = json.loads(parent_path.read_bytes())
    if parent_review.get("status") != "APPROVED_BAND_COMPLETED_PARENT_REFERENCES" or parent_review.get("unresolved_defects") != []:
        raise ValueError("Distinct parent approval must contain no unresolved defects")
    verify_bindings(parent_review)
    records = extraction["derived"]
    expected_ids = ["band_shared_ssl_frozen_readout_seed13_h96_replication_v3", "band_shared_ssl_full_finetune_seed13_h96_replication_v3"]
    if [record["id"] for record in records] != expected_ids:
        raise ValueError("Exactly the two previously approved seed13 endpoints required")
    jobs = []
    for record in records:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("Exact original materialized approval required")
        jobs.append((record["id"], path, "tools/execute_native_band_replication_job.py"))
    retry_proposal = ROOT / "orchestration/native_band_random_strong_retry_admission_v1.json"
    proposal = json.loads(retry_proposal.read_bytes())
    retry_path = Path(proposal["approval"]["runtime_arguments"]["review"])
    retry_receipt = json.loads((folder / "band-random-strong-retry-review-closeout-v1.json").read_bytes())
    if (retry_receipt.get("status") != "DISTINCT_EXACT_RANDOM_RETRY_REVIEW_VERIFIED_NOT_LAUNCHED"
            or retry_receipt["review_sha256"] != digest(retry_path)
            or retry_receipt["proposal_sha256"] != digest(retry_proposal)):
        raise ValueError("Previously closed original random retry review required")
    jobs.append(("band_random_frozen_frozen_readout_seed7_h96_native_retry01", retry_path, "tools/execute_native_band_job.py"))
    commands = []
    for identifier, path, entrypoint in jobs:
        review = json.loads(path.read_bytes())
        if (review.get("status") != "APPROVED_DOWNSTREAM_PREFIT"
                or review.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
                or review.get("unresolved_defects", []) != []):
            raise ValueError("Original distinct full approval leaf required")
        verify_bindings(review)
        runtime = review["runtime_arguments"]
        if runtime.get("resume") is not None or runtime["review"] != str(path) or runtime["trainer-review"] != str(path):
            raise ValueError("Fresh exact original review authority required")
        if any(Path(runtime[key]).exists() for key in ("output", "receipt")):
            raise FileExistsError("Preserve every earlier output and receipt")
        args = [part for key in ("kind", "config", "review", "output", "receipt", "encoder", "ancestor-review", "ancestor-config")
                if runtime.get(key) is not None for part in ("--" + key, runtime[key])]
        commands.append((identifier, runtime, [str(ROOT / ".venv/Scripts/python.exe"), "-u", entrypoint, *args]))
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    if (any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Scientific ownership and every journal must be closed")
    destination = folder / "band-remaining-original-controls-queue-attempt-01"
    destination.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "prior_queue_sha256": digest(prior_path), "jobs": []}
    for identifier, runtime, command in commands:
        (destination / "queue.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (destination / (identifier + ".log")).open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        report_path = Path(runtime["output"]) / "run.json"
        attempt_path = Path(runtime["receipt"]) / "attempt.json"
        valid = completed.returncode == 0 and report_path.exists() and attempt_path.exists()
        entry = {"id": identifier, "actual_exit_code": completed.returncode}
        if valid:
            report, attempt = (json.loads(path.read_bytes()) for path in (report_path, attempt_path))
            resources = attempt["resources_full_attempt"]
            valid = (report.get("status") == "COMPLETED" and report.get("evidence_kind") == "REAL_TRAIN_DEVELOPMENT_FIT"
                     and resources.get("exit_code") == 0 and resources.get("stopped_for") is None
                     and resources.get("owned_tree_cleanup_verified") is True and attempt.get("budget_family") == "native_band_v1")
            entry.update(run_sha256=digest(report_path), attempt_sha256=digest(attempt_path),
                         development_pinball_db=report.get("selected_daily_dev_pinball"))
        entry["status"] = "VERIFIED_COMPLETED" if valid else "FAILED_QUEUE_STOPPED"
        result["jobs"].append(entry)
        result["status"] = "RUNNING" if valid else "STOPPED_AFTER_FAILURE"
        (destination / "queue.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if not valid:
            return 1
    result["status"] = "COMPLETED_THREE_PREVIOUSLY_APPROVED_ORIGINAL_DOWNSTREAM_JOBS"
    (destination / "queue.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
