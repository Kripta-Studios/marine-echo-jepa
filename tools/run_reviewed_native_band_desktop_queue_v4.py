"""Run six exact reviewed recipes serially under the owner-authorized desktop policy."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_idle():
    path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(path.read_bytes())
    if (any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Scientific ownership and all journals must be closed")


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    proposal_path = ROOT / "orchestration/native_band_desktop_admission_v4.json"
    proposal = json.loads(proposal_path.read_bytes())
    extraction = json.loads((folder / "band-desktop-operational-review-extraction-v4.json").read_bytes())
    parent_path = Path(extraction["parent_review_path"])
    if (extraction.get("status") != "SIX_EXACT_DESKTOP_OPERATIONAL_APPROVALS_MATERIALIZED"
            or extraction.get("parent_review_sha256") != digest(parent_path)):
        raise ValueError("Actual closed exact operational prefit required")
    parent = json.loads(parent_path.read_bytes())
    if (parent.get("status") != "APPROVED_BAND_DESKTOP_OPERATIONAL_REFERENCES"
            or parent.get("proposal_sha256") != digest(proposal_path)
            or parent.get("reviewer_session_id") != "01a0ef27-876b-7692-917e-3975afc6893d"
            or parent.get("unresolved_defects") != []
            or parent.get("proof_bindings", {}).get(str(Path(__file__).resolve())) != digest(__file__)):
        raise ValueError("Exact distinct review and controller binding required")
    for group in ("bindings", "proof_bindings"):
        for name, expected in parent[group].items():
            if digest(name) != expected:
                raise ValueError(f"Reviewed bytes changed: {name}")
    records = extraction["derived"]
    if len(records) != 6 or [entry["id"] for entry in records] != [job["id"] for job in proposal["jobs"]]:
        raise ValueError("Exactly the original six ordered recipes required")
    commands = []
    for record, job in zip(records, proposal["jobs"], strict=True):
        path = Path(record["path"])
        if record["sha256"] != digest(path) or str(path) != job["review_path"]:
            raise ValueError("Exact extracted leaf approval required")
        review = json.loads(path.read_bytes())
        runtime = review["runtime_arguments"]
        if (runtime != job["approval"]["runtime_arguments"]
                or review["approved_config"] != job["approval"]["approved_config"]
                or runtime["review"] != str(path) or runtime["trainer-review"] != str(path)
                or any(Path(runtime[key]).exists() for key in ("output", "receipt"))):
            raise ValueError("Preserve exact scientific fields and every prior attempt")
        args = [item for flag in ("kind", "config", "review", "output", "receipt") for item in ("--" + flag, runtime[flag])]
        for flag in ("encoder", "ancestor-review", "ancestor-config", "resume"):
            if runtime.get(flag) is not None:
                args.extend(("--" + flag, runtime[flag]))
        commands.append((record["id"], runtime, [str(ROOT / ".venv/Scripts/python.exe"), "-u", "tools/execute_native_band_operational_job_v4.py", *args]))
    check_idle()
    destination = folder / "band-desktop-reviewed-queue-attempt-01"
    destination.mkdir(exist_ok=False)
    result = {"status": "RUNNING", "parent_review_sha256": digest(parent_path), "jobs": []}
    for identifier, runtime, command in commands:
        check_idle()
        (destination / "queue.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (destination / (identifier + ".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        report_path, attempt_path = Path(runtime["output"]) / "run.json", Path(runtime["receipt"]) / "attempt.json"
        valid = process.returncode == 0 and report_path.exists() and attempt_path.exists()
        entry = {"id": identifier, "actual_exit_code": process.returncode}
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
    result["status"] = "COMPLETED_SIX_REVIEWED_DESKTOP_OPERATIONAL_JOBS"
    (destination / "queue.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
