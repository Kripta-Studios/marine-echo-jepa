"""Supervise one reviewed frozen assessment with full owned-time accounting."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
STATUSES = {"final_test": "APPROVED_FINAL_ASSESSMENT_EXECUTION",
            "development": "APPROVED_DEVELOPMENT_ASSESSMENT_EXECUTION"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_ledger(value):
    pending = LEDGER.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(LEDGER)


def prelaunch(manifest, review, manifest_path, output):
    """No numerical decoding, model construction, RNG or subprocess in admission."""
    role = manifest.get("role")
    if (role not in STATUSES or review.get("status") != STATUSES[role]
            or review.get("scope") != "model_only_frozen_assessment"
            or review.get("allowed_roles") != [role]
            or manifest.get("evidence_kind") != "REVIEWED_FROZEN_ASSESSMENT"
            or review.get("evidence_kind") != manifest["evidence_kind"]
            or manifest.get("device") not in ("cpu", "cuda:0")
            or review.get("device") != manifest["device"]
            or review.get("output_path") != str(output.resolve())):
        raise ValueError("Exact independently reviewed real role/device/output required")
    reviewer = review.get("reviewer_session_id")
    authors = (manifest.get("implementer_session_id"), manifest.get("root_coordinator_session_id"))
    if (not all(isinstance(v, str) and v for v in (reviewer, *authors))
            or reviewer.casefold() in {value.casefold() for value in authors}
            or any(review.get(k) != manifest.get(k) for k in
                   ("implementer_session_id", "root_coordinator_session_id"))):
        raise ValueError("Distinct review identities required")
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Every immutable admission binding is required")
    for raw, expected in bindings.items():
        if digest(raw) != expected:
            raise ValueError(f"Changed assessment binding: {raw}")
    required = (Path(__file__).resolve(), ROOT / "tools/execute_native_assessment_worker.py",
                ROOT / "tools/native_reference_supervisor.py", manifest_path.resolve(),
                ROOT / "src/marine_echo/evaluation/native_assessment.py",
                ROOT / "src/marine_echo/training/native_resources.py",
                ROOT / "src/marine_echo/training/native_ssl.py")
    if any(bindings.get(str(path)) != digest(path) for path in required):
        raise ValueError("Worker, supervisor, runtime and manifest bindings required")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    prelaunch(manifest, review, args.manifest, args.output)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    if any(str(r.get("status", "")).startswith("RUNNING_") for r in ledger["runs"]) or lock.exists():
        raise ValueError("Serialize all scientific operations; preserve the existing owner")
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("Preserve all previous outputs and attempts")
    cuda = manifest["device"] == "cuda:0"
    spent = float(ledger.get("gpu_hours_spent_owned_scientific_jobs", 0))
    remaining = (float(ledger["gpu_limit_hours"]) - spent) * 3600
    deadline = min(3 * 3600, remaining) if cuda else 3 * 3600
    if deadline <= 0:
        raise ValueError("No remaining full-owned GPU allowance")
    args.receipt.mkdir(parents=True, exist_ok=False)
    ownership = args.receipt.resolve() / "ownership"
    command = [str(ROOT / ".venv/Scripts/python.exe"), "-u", "tools/execute_native_assessment_worker.py",
               "--manifest", str(args.manifest.resolve()), "--review", str(args.review.resolve()),
               "--output", str(args.output.resolve()), "--ownership-output", str(ownership)]
    record = {"id": args.output.name, "operation": "FROZEN_ASSESSMENT_NO_FITTING", "fitting": False,
              "status": "RUNNING_CUDA" if cuda else "RUNNING_CPU_FIT",
              "cpu_status_note": "Legacy scheduler exclusive CPU status; this operation performs no fitting",
              "role": manifest["role"], "device": manifest["device"], "command": command,
              "output": str(args.output), "receipt": str(args.receipt),
              "review_sha256": digest(args.review), "manifest_sha256": digest(args.manifest),
              "maximum_full_attempt_seconds": deadline}
    ledger["runs"].append(record)
    save_ledger(ledger)
    try:
        with (args.receipt / "console.log").open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            save_ledger(ledger)
            resources = supervise_owned(child, started=started, deadline_seconds=deadline,
                                        rss_limit_bytes=22 * 2**30)
        record["resources_full_attempt"] = resources
        if resources["stopped_for"] and lock.exists():
            owner = json.loads(lock.read_text(encoding="utf-8"))
            if (resources["owned_tree_cleanup_verified"]
                    and owner.get("pid") in resources["owned_process_pids"]
                    and owner.get("output") == str(ownership)):
                lock.unlink()
                record["owned_terminated_tree_lock_cleanup"] = True
            else:
                record["unknown_lock_preserved"] = True
        report = args.output / "completion.json"
        result = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        completed = (resources["exit_code"] == 0 and resources["stopped_for"] is None
                     and resources["owned_tree_cleanup_verified"]
                     and result.get("status") == "COMPLETED_FORECASTS"
                     and result.get("role") == manifest["role"]
                     and result.get("evidence_kind") == "REVIEWED_FROZEN_ASSESSMENT"
                     and not lock.exists())
        record["status"] = "COMPLETED_FROZEN_ASSESSMENT" if completed else "ASSESSMENT_FAILED_OR_BLOCKED"
        if report.exists():
            record["report_sha256"] = digest(report)
        if cuda:
            ledger["gpu_hours_spent_owned_scientific_jobs"] = spent + resources["elapsed_full_attempt_seconds"] / 3600
        save_ledger(ledger)
        with (args.receipt / "attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": record["status"], "fitting": False,
                          "exit_code": resources["exit_code"]}), flush=True)
        return 0 if completed else 1
    except BaseException:
        # Once a child exists, its supervised result must be preserved rather than
        # inventing cleanup or a successful scientific receipt after a parent error.
        record["parent_exception_requires_owned_state_reconciliation"] = True
        save_ledger(ledger)
        raise


if __name__ == "__main__":
    sys.exit(main())
