"""Bound one separately approved suffix assessment; never fit or grant access."""

import argparse
import ast
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import execute_native_prefix_matched_job_v4 as prefix_job
from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
AUTHORS = {"01a0f72d-927a-77a3-8b62-21d456f7ed85", "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"}


def required_sources():
    package = ROOT / "src/marine_echo"
    pending = [
        Path(__file__).resolve(),
        Path(prefix_job.__file__).resolve(),
        ROOT / "tools/execute_native_prefix_suffix_worker_v1.py",
        ROOT / "tools/native_reference_supervisor.py",
        ROOT / "tools/native_band_budget_history_v2.py",
        ROOT / "tools/execute_native_band_job.py",
        package / "evaluation/native_prefix_suffix_assessment_v1.py",
        package / "training/native_desktop_runtime_v3.py",
    ]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        if not path.is_file():
            raise ValueError("Missing exact source: " + str(path))
        found.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [
                    node.module or "",
                    *((node.module or "") + "." + item.name for item in node.names),
                ]
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = package.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    return sorted(found)


def validate_scope(manifest, review, output, receipt):
    if (
        manifest.get("kind") != "native_prefix_suffix_assessment_manifest_v1"
        or manifest.get("role") != "adapted_suffix"
        or manifest.get("evidence_kind") != "REVIEWED_PREFIX_SUFFIX_ASSESSMENT"
        or manifest.get("device") not in ("cpu", "cuda:0")
        or not isinstance(manifest.get("cells"), list)
        or not manifest["cells"]
    ):
        raise ValueError("Exact real adapted-suffix assessment scope required")
    authors = [manifest.get("implementer_session_id"), manifest.get("coordinator_session_id")]
    reviewer = review.get("reviewer_session_id")
    if any(
        not isinstance(v, str) or not v.strip() for v in [*authors, reviewer]
    ) or reviewer.casefold() in AUTHORS | {v.casefold() for v in authors}:
        raise ValueError("Distinct actual review required")
    expected = {
        "status": "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION",
        "scope": "prefix_suffix_assessment_execution",
        "allowed_roles": ["adapted_suffix"],
        "evidence_kind": manifest["evidence_kind"],
        "allowed_cells": manifest["cells"],
        "runtime": {"device": manifest["device"], "output": str(output.resolve())},
        "receipt_path": str(receipt.resolve()),
        "implementer_session_id": authors[0],
        "coordinator_session_id": authors[1],
    }
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact execution review, cells, runtime and receipt required")


def cpu_hours(ledger):
    value = ledger.get("cpu_assessment_hours_owned", 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Finite nonnegative CPU assessment accounting required")
    return value


def closed_status(device, completed):
    """Retain ledger vocabulary; role and fitting=False identify evaluation."""
    if device == "cuda:0":
        return "COMPLETED_REAL_CUDA" if completed else "FAILED_REAL_CUDA_ATTEMPT"
    if device == "cpu":
        return "COMPLETED_CPU_ASSESSMENT" if completed else "FAILED_CPU_ASSESSMENT"
    raise ValueError("Explicit owned assessment device required")


def charge_attempt(ledger, *, device, elapsed_seconds, spent, band_spent, cpu_spent):
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("Finite nonnegative full-owned assessment time required")
    if device == "cpu" and band_spent is None:
        ledger["cpu_assessment_hours_owned"] = cpu_spent + elapsed_seconds / 3600
    else:
        prefix_job.charge_attempt(
            ledger,
            device=device,
            elapsed_seconds=elapsed_seconds,
            spent=spent,
            band_spent=band_spent,
            cpu_spent=cpu_spent,
        )


def verify_completion(output, manifest, resources):
    path = output / "completion.json"
    if (
        resources["exit_code"] != 0
        or resources["stopped_for"] is not None
        or resources["owned_tree_cleanup_verified"] is not True
        or not path.is_file()
    ):
        return False
    result = json.loads(path.read_bytes())
    expected_names = {c["name"] for c in manifest["cells"]}
    if (
        result.get("kind") != "native_prefix_suffix_assessment_completion_v1"
        or result.get("status") not in ("COMPLETED", "NOT_ASSESSABLE")
        or result.get("role") != "adapted_suffix"
        or result.get("zero_shot") is not False
        or result.get("evidence_kind") != manifest["evidence_kind"]
        or result.get("device") != manifest["device"]
        or set(result.get("results", {})) != expected_names
        or any(
            result["results"][cell["name"]].get("zero_shot") is not cell["zero_shot"]
            for cell in manifest["cells"]
        )
    ):
        return False
    hashes = result.get("prediction_hashes", {})
    expected_paths = {str(output / (name + ".npz")) for name in expected_names}
    return set(hashes) == expected_paths and all(
        prefix_job.digest(name) == value for name, value in hashes.items()
    )


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.output, args.receipt = args.output.resolve(), args.receipt.resolve()
    if (
        not args.output.is_relative_to(ROOT / "outputs")
        or not args.receipt.is_relative_to(ROOT / "evidence")
        or args.output.exists()
        or args.receipt.exists()
    ):
        raise FileExistsError("Fresh protected output and receipt paths required")
    manifest = json.loads(args.manifest.read_bytes())
    review = json.loads(args.review.read_bytes())
    validate_scope(manifest, review, args.output, args.receipt)
    bindings = review.get("bindings", {})
    if any(
        bindings.get(str(p)) != prefix_job.digest(p)
        for p in [*required_sources(), args.manifest.resolve()]
    ):
        raise ValueError("Exact source and manifest bindings required before scientific import")
    from marine_echo.evaluation import native_prefix_suffix_assessment_v1 as assessment
    from marine_echo.training.native_desktop_runtime_v3 import validate_operational_authority

    validate_operational_authority()
    admission = assessment.admit(args.manifest, args.review, args.output)
    ledger = json.loads(LEDGER.read_bytes())
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    prefix_job.validate_idle_desktop_ledger(ledger, LEDGER, lock)
    device = manifest["device"]
    contains_band = any(a.config.family == "band" for a in admission.fitted.values())
    family = "reference" if device == "cpu" else "band" if contains_band else "cf"
    deadline, spent, band_spent = prefix_job.resource_budget(ledger, family)
    cpu_spent = cpu_hours(ledger)
    del admission
    gc.collect()
    import psutil

    memory = psutil.Process().memory_info()
    parent_peak = max(memory.rss, getattr(memory, "peak_wset", 0))
    child_ram_allowance = prefix_job.owned_child_ram_allowance(parent_peak)
    args.receipt.mkdir(parents=True, exist_ok=False)
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "tools/execute_native_prefix_suffix_worker_v1.py",
        "--manifest",
        str(args.manifest.resolve()),
        "--review",
        str(args.review.resolve()),
        "--output",
        str(args.output),
    ]
    record = {
        "id": args.receipt.name,
        "status": "RUNNING_CUDA" if device == "cuda:0" else "RUNNING_CPU_ASSESSMENT",
        "role": "adapted_suffix",
        "operation": "FROZEN_ADAPTED_SUFFIX_ASSESSMENT",
        "fitting": False,
        "device": device,
        "output": str(args.output),
        "receipt": str(args.receipt),
        "command": command,
        "review_sha256": prefix_job.digest(args.review),
        "manifest_sha256": prefix_job.digest(args.manifest),
        "deadline_seconds": deadline,
        "band_inference_present": contains_band,
        "launcher_peak_rss_bytes_before_child": parent_peak,
        "child_tree_ram_allowance_bytes": child_ram_allowance,
    }
    if band_spent is not None:
        record["budget_family"] = "native_band_v1"
        record["uses_band_model"] = True
    ledger["runs"].append(record)
    prefix_job.save_ledger(ledger)
    try:
        with (args.receipt / "console.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            prefix_job.save_ledger(ledger)
            resources = supervise_owned(
                child,
                started=started,
                deadline_seconds=deadline,
                rss_limit_bytes=child_ram_allowance,
            )
        record["resources_full_attempt"] = resources
        if resources["stopped_for"] and lock.exists():
            owner = json.loads(lock.read_bytes())
            if (
                resources["owned_tree_cleanup_verified"]
                and owner.get("pid") in resources["owned_process_pids"]
                and owner.get("output") == str(args.output)
            ):
                lock.unlink()
                record["verified_owned_dead_lock_removed"] = True
        completed = not lock.exists() and verify_completion(args.output, manifest, resources)
        record["assessment_status"] = (
            "COMPLETED_PREFIX_SUFFIX_ASSESSMENT"
            if completed
            else "PREFIX_SUFFIX_ASSESSMENT_FAILED_OR_BLOCKED"
        )
        record["status"] = closed_status(device, completed)
        if (args.output / "completion.json").is_file():
            record["report_sha256"] = prefix_job.digest(args.output / "completion.json")
        resources["elapsed_full_attempt_seconds"] = time.monotonic() - started
        charge_attempt(
            ledger,
            device=device,
            elapsed_seconds=resources["elapsed_full_attempt_seconds"],
            spent=spent,
            band_spent=band_spent,
            cpu_spent=cpu_spent,
        )
        prefix_job.save_ledger(ledger)
        with (args.receipt / "attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        print(json.dumps({"status": record["status"], "exit_code": resources["exit_code"]}))
        return 0 if completed else 1
    except BaseException:
        record["requires_reconciliation"] = True
        record["parent_exception_requires_owned_state_reconciliation"] = True
        prefix_job.save_ledger(ledger)
        raise


if __name__ == "__main__":
    sys.exit(main())
