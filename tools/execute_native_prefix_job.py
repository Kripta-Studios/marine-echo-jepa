"""Supervise one admitted prefix fit and charge its full owned process lifetime."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from native_reference_supervisor import supervise_owned

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "orchestration/native_ssl_run_ledger_v1.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_scope(manifest, review, output, receipt):
    expected = {
        "kind": "native_prefix_transfer_manifest_v1",
        "role": "prefix_transfer",
        "evidence_kind": "REVIEWED_PREFIX_TRANSFER",
    }
    config = manifest.get("config", {})
    if (
        any(manifest.get(k) != v for k, v in expected.items())
        or manifest.get("suffix_numeric_path") is not None
        or review.get("status") != "APPROVED_PREFIX_TRANSFER_PREFIT"
        or review.get("scope") != "native_prefix_transfer_fit"
        or review.get("role") != "prefix_transfer"
        or review.get("evidence_kind") != manifest["evidence_kind"]
        or review.get("config") != config
        or review.get("output_path") != str(output.resolve())
        or review.get("receipt_path") != str(receipt.resolve())
        or config.get("device") != "cuda:0"
        or config.get("family") not in ("core", "cf", "band")
        or config.get("mode") not in ("frozen_readout", "scratch_direct")
        or type(config.get("seed")) is not int
        or config["seed"] not in (7, 13, 23)
        or type(config.get("prefix_days")) is not int
        or config["prefix_days"] not in (1, 7, 30)
        or config.get("updates") != 2000
        or config.get("cadence") != 500
    ):
        raise ValueError("Exact real one-cell prefix-only CUDA admission required")
    authors = [manifest.get("implementer_session_id"), manifest.get("coordinator_session_id")]
    reviewer = review.get("reviewer_session_id")
    if (
        any(not isinstance(value, str) or not value for value in [*authors, reviewer])
        or reviewer.casefold() in {value.casefold() for value in authors}
        or any(
            review.get(k) != manifest.get(k)
            for k in ("implementer_session_id", "coordinator_session_id")
        )
    ):
        raise ValueError("Review must be distinct from both authors")
    return config


def save_ledger(value):
    pending = LEDGER.with_suffix(".prefix-pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(LEDGER)


def resource_budget(ledger, family):
    if any(str(r.get("status", "")).startswith("RUNNING_") for r in ledger["runs"]):
        raise ValueError("Another scientific operation remains active")
    spent = ledger.get("gpu_hours_spent_owned_scientific_jobs")
    if (
        not isinstance(spent, (int, float))
        or isinstance(spent, bool)
        or not math.isfinite(spent)
        or spent < 0
        or ledger.get("gpu_limit_hours") != 96
    ):
        raise ValueError("Verified finite aggregate96-hour accounting required")
    remaining, band_spent = (96 - spent) * 3600, None
    if family == "band":
        from execute_native_band_job import budget_totals

        aggregate, band_spent, band_remaining = budget_totals(ledger)
        if aggregate != spent:
            raise ValueError("Band and aggregate accounting disagree")
        remaining = min(remaining, band_remaining)
    deadline = min(2 * 3600, remaining)
    if deadline <= 0:
        raise ValueError("No remaining owned GPU allowance")
    return deadline, spent, band_spent


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.output, args.receipt = args.output.resolve(), args.receipt.resolve()
    if (
        not args.output.is_relative_to(ROOT / "outputs")
        or not args.receipt.is_relative_to(ROOT / "evidence")
        or args.receipt.exists()
        or (args.output.exists() and args.resume is None)
        or (
            args.resume is not None
            and (not args.resume.resolve().is_relative_to(args.output) or not args.resume.is_file())
        )
    ):
        raise FileExistsError("Separate protected root outputs and explicit owned resume required")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    config = validate_scope(manifest, review, args.output, args.receipt)
    bindings = review.get("bindings", {})
    required = [
        Path(__file__).resolve(),
        ROOT / "tools/execute_native_prefix_worker.py",
        ROOT / "tools/native_reference_supervisor.py",
        args.manifest.resolve(),
        ROOT / "src/marine_echo/training/native_prefix_transfer.py",
        ROOT / "src/marine_echo/training/native_resources.py",
        ROOT / "orchestration/native_prefix_execution_scope_v1.json",
    ]
    if config["family"] == "band":
        required += [
            ROOT / "tools/execute_native_band_job.py",
            ROOT / "orchestration/native_band_budget_owner_resolution_v1.json",
        ]
    if args.resume is not None:
        required.append(args.resume.resolve())
    if not bindings or any(str(path) not in bindings for path in required):
        raise ValueError("Exact worker/source/runtime/manifest/scope bindings required")
    for name, expected in bindings.items():
        if digest(name) != expected:
            raise ValueError(f"Changed prefix prefit binding: {name}")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    if (
        any(str(r.get("status", "")).startswith("RUNNING_") for r in ledger["runs"])
        or lock.exists()
        or any(
            LEDGER.with_suffix(s).exists() for s in (".pending", ".prefix-pending", ".band-pending")
        )
    ):
        raise ValueError("Wait for prior scientific completion; preserve pending owner/journal")
    if config["family"] == "band":
        from execute_native_band_job import validate_budget

        resolution = json.loads(
            (ROOT / "orchestration/native_band_budget_owner_resolution_v1.json").read_text(
                encoding="utf-8"
            )
        )
        validate_budget(resolution)
    deadline, spent, band_spent = resource_budget(ledger, config["family"])
    # Prefix admission snapshots bytes/JSON only; no tensors, arrays or RNG setup.
    from marine_echo.training.native_prefix_transfer import admit

    admission = admit(args.manifest, args.review, args.output, resume=args.resume)
    if (
        len(
            admission.documents[
                next(
                    p
                    for p in admission.documents
                    if str(p)
                    == str((args.manifest.resolve().parent / manifest["raw_intervals"]).resolve())
                )
            ]["sources"]
        )
        != 1
    ):
        raise ValueError("Each scientific adaptation cell must contain one true deployment")
    args.receipt.mkdir(parents=True, exist_ok=False)
    command = [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "tools/execute_native_prefix_worker.py",
        "--manifest",
        str(args.manifest.resolve()),
        "--review",
        str(args.review.resolve()),
        "--output",
        str(args.output),
    ]
    if args.resume is not None:
        command += ["--resume", str(args.resume.resolve())]
    record = {
        "id": args.receipt.name,
        "status": "RUNNING_CUDA",
        "role": "prefix_transfer",
        "operation": "PREFIX_FIT_ORIGINAL_DEV_SELECTION",
        "fitting": True,
        "device": "cuda:0",
        "family": config["family"],
        "config": config,
        "output": str(args.output),
        "receipt": str(args.receipt),
        "command": command,
        "review_sha256": digest(args.review),
        "manifest_sha256": digest(args.manifest),
        "deadline_seconds": deadline,
    }
    if band_spent is not None:
        record["budget_family"] = "native_band_v1"
    ledger["runs"].append(record)
    save_ledger(ledger)
    try:
        with (args.receipt / "console.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            save_ledger(ledger)
            resources = supervise_owned(
                child, started=started, deadline_seconds=deadline, rss_limit_bytes=22 * 2**30
            )
        record["resources_full_attempt"] = resources
        if resources["stopped_for"] and lock.exists():
            owner = json.loads(lock.read_text(encoding="utf-8"))
            if (
                resources["owned_tree_cleanup_verified"]
                and owner.get("pid") in resources["owned_process_pids"]
                and owner.get("output") == str(args.output)
            ):
                lock.unlink()
                record["verified_owned_dead_lock_removed"] = True
        path = args.output / "completion.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        completed = (
            resources["exit_code"] == 0
            and resources["stopped_for"] is None
            and resources["owned_tree_cleanup_verified"] is True
            and not lock.exists()
            and result.get("status") in ("COMPLETED", "NOT_ASSESSABLE")
            and result.get("evidence_kind") == "REVIEWED_PREFIX_TRANSFER"
        )
        record["status"] = (
            (
                "COMPLETED_REAL_PREFIX_TRANSFER"
                if result.get("status") == "COMPLETED"
                else "PREFIX_TRANSFER_NOT_ASSESSABLE"
            )
            if completed
            else "PREFIX_TRANSFER_FAILED_OR_BLOCKED"
        )
        if path.exists():
            record["report_sha256"] = digest(path)
        # Include startup, data validation, output verification and failed attempts.
        resources["elapsed_full_attempt_seconds"] = time.monotonic() - started
        ledger["gpu_hours_spent_owned_scientific_jobs"] = (
            spent + resources["elapsed_full_attempt_seconds"] / 3600
        )
        if band_spent is not None:
            ledger["native_band_gpu_hours_spent_full_owned"] = (
                band_spent + resources["elapsed_full_attempt_seconds"] / 3600
            )
        save_ledger(ledger)
        with (args.receipt / "attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": record["status"], "exit_code": resources["exit_code"]}))
        return 0 if completed else 1
    except BaseException:
        record["requires_reconciliation"] = True
        record["parent_exception_requires_owned_state_reconciliation"] = True
        save_ledger(ledger)
        raise


if __name__ == "__main__":
    sys.exit(main())
