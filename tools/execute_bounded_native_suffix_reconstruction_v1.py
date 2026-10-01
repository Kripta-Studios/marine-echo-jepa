"""Supervise an approved saved adapted-suffix reconstruction, CPU only.

No review authority or numeric permission is generated here. Original source,
access, execution and reconstruction gates remain in the child calculator.
"""

import argparse
import ast
import hashlib
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
RECIPE = {
    "bootstrap_seed": 20260929,
    "bootstrap_replicates": 2000,
    "block_hours": 48,
    "block_days": 2,
    "floor": 18,
}
DEADLINE = 600
RAM_LIMIT = 22 * 2**30


def runtime(output, receipt):
    return {
        "device": "cpu",
        "output": str(output.absolute()),
        "receipt": str(receipt.absolute()),
        "deadline_seconds": DEADLINE,
        "ram_limit_bytes": RAM_LIMIT,
    }


def validate_scope(manifest, review, output, receipt):
    if (
        manifest.get("kind") != "native_prefix_suffix_reconstruction_manifest_v1"
        or manifest.get("role") != "adapted_suffix"
        or manifest.get("evidence_kind") != "REVIEWED_PREFIX_SUFFIX_ASSESSMENT"
        or manifest.get("recipe") != RECIPE
    ):
        raise ValueError("Exact real adapted-suffix reconstruction scope and recipe required")
    authors = [manifest.get(k) for k in ("implementer_session_id", "coordinator_session_id")]
    reviewer = review.get("reviewer_session_id")
    if any(
        not isinstance(v, str) or not v.strip() or v != v.strip() for v in [*authors, reviewer]
    ) or reviewer.casefold() in AUTHORS | {v.casefold() for v in authors}:
        raise ValueError("Genuinely distinct reconstruction reviewer required")
    expected = {
        "status": "APPROVED_PREFIX_SUFFIX_RECONSTRUCTION",
        "scope": "saved_adapted_suffix_reconstruction",
        "allowed_roles": ["adapted_suffix"],
        "allowed_uses": ["saved_adapted_suffix_reconstruction"],
        "evidence_kind": manifest["evidence_kind"],
        "implementer_session_id": authors[0],
        "coordinator_session_id": authors[1],
        "runtime": runtime(output, receipt),
    }
    if any(review.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact genuine scope/runtime/receipt approval required")


def cpu_hours(ledger):
    value = ledger.get("cpu_reconstruction_hours_owned", 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Finite nonnegative full-owned CPU reconstruction hours required")
    return value


def charge_attempt(ledger, elapsed_seconds):
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0
    ):
        raise ValueError("Finite nonnegative full-owned elapsed time required")
    ledger["cpu_reconstruction_hours_owned"] = cpu_hours(ledger) + elapsed_seconds / 3600


def child_ram_allowance(parent_peak_bytes):
    if type(parent_peak_bytes) is not int or not 0 <= parent_peak_bytes < RAM_LIMIT:
        raise ValueError("Owned launcher reaches RAM cap or has invalid peak")
    return RAM_LIMIT - parent_peak_bytes


def destination(path, named_root):
    original = path.absolute()
    for part in (original, *original.parents):
        if part.exists() and (
            part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("No symlink/reparse destination paths")
    normalized = original.resolve()
    if not normalized.is_relative_to(named_root.resolve()) or normalized == named_root.resolve():
        raise ValueError("Destination must stay inside its named root")
    return normalized


def close_spawn_failure(ledger, record, elapsed_seconds, error):
    """Popen raised before returning an owned child: retain and charge the attempt."""
    charge_attempt(ledger, elapsed_seconds)
    record.update(
        status="FAILED_CPU_RECONSTRUCTION_BEFORE_SPAWN",
        fitting=False,
        elapsed_owned_seconds=elapsed_seconds,
        owned_tree_cleanup_verified=True,
        spawn_error_type=type(error).__name__,
        spawn_error=str(error),
    )


def required_sources():
    package = ROOT / "src/marine_echo"
    pending = [
        Path(__file__).absolute(),
        Path(prefix_job.__file__).absolute(),
        ROOT / "tools/native_reference_supervisor.py",
        ROOT / "tools/native_band_budget_history_v2.py",
        package / "evaluation/native_suffix_reconstruction_v1.py",
        package / "training/native_desktop_runtime_v3.py",
        package / "__init__.py",
        package / "evaluation/__init__.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / ".venv/Scripts/python.exe",
        ROOT / "orchestration/native_vlc_desktop_owner_resolution_v1.json",
        ROOT / "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md",
    ]
    found = set()
    while pending:
        path = pending.pop().absolute()
        if path in found:
            continue
        if not path.is_file():
            raise ValueError("Missing exact source/dependency: " + str(path))
        found.add(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or "", *((node.module or "") + "." + a.name for a in node.names)]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                if name.startswith("marine_echo."):
                    candidate = package.joinpath(*name.split(".")[1:]).with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
    return sorted(found)


def verify_completion(output, manifest, review, manifest_hash, review_hash, resources):
    if (
        resources.get("exit_code") != 0
        or resources.get("stopped_for") is not None
        or resources.get("owned_tree_cleanup_verified") is not True
        or not output.is_file()
    ):
        return False
    result = json.loads(output.read_bytes())
    expected = {
        "kind": "native_prefix_suffix_reconstruction_completion_v1",
        "status": "RECONSTRUCTED",
        "role": "adapted_suffix",
        "zero_shot": False,
        "evidence_kind": manifest["evidence_kind"],
        "independent_scientific_approval": False,
        "manifest_sha256": manifest_hash,
        "review_sha256": review_hash,
        "reviewer_session_id": review["reviewer_session_id"],
    }
    return all(result.get(k) == v for k, v in expected.items())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker_command(manifest_path, review_path, output):
    return [
        str(ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "-m",
        "marine_echo.evaluation.native_suffix_reconstruction_v1",
        "--manifest",
        str(manifest_path),
        "--review",
        str(review_path),
        "--output",
        str(output),
    ]


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output = destination(args.output, ROOT / "outputs")
    args.receipt = destination(args.receipt, ROOT / "evidence")
    if (
        not args.output.is_relative_to(ROOT / "outputs")
        or not args.receipt.is_relative_to(ROOT / "evidence")
        or args.output.exists()
        or args.receipt.exists()
        or args.output == args.receipt
        or args.output in args.receipt.parents
        or args.receipt in args.output.parents
    ):
        raise FileExistsError("Separate fresh protected output and receipt paths required")
    from marine_echo.evaluation import native_suffix_reconstruction_v1 as reconstruction

    manifest_path, review_path = [reconstruction.regular(p) for p in (args.manifest, args.review)]
    for target in (args.output, args.receipt):
        for parent in target.parents:
            if parent.exists() and (
                parent.is_symlink() or getattr(parent.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise ValueError("No reparse output/receipt paths")
    manifest, _ = reconstruction.comparison._read_json(manifest_path)
    review, _ = reconstruction.comparison._read_json(review_path)
    validate_scope(manifest, review, args.output, args.receipt)
    bindings = review.get("bindings", {})
    for source in [*required_sources(), manifest_path]:
        source = reconstruction.regular(source)
        if bindings.get(str(source)) != digest(source):
            raise ValueError("Exact wrapper/calculator/source/manifest binding required")
    from marine_echo.training.native_desktop_runtime_v3 import validate_operational_authority

    validate_operational_authority()
    ledger = json.loads(LEDGER.read_bytes())
    prefix_job.validate_idle_desktop_ledger(
        ledger, LEDGER, ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    )
    cpu_hours(ledger)
    # Failures are charged to CPU only; original GPU/Band/CF counters are retained.
    import psutil

    memory = psutil.Process().memory_info()
    parent_peak = max(memory.rss, getattr(memory, "peak_wset", 0))
    allowance = child_ram_allowance(parent_peak)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.mkdir(parents=True, exist_ok=False)
    command = worker_command(manifest_path, review_path, args.output)
    record = {
        "id": args.receipt.name,
        "status": "RUNNING_CPU_RECONSTRUCTION",
        "role": "adapted_suffix",
        "operation": "SAVED_ADAPTED_SUFFIX_RECONSTRUCTION",
        "fitting": False,
        "device": "cpu",
        "output": str(args.output),
        "receipt": str(args.receipt),
        "command": command,
        "manifest_sha256": digest(manifest_path),
        "review_sha256": digest(review_path),
        "deadline_seconds": DEADLINE,
        "launcher_peak_rss_bytes_before_child": parent_peak,
        "child_tree_ram_allowance_bytes": allowance,
    }
    ledger["runs"].append(record)
    prefix_job.save_ledger(ledger)
    try:
        with (args.receipt / "console.log").open("x", encoding="utf-8") as log:
            try:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            except OSError as error:
                close_spawn_failure(ledger, record, time.monotonic() - started, error)
                prefix_job.save_ledger(ledger)
                with (args.receipt / "attempt.json").open("x", encoding="utf-8") as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
                print(json.dumps({"status": record["status"], "exit_code": None}))
                return 1
            record["pid"] = child.pid
            prefix_job.save_ledger(ledger)
            resources = supervise_owned(
                child, started=started, deadline_seconds=DEADLINE, rss_limit_bytes=allowance
            )
        completed = verify_completion(
            args.output,
            manifest,
            review,
            record["manifest_sha256"],
            record["review_sha256"],
            resources,
        )
        record["status"] = (
            "COMPLETED_CPU_RECONSTRUCTION" if completed else "FAILED_CPU_RECONSTRUCTION"
        )
        resources["elapsed_full_attempt_seconds"] = time.monotonic() - started
        record["resources_full_attempt"] = resources
        charge_attempt(ledger, resources["elapsed_full_attempt_seconds"])
        if args.output.is_file():
            record["report_sha256"] = digest(args.output)
        prefix_job.save_ledger(ledger)
        with (args.receipt / "attempt.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print(json.dumps({"status": record["status"], "exit_code": resources["exit_code"]}))
        return 0 if completed else 1
    except BaseException:
        record["requires_reconciliation"] = True
        record["parent_exception_requires_owned_state_reconciliation"] = True
        prefix_job.save_ledger(ledger)
        raise


if __name__ == "__main__":
    sys.exit(main())
