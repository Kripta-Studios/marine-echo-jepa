"""Own one isolated CPU snapshot QA; no GPU, fitting, or scientific claims."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from package_native_pretrained_snapshot_v2 import digest, document, fresh, idle, regular

ROOT = Path(__file__).resolve().parents[1]


def execute(root, snapshot_path, destination, *, launcher=None, supervisor=None):
    root = Path(root).absolute()
    started = time.monotonic()
    destination = fresh(destination, root, "evidence")
    idle(root)
    snapshot = document(snapshot_path)
    bundle = regular(Path(snapshot["directory"]) / "manifest.json").parent
    if (
        snapshot.get("kind") != "native_pretrained_model_snapshot_receipt_v2"
        or snapshot.get("status") != "SEVEN_FIXED_PRETRAINED_MODELS_COPIED_QA_PENDING"
        or len(snapshot.get("models", [])) != 7
        or snapshot.get("manifest_sha256") != digest(bundle / "manifest.json")
        or digest(snapshot["archive"]) != snapshot.get("archive_sha256")
    ):
        raise ValueError("Exact immutable seven-model snapshot receipt required")
    for name, expected in snapshot.get("files", {}).items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or digest(bundle / path) != expected:
            raise ValueError("Copied snapshot bytes changed")
    from check_native_pretrained_snapshot_worker_v3 import validate_bundle

    validate_bundle(bundle)
    bindings = snapshot.get("source_bindings", {})
    required = [
        Path(__file__).resolve(),
        Path(__file__).with_name("check_native_pretrained_snapshot_worker_v3.py").resolve(),
        Path(__file__).with_name("package_native_pretrained_snapshot_v2.py").resolve(),
        root / "tools/native_reference_supervisor.py",
    ]
    if not bindings or any(bindings.get(str(p)) != digest(p) for p in required):
        raise ValueError("Exact original supervisor/package/worker/owner source bindings required")
    idle(root)
    destination.mkdir()
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-I",
        "-B",
        str(root / "tools/check_native_pretrained_snapshot_worker_v3.py"),
        str(bundle),
        str(destination / "completion.json"),
    ]
    record = {
        "kind": "native_pretrained_snapshot_owned_qa_receipt_v3",
        "status": "STARTING_CPU_QA",
        "command": command,
        "snapshot_sha256": digest(snapshot_path),
        "manifest_sha256": digest(bundle / "manifest.json"),
        "archive_sha256": digest(snapshot["archive"]),
        "device": "cpu",
        "fitting": False,
        "scientific_assessment": False,
        "source_bindings": {str(p): digest(p) for p in required},
    }
    with (destination / "attempt-start.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        if supervisor is None:
            from native_reference_supervisor import supervise_owned

            supervisor = supervise_owned
        if launcher is None:
            launcher = subprocess.Popen
        with (destination / "stdout.log").open("x", encoding="utf-8") as log:
            idle(root)
            child = launcher(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT)
            record["child_pid"] = child.pid
            resources = supervisor(
                child, started=started, deadline_seconds=600, rss_limit_bytes=22 * 1024**3
            )
        completion_path = destination / "completion.json"
        completion = document(completion_path) if completion_path.is_file() else {}
        peak, elapsed = (
            resources.get(key) for key in ("peak_process_rss_bytes", "elapsed_full_attempt_seconds")
        )
        bounded = (
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (peak, elapsed))
            and peak < 22 * 1024**3
            and elapsed < 600
        )
        records = completion.get("records", [])
        completed = (
            resources.get("exit_code") == 0
            and resources.get("stopped_for") is None
            and resources.get("owned_tree_cleanup_verified") is True
            and bounded
            and completion.get("status")
            == "COPIED_SEVEN_MODEL_PACKAGE_ISOLATED_CPU_CORRECTNESS_PASSED"
            and completion.get("manifest_sha256") == record["manifest_sha256"]
            and isinstance(records, list)
            and len(records) == 7
            and {r.get("id") for r in records} == {m["id"] for m in snapshot["models"]}
            and completion.get("all_imports_from_copied_source") is True
            and completion.get("cuda_initialized") is False
            and completion.get("rng_unchanged") is True
        )
        record.update(
            status="COPIED_PACKAGE_CPU_QA_PASSED" if completed else "COPIED_PACKAGE_CPU_QA_FAILED",
            resources=resources,
            elapsed_full_attempt_seconds=time.monotonic() - started,
        )
        if completion_path.is_file():
            record["completion_sha256"] = digest(completion_path)
        with (destination / "resources.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return (0 if completed else 1), record
    except BaseException as error:
        record.update(
            status="QA_PARENT_EXCEPTION_RECONCILIATION_REQUIRED",
            requires_reconciliation=True,
            exception_type=type(error).__name__,
        )
        with (destination / "parent-exception.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise


def main():
    if ROOT.name != "marine-echo-jepa":
        raise RuntimeError("Actual owned copied-model QA is ROOT-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-pretrained-model-snapshot-v2.json",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/ssl-research-v1/native-pretrained-snapshot-isolated-cpu-v3",
    )
    args = parser.parse_args()
    code, record = execute(ROOT, args.snapshot, args.receipt)
    print(
        json.dumps(
            {"status": record["status"], "actual_exit_code": record["resources"]["exit_code"]}
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
