"""New root-only CPU resource-worker fixture, explicitly synthetic and unfitted."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = args.output.resolve()
    if (
        root.name != "marine-echo-jepa"
        or not target.is_relative_to(root / "evidence")
        or target.exists()
    ):
        raise ValueError("New root evidence directory required")
    ledger = json.loads(
        (root / "orchestration/native_ssl_run_ledger_v1.json").read_text(encoding="utf-8")
    )
    lock = root / "evidence/ssl-builder-v1/gpu-owner.lock"
    if lock.exists() or any(
        r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]
    ):
        raise ValueError("Run this engineering fixture only after the scientific queue closes")
    spec = importlib.util.spec_from_file_location(
        "assessment_worker_fixture", root / "tests/unit/test_native_assessment.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    worker = root / "tools/execute_native_assessment_worker.py"
    with pytest.MonkeyPatch.context() as patch:
        fs = helper.MemoryFS(patch)
        fs.base, fs.dirs = target, {target}
        case = helper.Case(fs)
        case.spec["loader"] = "builtin"
        case.documents["source.json"]["bindings"][str(worker)] = helper.assessment.sha(
            worker.read_bytes()
        )
        case.seal()
        case.review["bindings"][str(worker)] = helper.assessment.sha(worker.read_bytes())
        fs.files[case.review_path] = helper._encoded(case.review)
        files = dict(fs.files)
        paths = case.manifest_path, case.review_path, case.output
    target.mkdir()
    for path, payload in files.items():
        if path.parent != target:
            raise ValueError("Fixture escaped its own directory")
        with path.open("xb") as stream:
            stream.write(payload)
    ownership = target / "owned-worker-resources"
    command = [
        str(root / ".venv/Scripts/python.exe"),
        "-u",
        str(worker),
        "--manifest",
        str(paths[0]),
        "--review",
        str(paths[1]),
        "--output",
        str(paths[2]),
        "--ownership-output",
        str(ownership),
    ]
    with (target / "worker-console.log").open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed exit {completed.returncode}; preserve fixture and logs")
    resources = json.loads((ownership / "worker-resources.json").read_text(encoding="utf-8"))
    if lock.exists() or resources["peak_rss_bytes"] >= 22 * 2**30:
        raise AssertionError("Exclusive resource cleanup/limit check failed")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "actual_worker_exit_code": completed.returncode,
                "cpu_lock_removed": True,
                "fitting": False,
                "gpu_execution": False,
                "resource_receipt": str(ownership),
            }
        )
    )


if __name__ == "__main__":
    main()
