"""Reviewed load-only worker; stdlib graph admission precedes scientific imports."""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from execute_bounded_native_control_assessment_v3 import (
    LOCK,
    ROOT,
    digest,
)


@contextmanager
def owned_resources(resources):
    """Preserve immutable Resources math and refuse cleanup of a changed lock."""
    resources.__enter__()
    lock = resources.lock
    original = lock.read_bytes() if lock is not None else None
    if lock is not None:
        owner = json.loads(original)
        if owner.get("pid") != os.getpid() or owner.get("output") != str(
            resources.output.resolve()
        ):
            raise RuntimeError("Resource lock is not the actual worker; preserve it.")
    try:
        yield resources
    finally:
        if lock is not None and (not lock.exists() or lock.read_bytes() != original):
            raise RuntimeError("Resource owner changed; preserve unknown lock.")
        resources.__exit__(None, None, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "receipt", "ownership-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    # The supervisor creates receipt and reserves a running ledger record before
    # this child starts. Perform the complete immutable metadata gate directly;
    # prelaunch's fresh-receipt/idle-ledger checks belong to the parent only.
    from execute_bounded_native_control_assessment_v3 import (
        LEDGER,
        OWNER,
        _policy,
        admit,
        read_json,
    )

    review = read_json(args.review)
    admitted = admit(args.manifest, args.review, args.output)
    if admitted.manifest["evidence_kind"] != "REVIEWED_FROZEN_ASSESSMENT":
        raise ValueError("Worker requires actual distinct reviewed frozen assessment.")
    for path, expected in review["bindings"].items():
        if digest(path) != expected:
            raise ValueError("Worker stale binding before scientific import.")
    expected_runtime = {
        "manifest": str(args.manifest.resolve()),
        "review": str(args.review.resolve()),
        "output": str(args.output.resolve()),
        "receipt": str(args.receipt.resolve()),
        "device": admitted.manifest["device"],
    }
    if review.get("runtime_arguments") != expected_runtime or review.get("receipt_path") != str(
        args.receipt.resolve()
    ):
        raise ValueError("Worker runtime/receipt identity differs.")
    ownership, receipt = args.ownership_output.resolve(), args.receipt.resolve()
    if (
        ownership != args.output.resolve()
        or not ownership.is_relative_to(ROOT / "evidence")
        or not receipt.is_relative_to(ROOT / "evidence")
        or not receipt.is_dir()
    ):
        raise ValueError("Exact root-owned protected destinations required.")
    for path in (
        Path(__file__).resolve(),
        ROOT / "tools/execute_bounded_native_control_assessment_v3.py",
    ):
        if review["bindings"].get(str(path)) != digest(path):
            raise ValueError("Worker/wrapper must be independently bound.")
    budget_path = ROOT / OWNER
    if (
        review.get("band_budget_status") != "ROOT_RESOLVED"
        or review.get("budget_resolution_path") != str(budget_path)
        or review["bindings"].get(str(budget_path)) != digest(budget_path)
        or review["bindings"].get(str(ROOT / ".venv/Scripts/python.exe"))
        != digest(ROOT / ".venv/Scripts/python.exe")
    ):
        raise ValueError("Worker requires exact owner/runtime binding.")
    _policy(ROOT, review["bindings"]).validate_budget(read_json(budget_path))
    ledger = read_json(ROOT / LEDGER)
    matches = [
        r
        for r in ledger.get("runs", [])
        if r.get("receipt") == str(receipt)
        and r.get("output") == str(ownership)
        and r.get("review_sha256") == digest(args.review)
        and r.get("manifest_sha256") == digest(args.manifest)
    ]
    if (
        len(matches) != 1
        or matches[0].get("status")
        != ("RUNNING_CUDA" if admitted.manifest["device"] == "cuda:0" else "RUNNING_CPU_FIT")
        or matches[0].get("operation") != "FROZEN_ASSESSMENT_NO_FITTING"
        or matches[0].get("fitting") is not False
        or type(matches[0].get("pid")) is not int
    ):
        raise ValueError("Worker requires the exact active supervisor-owned journal.")
    import psutil

    process = psutil.Process()
    if matches[0]["pid"] not in {os.getpid(), *(p.pid for p in process.parents())}:
        raise ValueError("Worker is outside the explicitly journalled owned child tree.")

    # Fixed imported modules are now source-bound. No manifest import paths,
    # optimizers, fitting policy, cache setup or provenance execution.
    if any(name == "marine_echo" or name.startswith("marine_echo.") for name in sys.modules):
        raise ValueError("Fresh worker must not inherit previously imported scientific modules.")
    sys.path.insert(0, str(ROOT / "src"))
    from marine_echo.evaluation.native_assessment_controls_v3 import execute_assessment
    from marine_echo.training.native_ssl import Resources

    device = admitted.manifest["device"]
    lock = ROOT / LOCK
    cpu_owner = None
    if device == "cpu":
        cpu_owner = json.dumps(
            {
                "pid": os.getpid(),
                "output": str(ownership),
                "device": "cpu",
                "operation": "FROZEN_ASSESSMENT_NO_FITTING",
            }
        ).encode()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(cpu_owner)
    try:
        with owned_resources(Resources(device, ownership)) as resources:
            result = execute_assessment(args.manifest, args.review, args.output)
            snapshot = resources.snapshot()
        with (receipt / "worker-resources.json").open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "role": result["role"],
                    "device": device,
                    "fitting": False,
                }
            ),
            flush=True,
        )
    finally:
        if cpu_owner is not None:
            if not lock.exists() or lock.read_bytes() != cpu_owner:
                raise RuntimeError("CPU owner changed; preserve unknown lock.")
            lock.unlink()


if __name__ == "__main__":
    main()
