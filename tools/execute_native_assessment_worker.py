"""Own local resources around an independently admitted frozen assessment."""

import argparse
import json
import os
from pathlib import Path

from marine_echo.evaluation.native_assessment import admit, execute_assessment, sha
from marine_echo.training.native_ssl import Resources

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "review", "output", "ownership-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    # Complete recursive source/model/data admission before decode or CUDA use.
    admitted = admit(args.manifest, args.review, args.output)
    own_source = str(Path(__file__).resolve())
    if admitted.provenance["bindings"].get(own_source) != sha(Path(__file__).read_bytes()):
        raise ValueError("Worker itself must be independently source-bound")
    ownership = args.ownership_output.resolve()
    if not ownership.is_relative_to(ROOT / "evidence"):
        raise ValueError("Root-owned evidence resource receipt required")
    ownership.mkdir(exist_ok=False)
    device = admitted.manifest["device"]
    lock = ROOT / "evidence/ssl-builder-v1/gpu-owner.lock"
    cpu_owner = None
    if device == "cpu":
        # Legacy exclusive scientific lock also serializes CPU assessment.
        # No CUDA ownership or fitting is claimed for this CPU operation.
        cpu_owner = json.dumps({"pid": os.getpid(), "output": str(ownership), "device": "cpu",
                                "operation": "FROZEN_ASSESSMENT_NO_FITTING"}).encode()
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(cpu_owner)
    try:
        with Resources(device, ownership) as resources:
            result = execute_assessment(args.manifest, args.review, args.output)
            snapshot = resources.snapshot()
        with (ownership / "worker-resources.json").open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "role": result["role"],
                          "device": device, "fitting": False}), flush=True)
    finally:
        if cpu_owner is not None:
            if not lock.exists() or lock.read_bytes() != cpu_owner:
                raise RuntimeError("CPU scientific lock changed; preserve unknown ownership")
            lock.unlink()


if __name__ == "__main__":
    main()
