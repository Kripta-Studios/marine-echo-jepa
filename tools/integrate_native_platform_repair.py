"""Integrate bounded builder patches after verifying exact main-before bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ledger = json.loads((ROOT / "orchestration/native_ssl_run_ledger_v1.json").read_text())
    if any(r.get("status") in ("RUNNING_CUDA", "RUNNING_CPU_FIT") for r in ledger["runs"]):
        raise ValueError("Never change bound sources during a scientific attempt.")
    if (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists():
        raise ValueError("GPU owner is still active.")
    folder = BUILDER / "evidence/ssl-platform-builder-v1"
    manifest = json.loads((folder / "patch-manifest.json").read_text())
    replacements = []
    for item in manifest["changes"]:
        target = ROOT / item["target"]
        if digest(target.read_bytes()) != item["main_before_sha256"]:
            raise ValueError(f"Main changed since bounded delivery: {target}")
        candidate = folder / (target.stem + ".after.py")
        data = candidate.read_bytes()
        if digest(data) != item["after_sha256"]:
            raise ValueError("Candidate differs from inspected patch manifest.")
        if digest((folder / item["patch"]).read_bytes()) != item["patch_sha256"]:
            raise ValueError("Patch bytes changed.")
        replacements.append((target, data))
    tests = ("test_native_cf_pooling.py", "test_native_booster_serialization.py")
    destination = ROOT / "evidence/ssl-platform-builder-v1"
    if destination.exists() or any((ROOT / "tests/unit" / name).exists() for name in tests):
        raise ValueError("New integration paths exist; preserve prior evidence.")
    destination.mkdir()
    for original in sorted(folder.rglob("*")):
        if original.is_file():
            target = destination / original.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(original.read_bytes())
    for name in tests:
        with (ROOT / "tests/unit" / name).open("xb") as stream:
            stream.write((BUILDER / "tests/unit" / name).read_bytes())
    for target, data in replacements:
        target.write_bytes(data)
    # Root-owned indexed-device variant. Preserve the builder's original script.
    text = (destination / "root_cuda_correctness.py").read_text()
    if 'choices=["cuda"]' not in text:
        raise ValueError("Unexpected correctness bootstrap device declaration.")
    with (destination / "root_cuda_correctness_indexed.py").open("x", encoding="utf-8") as stream:
        stream.write(text.replace('choices=["cuda"]', 'choices=["cuda:0"]'))
    receipt = {
        "status": "INTEGRATED_NOT_PREFIT_APPROVED",
        "changes": manifest["changes"],
        "historical_shared_source_preserved": True,
        "scientific_fits": "NOT_RUN",
    }
    with (ROOT / "evidence/ssl-research-v1/platform-integration.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
