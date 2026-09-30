"""Integrate a closed bounded revision over its exact synchronized root baseline."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-prefix-completion-builder-v2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    witness = load(ROOT / "evidence/ssl-research-v1/prefix-completion-builder-exit-witness-v2.json")
    if (witness.get("actual_cli_exit_code") != 0
            or witness.get("session_id") != 19397
            or witness.get("builder_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"):
        raise ValueError("Observed closed builder CLI required before integration")
    baseline = load(BUILDER / FOLDER / "synchronized-baseline-v1.json")
    proof = load(BUILDER / FOLDER / "source-proof-final-v2.json")
    allowed = {
        "src/marine_echo/training/native_prefix_transfer.py",
        "tests/unit/test_native_prefix_transfer.py",
        "tests/integration/test_native_prefix_transfer.py",
    }
    if (set(baseline["sha256"]) != allowed or set(proof["authored_sha256"]) != allowed
            or baseline.get("byte_identical_to_root") is not True
            or proof.get("snapshots_byte_exact") is not True):
        raise ValueError("Exact three-path synchronized baseline and closed proof required")
    if (ROOT / FOLDER).exists():
        raise FileExistsError("Preserve previous integration/evidence")
    for paths in (proof["protected_main_sha256"], proof["runtime_package_sha256"]):
        for name, expected in paths.items():
            if digest(name) != expected:
                raise ValueError(f"Protected root source changed: {name}")
    for name, expected in baseline["sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Current root repair differs from synchronized builder baseline")
    for name, expected in proof["other_closed_builder_sources_unchanged"].items():
        if digest(BUILDER / name) != expected or digest(ROOT / name) != expected:
            raise ValueError("Other closed/active fitting source changed")
    files = dict(proof["authored_sha256"])
    files.update({str(path.relative_to(BUILDER)): digest(path)
                  for path in (BUILDER / FOLDER).iterdir() if path.is_file()})
    payloads = {}
    for name, expected in files.items():
        raw = (BUILDER / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Closed builder bytes changed during integration")
        if name not in allowed and (ROOT / name).exists():
            raise FileExistsError("Evidence collision must remain intact")
        payloads[name] = raw
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Original/root baseline bytes remain in exact archived delivery snapshots.
        with destination.open("wb" if name in allowed else "xb") as stream:
            stream.write(raw)
        if digest(destination) != files[name]:
            raise ValueError("Integrated bytes differ from closed delivery")
    receipt = {
        "status": "CLOSED_PREFIX_COMPATIBILITY_SNAPSHOT_INTEGRATED_ROOT_CHECKS_PENDING",
        "authored_sha256": proof["authored_sha256"], "copied_sha256": files,
        "prior_root_sha256": baseline["sha256"], "actual_cli_exit_witness": witness,
        "current_fitting_sources_changed": False, "builder_checkout_modified": False,
        "public_numeric_access": "NOT_RUN", "independent_review": "NOT_RUN",
    }
    with (ROOT / FOLDER / "root-integration-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied_files": len(files)}))


if __name__ == "__main__":
    main()
