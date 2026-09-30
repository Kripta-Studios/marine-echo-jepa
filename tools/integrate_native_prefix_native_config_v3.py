"""Copy a closed native-configuration revision over its byte-exact root baseline."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-prefix-native-config-builder-v3")
ALLOWED = {
    "src/marine_echo/training/native_prefix_transfer.py",
    "tests/unit/test_native_prefix_transfer.py",
    "tests/integration/test_native_prefix_transfer.py",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", required=True, type=Path)
    args = parser.parse_args()
    witness_path = (
        ROOT / "evidence/ssl-research-v1/prefix-native-config-builder-exit-witness-v3.json"
    )
    witness = json.loads(witness_path.read_bytes())
    if (
        witness.get("actual_cli_exit_code") != 0
        or witness.get("session_id") != 79981
        or witness.get("builder_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
    ):
        raise ValueError("Actual closed builder CLI required")
    proof_path = args.proof.resolve()
    if proof_path.parent != BUILDER / FOLDER:
        raise ValueError("Proof must belong to this closed builder delivery")
    proof = json.loads(proof_path.read_bytes())
    baseline = json.loads((BUILDER / FOLDER / "before-source-snapshot-v3.json").read_bytes())
    if (
        proof.get("kind") != "native_prefix_native_configuration_source_proof_v3"
        or set(proof["authored_sha256"]) != ALLOWED
        or set(proof["before_source_sha256"]) != ALLOWED
        or proof.get("closed_v2_evidence_unchanged") is not True
        or proof.get("public_numeric_access") != "NOT_RUN"
    ):
        raise ValueError("Exact bounded three-path delivery required")
    if (ROOT / FOLDER).exists():
        raise FileExistsError("Preserve earlier integration and evidence")
    for name, expected in proof["protected_main_sha256"].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError(f"Protected root source changed: {name}")
    for name in ALLOWED:
        archived = baseline["files"][name]
        if (
            digest((ROOT / name).read_bytes()) != proof["before_source_sha256"][name]
            or archived["sha256"] != proof["before_source_sha256"][name]
            or digest(archived["text"].encode("utf-8")) != archived["sha256"]
        ):
            raise ValueError("Root differs from the preserved V2 baseline")
    for name, expected in baseline["closed_v2_evidence_sha256"].items():
        if digest((BUILDER / name).read_bytes()) != expected:
            raise ValueError("Historical closed builder evidence changed")
    payloads = {name: (BUILDER / name).read_bytes() for name in ALLOWED}
    for name, raw in payloads.items():
        if digest(raw) != proof["authored_sha256"][name]:
            raise ValueError("Closed authored bytes differ from proof")
    payloads.update(
        {
            str(path.relative_to(BUILDER)): path.read_bytes()
            for path in (BUILDER / FOLDER).iterdir()
            if path.is_file()
        }
    )
    for name in payloads:
        if name not in ALLOWED and (ROOT / name).exists():
            raise FileExistsError("Evidence collision must remain intact")
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb" if name in ALLOWED else "xb") as stream:
            stream.write(raw)
        if digest(destination.read_bytes()) != digest(raw):
            raise ValueError("Copy differs from closed bytes")
    receipt = {
        "status": "CLOSED_NATIVE_CONFIG_V3_INTEGRATED_ROOT_CHECKS_PENDING",
        "actual_cli_exit_witness": witness,
        "before_source_sha256": proof["before_source_sha256"],
        "authored_sha256": proof["authored_sha256"],
        "copied_sha256": {name: digest(raw) for name, raw in payloads.items()},
        "proof_sha256": digest(proof_path.read_bytes()),
        "public_numeric_access": "NOT_RUN",
        "scientific_execution": "NOT_RUN",
        "independent_review": "NOT_RUN",
        "active_fitting_sources_changed": False,
    }
    with (ROOT / FOLDER / "root-integration-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
