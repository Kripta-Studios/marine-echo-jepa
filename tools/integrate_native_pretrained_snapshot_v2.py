"""Integrate only the closed seven-model packaging/CPU-QA tooling and evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
ALLOWED = {
    "tools/package_native_pretrained_snapshot_v2.py",
    "tools/check_native_pretrained_snapshot_worker_v3.py",
    "tools/check_native_pretrained_snapshot_owned_v3.py",
    "tests/unit/test_native_pretrained_snapshot_v2_contract.py",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    relative = Path("evidence/ssl-pretrained-snapshot-builder-v2")
    handoff_path = BUILDER / relative / "handoff-v2.json"
    if digest(handoff_path) != "e8619f884cc76450ed800148d390b7b73e01f0b5d4151da42100c516914d3bf9":
        raise ValueError("Exact closed seven-model builder handoff required")
    handoff = json.loads(handoff_path.read_bytes())
    if handoff.get("status") != "CLOSED_ENGINEERING_DELIVERY" or handoff.get("protected_sources_unchanged") is not True or set(handoff["files"]) != ALLOWED:
        raise ValueError("Exact bounded closed delivery required")
    for name, expected in handoff["dependencies"].items():
        if digest(name) != expected:
            raise ValueError(f"Closed protected dependency changed: {name}")
    payloads = dict(handoff["files"])
    for name, expected in handoff["evidence_files"].items():
        if Path(name).name != name:
            raise ValueError("Only explicit top-level small evidence; no fixture tree")
        payloads[(relative / name).as_posix()] = expected
    payloads[(relative / "handoff-v2.json").as_posix()] = digest(handoff_path)
    if (ROOT / relative).exists():
        raise FileExistsError("Preserve previous integrations")
    for name, expected in payloads.items():
        if digest(BUILDER / name) != expected or (ROOT / name).exists():
            raise ValueError(f"Exact bytes and absent destinations required: {name}")
    (ROOT / relative).mkdir()
    for name, expected in payloads.items():
        with (ROOT / name).open("xb") as stream:
            stream.write((BUILDER / name).read_bytes())
        if digest(ROOT / name) != expected:
            raise ValueError("Copied bytes differ")
    record = {
        "status": "SEVEN_MODEL_PACKAGING_ENGINEERING_INTEGRATED_ROOT_QA_PENDING",
        "actual_builder_cli_session_id": 68045,
        "actual_builder_exit_code": 0,
        "actual_builder_exit_chunk": "b5af89",
        "handoff_sha256": digest(handoff_path),
        "copied_files": payloads,
        "fixture_trees_copied": False,
        "scientific_authority": False,
        "source_bindings": {str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
    }
    with (ROOT / relative / "root-integration-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
