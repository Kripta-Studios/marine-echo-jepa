"""Copy only the closed, hash-bound metadata tooling and small handoff evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
ALLOWED = {
    "tools/prepare_completed_native_inventory_v4.py",
    "tools/execute_completed_native_inventory_owned_v4.py",
    "tests/unit/test_completed_native_inventory_v4_preparation.py",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    relative = Path("evidence/ssl-inventory-preparation-builder-v4")
    handoff_path = BUILDER / relative / "handoff-v4.json"
    if digest(handoff_path) != "f9cb4171ac49ca76714fdb2f66c9e4c28702ef2db6e3d81b39f2c395865c5181":
        raise ValueError("Exact closed builder handoff required")
    handoff = json.loads(handoff_path.read_bytes())
    if (handoff.get("status") != "CLOSED_ENGINEERING_DELIVERY"
            or handoff.get("protected_sources_unchanged") is not True
            or set(handoff["files"]) != ALLOWED
            or handoff.get("scientific_review_authority") is not False):
        raise ValueError("Bounded closed engineering delivery required")
    for name, expected in handoff["dependencies"].items():
        if digest(name) != expected:
            raise ValueError(f"Closed dependency changed: {name}")
    payloads = dict(handoff["files"])
    for name, expected in handoff["top_level_evidence"].items():
        path = Path(name)
        if path.parent != relative:
            raise ValueError("Small top-level evidence only; no fixture-tree copy")
        payloads[path.as_posix()] = expected
    payloads[(relative / "handoff-v4.json").as_posix()] = digest(handoff_path)
    for name, expected in payloads.items():
        if digest(BUILDER / name) != expected or (ROOT / name).exists():
            raise ValueError(f"Exact unchanged source and absent destination required: {name}")
    destination = ROOT / relative
    if destination.exists():
        raise FileExistsError("Preserve every earlier integration")
    destination.mkdir()
    for name, expected in payloads.items():
        target = ROOT / name
        with target.open("xb") as stream:
            stream.write((BUILDER / name).read_bytes())
        if digest(target) != expected:
            raise ValueError("Exact copied bytes required")
    record = {
        "status": "EXACT_METADATA_ENGINEERING_DELIVERY_INTEGRATED_NOT_REVIEWED",
        "builder_actual_cli_session_id": 34870,
        "builder_actual_exit_code": 0,
        "builder_actual_exit_chunk": "3bff9b",
        "handoff_sha256": digest(handoff_path),
        "copied_files": payloads,
        "fixture_trees_copied": False,
        "root_checks": "NOT_RUN_AT_INTEGRATION",
        "scientific_review_authority": False,
        "source_bindings": {str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
    }
    with (destination / "root-integration-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
