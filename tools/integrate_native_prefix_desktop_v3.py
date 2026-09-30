"""Integrate exact closed additive desktop prefix code and bounded evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-prefix-desktop-builder-v3")
ALLOWED = {"src/marine_echo/training/native_desktop_runtime_v3.py", "src/marine_echo/training/native_prefix_desktop_transfer_v3.py",
           "tools/execute_native_prefix_desktop_job_v3.py", "tools/execute_native_prefix_desktop_worker_v3.py",
           "tests/unit/test_native_prefix_desktop_execution_v3.py", "tests/integration/test_native_prefix_desktop_transfer_v3.py"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = BUILDER / FOLDER / "handoff-v3.json"
    if digest(path) != "89f6f214db18442d182bafc1ce9210af44c90b2ae833d0e930bbacd7b0357a24":
        raise ValueError("Actual closed handoff required")
    handoff = json.loads(path.read_bytes())
    if (handoff.get("status") != "CLOSED_ENGINEERING_DELIVERY" or set(handoff["authored"]) != ALLOWED
            or handoff.get("original_sources_modified") is not False
            or handoff.get("implementer_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"):
        raise ValueError("Exact bounded engineering delivery required")
    for name, record in handoff["protected_sources"].items():
        if digest(name) != record["baseline_sha256"] or record["unchanged"] is not True:
            raise ValueError("Protected current source changed")
    records = {**handoff["authored"], **handoff["support"], **handoff["evidence_files"]}
    payloads = {}
    for name, sha in records.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or (name not in ALLOWED and not relative.is_relative_to(FOLDER))):
            raise ValueError("Bounded exact delivery paths required")
        source = BUILDER / relative
        if source.is_symlink() or digest(source) != sha:
            raise ValueError("Closed delivery bytes changed")
        payloads[str(relative)] = source.read_bytes()
    payloads[str(FOLDER / "handoff-v3.json")] = path.read_bytes()
    if any((ROOT / name).exists() for name in payloads):
        raise FileExistsError("Preserve existing integrations")
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
    receipt = {"status": "CLOSED_DESKTOP_PREFIX_DELIVERY_INTEGRATED_ROOT_CHECKS_PENDING",
               "actual_cli_exit_code": 0, "actual_session_id": 63035, "actual_exit_chunk_id": "8e8f20",
               "handoff_sha256": digest(path), "authored": handoff["authored"],
               "copied_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
               "protected_sources_verified": len(handoff["protected_sources"]),
               "scientific_prefit": "NOT_APPROVED_BY_SOFTWARE_DELIVERY", "public_prefix_fitting": "NOT_RUN"}
    with (ROOT / FOLDER / "root-integration-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "files": len(payloads)}))


if __name__ == "__main__":
    main()
