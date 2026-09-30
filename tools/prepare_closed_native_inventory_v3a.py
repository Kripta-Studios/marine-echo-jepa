"""Reserve the fixed historical 35-endpoint audit using a closed ledger snapshot."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    if (ledger.get("requires_reconciliation")
            or any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Wait for closed scientific ownership and every journal")
    original_path = ROOT / "orchestration/native_completed_inventory_v3.json"
    original = json.loads(original_path.read_bytes())
    if len(original.get("endpoints", {})) != 35:
        raise ValueError("Exactly the fixed historical 35 endpoints required")
    # This is a fresh reservation, not an amendment or approval of the stale manifest.
    bindings = dict(original["bindings"])
    old_ledger_sha256 = bindings.pop(str(ledger_path))
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise ValueError(f"Original non-ledger input changed: {path}")
    folder = ROOT / "evidence/ssl-research-v1/native-fixed35-reservation-v3a"
    target = ROOT / "orchestration/native_completed_inventory_v3a.json"
    output = ROOT / "evidence/native-completed-inventory-v3a"
    if folder.exists() or target.exists() or output.exists():
        raise FileExistsError("Every prospective reservation and audit destination must be fresh")
    if ledger_path.read_bytes() != ledger_bytes:
        raise ValueError("Ledger changed while preparing the immutable capture")
    folder.mkdir()
    snapshot = folder / "closed-ledger.json"
    with snapshot.open("xb") as stream:
        stream.write(ledger_bytes)
    manifest = copy.deepcopy(original)
    manifest["output_path"] = str(output)
    manifest["bindings"] = bindings
    manifest["bindings"].update({str(path): digest(path) for path in (snapshot, original_path, Path(__file__).resolve())})
    record = {
        "status": "FIXED35_METADATA_RESERVATION_NOT_COMPLETE_PROGRAMME_FREEZE",
        "endpoint_count": 35,
        "ledger_snapshot_sha256": digest(snapshot),
        "previous_live_ledger_binding_sha256": old_ledger_sha256,
        "fitting_authorized": False,
        "selection_authorized": False,
        "final_numeric_access": False,
        "newer_completed_endpoints_included": False,
        "bindings": {str(path): digest(path) for path in (original_path, snapshot, Path(__file__).resolve())},
    }
    with (folder / "reservation.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    if ledger_path.read_bytes() != ledger_bytes:
        raise ValueError("Ledger changed after snapshot; preserved receipt grants no audit authority")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "fitting_authorized": False}))


if __name__ == "__main__":
    main()
