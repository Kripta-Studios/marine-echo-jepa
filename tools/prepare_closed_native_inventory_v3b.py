"""Reserve the fixed35 audit after correcting only supervised cadence metadata."""

import copy
import hashlib
import json
from pathlib import Path

from prepare_closed_native_inventory_v3a import ROOT


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ledger_path = ROOT / "orchestration/native_ssl_run_ledger_v1.json"
    ledger = json.loads(ledger_path.read_bytes())
    if (ledger.get("requires_reconciliation")
            or any(str(run.get("status", "")).startswith("RUNNING_") or run.get("requires_reconciliation") for run in ledger["runs"])
            or (ROOT / "evidence/ssl-builder-v1/gpu-owner.lock").exists()
            or any(ledger_path.with_suffix(suffix).exists() for suffix in (".pending", ".prefix-pending", ".band-pending", ".assessment-pending", ".reconciliation-pending"))):
        raise ValueError("Wait for actual scientific closure before the new reservation")
    original_path = ROOT / "orchestration/native_completed_inventory_v3a.json"
    original = json.loads(original_path.read_bytes())
    failure = ROOT / "evidence/ssl-research-v1/native-completed-inventory-owned-v3a/resources.json"
    failed = json.loads(failure.read_bytes())
    resources = failed["resources"]
    if resources.get("exit_code") != 1 or resources.get("owned_tree_cleanup_verified") is not True or resources.get("stopped_for") is not None:
        raise ValueError("Actual closed and preserved failed metadata audit required")
    if failed.get("manifest_sha256") != digest(original_path) or len(original["endpoints"]) != 35:
        raise ValueError("Exact failed fixed35 manifest required")
    source = ROOT / "src/marine_echo/evaluation/native_ancestry_inventory.py"
    if original["bindings"][str(source)] != "a21fc392f596dbfd7ed8575c8085169c13ccff40916fb4141b406040d5c53787":
        raise ValueError("Exact original metadata decoder revision required")
    manifest = copy.deepcopy(original)
    for path, expected in manifest["bindings"].items():
        if Path(path) != source and digest(path) != expected:
            raise ValueError("Every original non-decoder input must remain byte-identical")
    target = ROOT / "orchestration/native_completed_inventory_v3b.json"
    output = ROOT / "evidence/native-completed-inventory-v3b"
    if target.exists() or output.exists():
        raise FileExistsError("Every prospective destination must be fresh")
    manifest["output_path"] = str(output)
    manifest["bindings"][str(source)] = digest(source)
    paths = (
        original_path, failure,
        failure.parent / "stdout.log",
        ROOT / "tests/unit/test_native_inventory_direct_cadence.py",
        ROOT / "evidence/ssl-native-ancestry-inventory-builder-v1/root-direct-cadence-red-v3b.log",
        ROOT / "evidence/ssl-native-ancestry-inventory-builder-v1/root-direct-cadence-green-v3b.log",
        Path(__file__).resolve(),
    )
    manifest["bindings"].update({str(path): digest(path) for path in paths})
    with target.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "FIXED35_SUPERVISED_CADENCE_CORRECTION_RESERVED_NOT_APPROVED", "fitting": False, "final_freeze": False}))


if __name__ == "__main__":
    main()
