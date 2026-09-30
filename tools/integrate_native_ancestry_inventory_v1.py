"""Integrate the actual closed bounded ancestry delivery without fixture trees."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-native-ancestry-inventory-builder-v1")
ALLOWED = {"src/marine_echo/evaluation/native_ancestry_inventory.py", "tools/prepare_native_research_inventory.py",
           "tests/unit/test_native_ancestry_inventory.py", "tests/integration/test_native_ancestry_inventory.py"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = BUILDER / FOLDER / "handoff-v1.json"
    if digest(path) != "c673207d217486b6eecef23acdfc35a4e44b48abcb4d1a793ed4d5be71bfe614":
        raise ValueError("Actual closed delivery hash required")
    handoff = json.loads(path.read_bytes())
    if (handoff.get("status") != "CLOSED" or set(handoff["authored"]) != ALLOWED or handoff.get("protected_changed") != []
            or handoff.get("implementer_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"):
        raise ValueError("Exact bounded delivery required")
    baseline = json.loads((BUILDER / FOLDER / "protected-baseline-v1.json").read_bytes())
    for name, snapshot in baseline.items():
        if digest(name) != snapshot["sha256"]:
            raise ValueError(f"Protected bytes changed: {name}")
    payloads = {name: (BUILDER / name).read_bytes() for name in ALLOWED}
    for name, raw in payloads.items():
        if hashlib.sha256(raw).hexdigest() != handoff["authored"][name]:
            raise ValueError("Closed authored bytes changed")
    for name, expected in handoff["evidence"].items():
        if Path(name).name != name or digest(BUILDER / FOLDER / name) != expected:
            raise ValueError("Closed bounded evidence changed")
        payloads[str(FOLDER / name)] = (BUILDER / FOLDER / name).read_bytes()
    payloads[str(FOLDER / "handoff-v1.json")] = path.read_bytes()
    if any((ROOT / name).exists() for name in payloads):
        raise FileExistsError("Preserve previous integration")
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
    receipt = {"status": "CLOSED_ANCESTRY_DELIVERY_INTEGRATED_ROOT_CHECKS_PENDING",
               "actual_cli_exit_code": 0, "session_id": 93077, "actual_exit_chunk_id": "a058d6",
               "handoff_sha256": digest(path), "authored": handoff["authored"],
               "copied_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
               "protected_files_verified": len(baseline), "actual_weight_audit": "NOT_RUN", "independent_review": "NOT_RUN"}
    with (ROOT / FOLDER / "root-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
