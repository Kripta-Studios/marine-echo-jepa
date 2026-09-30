"""Integrate the exact closed four-file delivery and its bounded evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-native-transfer-corpus-builder-v1")
ALLOWED = {"src/marine_echo/data/native_transfer_corpus.py", "tools/materialize_native_transfer_corpus.py",
           "tests/unit/test_native_transfer_corpus.py", "tests/integration/test_native_transfer_corpus.py"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    handoff_path = BUILDER / FOLDER / "handoff-v1.json"
    if digest(handoff_path) != "6380340341b88fe794ddea04dd02a1f22a293d455b6659b3b5dd6506c2eb2139":
        raise ValueError("Actual closed delivery hash required")
    handoff = json.loads(handoff_path.read_bytes())
    if (handoff.get("status") != "CLOSED" or set(handoff["authored"]) != ALLOWED
            or handoff.get("protected_changed") != []
            or handoff.get("implementer_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"):
        raise ValueError("Exact bounded closed delivery required")
    baseline = json.loads((BUILDER / FOLDER / "baseline-source-snapshots-v1.json").read_bytes())
    for name, snapshot in baseline.items():
        if digest(name) != snapshot["sha256"]:
            raise ValueError(f"Protected baseline changed: {name}")
    if (ROOT / FOLDER).exists() or any((ROOT / name).exists() for name in ALLOWED):
        raise FileExistsError("Preserve prior integrations")
    payloads = {}
    for name, expected in handoff["authored"].items():
        if digest(BUILDER / name) != expected:
            raise ValueError("Closed authored bytes changed")
        payloads[name] = (BUILDER / name).read_bytes()
    for name, expected in handoff["evidence"].items():
        path = BUILDER / FOLDER / name
        if Path(name).name != name or digest(path) != expected:
            raise ValueError("Closed bounded evidence changed")
        payloads[str(FOLDER / name)] = path.read_bytes()
    payloads[str(FOLDER / "handoff-v1.json")] = handoff_path.read_bytes()
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        if digest(destination) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Exact byte copy failed")
    receipt = {"status": "CLOSED_NATIVE_TRANSFER_CORPUS_INTEGRATED_ROOT_CHECKS_PENDING",
               "actual_cli_exit_code": 0, "session_id": 71913, "actual_exit_chunk_id": "e1b8f3",
               "handoff_sha256": digest(handoff_path), "authored": handoff["authored"],
               "copied_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
               "protected_baseline_files_verified": len(baseline), "fitted_sources_changed": False,
               "public_numeric_access": "NOT_RUN", "scientific_fitting": "NOT_RUN", "independent_review": "NOT_RUN"}
    with (ROOT / FOLDER / "root-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
