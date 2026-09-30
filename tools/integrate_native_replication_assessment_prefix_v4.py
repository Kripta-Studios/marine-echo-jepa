"""Integrate eight closed, bounded source/test files without changing fitted code."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-replication-assessment-prefix-builder-v4")
EXISTING = {"src/marine_echo/training/native_prefix_transfer.py", "tests/unit/test_native_prefix_transfer.py",
            "tests/integration/test_native_prefix_transfer.py"}
NEW = {"src/marine_echo/evaluation/native_assessment_replication.py",
       "tools/execute_bounded_native_replication_assessment.py", "tools/execute_native_replication_assessment_worker.py",
       "tests/unit/test_native_replication_assessment.py", "tests/integration/test_native_replication_assessment.py"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    witness = json.loads((ROOT / "evidence/ssl-research-v1/replication-assessment-prefix-builder-exit-witness-v4.json").read_bytes())
    handoff_path = BUILDER / FOLDER / "handoff-v4.json"
    if (witness.get("session_id") != 19181 or witness.get("actual_cli_exit_code") != 0
            or witness.get("handoff_sha256") != digest(handoff_path)):
        raise ValueError("Actual closed exact handoff required")
    handoff = json.loads(handoff_path.read_bytes())
    proof = json.loads((BUILDER / FOLDER / "source-proof-closed-v4.json").read_bytes())
    if (handoff.get("status") != "ENGINEERING_DELIVERY_CLOSED_NOT_INDEPENDENTLY_REVIEWED"
            or set(handoff["authored"]) != EXISTING | NEW or proof["authored"] != handoff["authored"]):
        raise ValueError("Exact bounded eight-path delivery required")
    for name, sha in proof["main_protected_unchanged"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Protected root baseline differs: {name}")
    for name, sha in handoff["source_closure"].items():
        if digest(name) != sha:
            raise ValueError(f"Closed current dependency differs: {name}")
    exception = proof["main_read_only_observed_external_changes"]
    key = "src/marine_echo/inference/native_band_replication_acoustic.py"
    if (set(exception) != {key}
            or exception[key]["current_sha256"] != "e689d80077152b4c784966426846bfb154eb4e16a46494f2140129c003288149"
            or digest(ROOT / key) != exception[key]["current_sha256"]):
        raise ValueError("Only the recorded root-authorized unfitted metadata repair is accepted")
    if (ROOT / FOLDER).exists() or any((ROOT / name).exists() for name in NEW):
        raise FileExistsError("Preserve prior integrations")
    payloads = {}
    for name, sha in handoff["authored"].items():
        if digest(BUILDER / name) != sha:
            raise ValueError("Closed authored source changed")
        payloads[name] = (BUILDER / name).read_bytes()
    for name, sha in handoff["evidence"].items():
        path = BUILDER / FOLDER / name
        if digest(path) != sha:
            raise ValueError("Closed evidence changed")
        payloads[str(FOLDER / name)] = path.read_bytes()
    payloads[str(FOLDER / "handoff-v4.json")] = handoff_path.read_bytes()
    before = {name: {"sha256": digest(ROOT / name), "text": (ROOT / name).read_bytes().decode("utf-8")}
              for name in EXISTING}
    for name, raw in payloads.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb" if name in EXISTING else "xb") as stream:
            stream.write(raw)
        if digest(destination) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Exact copy mismatch")
    receipt = {"status": "CLOSED_V4_INTEGRATED_ROOT_CHECKS_PENDING", "actual_cli_exit_witness": witness,
               "before_source_snapshot": before, "authored": handoff["authored"],
               "copied_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
               "root_authorized_dependency_exception": {"path": key, "current_sha256": digest(ROOT / key),
                 "cause": "Root-known unfitted loader review_sha256 metadata repair; builder reported unknown external change",
                 "evidence": "evidence/ssl-band-replication-builder-v2/root-checks-closeout-repaired-v2.json"},
               "fitted_sources_changed": False, "scientific_execution": "NOT_RUN", "final_numeric_access": "NOT_RUN"}
    with (ROOT / FOLDER / "root-integration-v4.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied_files": len(payloads)}))


if __name__ == "__main__":
    main()
