"""Retain the closed builder handoff and root-only correctness evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = Path("evidence/ssl-band-builder-v1")
    copied = []
    for name in ("NATIVE_BAND_CLOSEOUT_20260930_V1.md", "closeout-handoff-20260930-v1.json"):
        source, target = BUILDER / folder / name, ROOT / folder / name
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        copied.append({"path": str(folder / name), "sha256": digest(target)})
    original = ROOT / folder / "band_test_support.py.builder-original-v1.txt"
    with original.open("xb") as stream:
        stream.write((BUILDER / folder / "band_test_support.py").read_bytes())
    proof = json.loads((ROOT / folder / "ast-proof-final-v1.json").read_text(encoding="utf-8"))
    current = {name: digest(ROOT / name) for name in proof["authored_sha256"]}
    delta = [name for name, expected in proof["authored_sha256"].items() if current[name] != expected]
    if delta != ["tests/unit/test_native_band_inference.py"]:
        raise ValueError("Unexpected integrated authored source changes")
    log = ROOT / "evidence/ssl-research-v1/band-root-durable-tests-02.log"
    if "83 passed" not in log.read_text(encoding="utf-8"):
        raise ValueError("Actual root durable correctness completion absent")
    receipt = {"status": "BAND_ENGINEERING_DELIVERY_CLOSED_NOT_APPROVED_FOR_FITTING",
               "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "copied_closeout": copied,
               "current_authored_sha256": current,
               "root_test_import_only_delta": delta,
               "test_fixture_correction": "Real NumPy codec on explicit virtual file handle; no fitter/factory or denied-path retry",
               "root_fixture_sha256": digest(ROOT / folder / "band_test_support.py"),
               "builder_original_fixture_sha256": digest(original),
               "root_first_durable_run": "6 failed,77 passed; fixture FileNotFoundError retained",
               "root_second_durable_run": "83 passed64.18s; real synthetic CPU optimizers/resume/inference",
               "durable_log_sha256": digest(log),
               "model_training_inference_sources_modified_since_builder_proof": False,
               "prior_scientific_sources_modified": False, "scientific_optimizer_updates": 0,
               "budget_resolution": "OWNER_CLARIFICATION_PENDING", "prefit": "NOT_RUN"}
    with (ROOT / "evidence/ssl-research-v1/band-engineering-integration-final.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
