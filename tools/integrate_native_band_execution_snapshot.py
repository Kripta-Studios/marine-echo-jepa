"""Copy a closed band execution delivery with complete immutable source proof."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-band-execution-builder-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handoff = json.loads((BUILDER / FOLDER / "handoff-v1.json").read_text(encoding="utf-8"))
    authored = handoff["authored_sha256"]
    expected_paths = {"tools/execute_native_band_job.py", "tools/prepare_native_band_configs.py", "tests/unit/test_native_band_execution.py"}
    if set(authored) != expected_paths:
        raise ValueError("Only bounded execution delivery may be integrated")
    proof_path = BUILDER / FOLDER / "source-proof-final-v1.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("authored_sha256") != authored:
        raise ValueError("Closed proof and handoff differ")
    if not handoff.get("protected_main_unchanged") or not handoff.get("closed_prefix_three_sources_unchanged"):
        raise ValueError("Immutable scientific source preservation proof required")
    for path, expected in handoff["protected_main_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Referenced protected source changed: {path}")
    for relative, expected in handoff["support_and_evidence_sha256"].items():
        if digest(BUILDER / relative) != expected:
            raise ValueError("Closed support/evidence bytes changed")
    for relative, expected in authored.items():
        if digest(BUILDER / relative) != expected or (ROOT / relative).exists():
            raise ValueError("Stable builder bytes and protected root paths required")
    files = dict(authored)
    files.update({str(p.relative_to(BUILDER)): digest(p) for p in (BUILDER / FOLDER).iterdir() if p.is_file()})
    if any((ROOT / relative).exists() for relative in files):
        raise ValueError("Preserve all earlier evidence")
    for relative, expected in files.items():
        raw = (BUILDER / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Builder changed during snapshot")
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        if digest(target) != expected or digest(BUILDER / relative) != expected:
            raise ValueError("Snapshot changed bytes")
    receipt = {"status": "CLOSED_BAND_EXECUTION_SNAPSHOT_INTEGRATED", "copied_sha256": files,
               "source_bytes_identical": True, "builder_checkout_modified": False,
               "root_checks": "NOT_RUN", "config_generation": "NOT_RUN", "prefit": "NOT_RUN", "fitting": False}
    with (ROOT / FOLDER / "root-snapshot-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied": len(files)}))


if __name__ == "__main__":
    main()
