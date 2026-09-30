"""Integrate only a closed additive replication delivery with exact source hashes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-band-replication-builder-v2")
ALLOWED = {
    "src/marine_echo/training/native_band_replication_ssl.py",
    "src/marine_echo/training/native_band_replication_downstream.py",
    "src/marine_echo/inference/native_band_replication_acoustic.py",
    "src/marine_echo/inference/native_band_replication_encoder.py",
    "src/marine_echo/inference/native_band_replication_latent.py",
    "tools/execute_native_band_replication_job.py",
    "tools/prepare_native_band_replication_configs.py",
    "tests/unit/test_native_band_replication.py",
    "tests/unit/test_native_band_replication_execution.py",
    "tests/unit/test_native_band_replication_inference.py",
    "tests/unit/test_native_band_replication_latent.py",
    "tests/integration/test_native_band_replication_ssl.py",
    "tests/integration/test_native_band_replication_downstream.py",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    witness = json.loads((ROOT / "evidence/ssl-research-v1/band-replication-builder-exit-witness-closeout-v2.json").read_bytes())
    if (witness.get("actual_cli_exit_code") != 0 or witness.get("session_id") != 20337
            or witness.get("builder_session_id") != "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"):
        raise ValueError("Actual closed recovered builder delivery required")
    handoff = json.loads((BUILDER / FOLDER / "handoff-v2.json").read_bytes())
    authored = handoff["authored_sha256"]
    if set(authored) != ALLOWED:
        raise ValueError("Exact thirteen additive files required")
    proof_path = BUILDER / FOLDER / "source-proof-final-v2b.json"
    proof = json.loads(proof_path.read_bytes())
    if (proof.get("kind") != "native_band_replication_scientific_ast_proof_v2"
            or proof.get("protected_main_unchanged") is not True
            or proof.get("architecture") != "nonlinear_frequency_conditioned_v1"
            or proof.get("budget_family") != "native_band_v1"
            or proof.get("scientific_approval") is not False):
        raise ValueError("Immutable science and budget-family proof required")
    for name, expected in proof["source_closure_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"Closed source dependency changed: {name}")
    if handoff["source_closure_sha256"] != proof["source_closure_sha256"]:
        raise ValueError("Closed handoff and proof source closures differ")
    for group in ("support_sha256", "evidence_sha256"):
        for name, expected in handoff[group].items():
            if Path(name).name != name or digest(BUILDER / FOLDER / name) != expected:
                raise ValueError("Closed declared support/evidence changed")
    for relative, expected in authored.items():
        if digest(BUILDER / relative) != expected or (ROOT / relative).exists():
            raise ValueError("Stable closed bytes and unused root paths required")
    files = dict(authored)
    files.update({str(p.relative_to(BUILDER)): digest(p)
                  for p in (BUILDER / FOLDER).iterdir() if p.is_file()})
    if any((ROOT / relative).exists() for relative in files):
        raise FileExistsError("Preserve prior source and evidence")
    payloads = {}
    for relative, expected in files.items():
        raw = (BUILDER / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Builder delivery changed during snapshot")
        payloads[relative] = raw
    for relative, raw in payloads.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        if digest(target) != files[relative]:
            raise ValueError("Snapshot changed exact bytes")
    result = {"status": "CLOSED_BAND_REPLICATION_V2_SNAPSHOT_INTEGRATED_ROOT_CHECKS_PENDING",
              "actual_cli_exit_witness": witness, "copied_sha256": files,
              "handoff_sha256": digest(BUILDER / FOLDER / "handoff-v2.json"),
              "scientific_proof_sha256": digest(proof_path),
              "old_fitted_sources_changed": False, "builder_checkout_modified": False,
              "root_checks": "NOT_RUN", "independent_prefit": "NOT_RUN",
              "scientific_fitting": False, "final_numeric_access": "NOT_RUN"}
    with (ROOT / FOLDER / "root-snapshot-integration-v2.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "copied_files": len(files)}))


if __name__ == "__main__":
    main()
