"""Integrate closed latent API bytes while preserving every active fit source."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-latent-builder-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    handoff = json.loads((BUILDER / FOLDER / "handoff-v1.json").read_text(encoding="utf-8"))
    authored = handoff["authored_sha256"]
    if set(authored) != {"src/marine_echo/inference/native_latent.py", "tests/unit/test_native_latent.py", "tests/integration/test_native_latent.py"}:
        raise ValueError("Only the closed bounded inference delivery may be integrated")
    for path, expected in handoff["protected_main_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Immutable active scientific source changed: {path}")
    for relative, expected in {**authored, **handoff["support_and_evidence_sha256"]}.items():
        if digest(BUILDER / relative) != expected:
            raise ValueError("Closed builder bytes changed")
    files = dict(authored)
    files.update({str(p.relative_to(BUILDER)): digest(p) for p in (BUILDER / FOLDER).iterdir() if p.is_file()})
    if any((ROOT / relative).exists() for relative in files):
        raise FileExistsError("Preserve all earlier root evidence/source")
    for relative, expected in files.items():
        raw = (BUILDER / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Delivery changed during integration")
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        if digest(target) != expected or digest(BUILDER / relative) != expected:
            raise ValueError("Transport changed bytes")
    receipt = {"status": "CLOSED_LATENT_API_SNAPSHOT_INTEGRATED", "copied_sha256": files,
               "source_bytes_identical": True, "builder_checkout_modified": False,
               "root_checks": "NOT_RUN", "real_checkpoint_replay": "NOT_RUN", "independent_review": "NOT_RUN",
               "scientific_fit_sources_unchanged": True, "fitting": False, "final_numeric_access": "NOT_RUN"}
    with (ROOT / FOLDER / "root-snapshot-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "copied": len(files)}))


if __name__ == "__main__":
    main()
