"""Integrate exact reviewed-path builder bytes; never edit the builder checkout."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    relative_folder = Path("evidence/ssl-assessment-builder-v1")
    proof = json.loads((BUILDER / relative_folder / "source-proof-final-v2.json").read_text(encoding="utf-8"))
    authored = proof["authored_sha256"]
    if not proof["protected_main_unchanged"] or not proof["closed_band_eight_sources_unchanged"]:
        raise ValueError("Builder source preservation proof is incomplete")
    for path, expected in proof["protected_main_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Existing scientific source changed")
    files = {**authored, **{str(path.relative_to(BUILDER)): digest(path)
                           for path in (BUILDER / relative_folder).iterdir() if path.is_file()}}
    for relative, expected in files.items():
        source, destination = BUILDER / relative, ROOT / relative
        if destination.exists() or digest(source) != expected:
            raise ValueError("Preserve root paths and bind a stable builder snapshot")
    for relative, expected in files.items():
        source, destination = BUILDER / relative, ROOT / relative
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Builder changed during snapshot; preserve partial integration")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        if digest(source) != expected or digest(destination) != expected:
            raise ValueError("Snapshot transport changed bytes")
    receipt = {"status": "SNAPSHOT_INTEGRATED_PENDING_BUILDER_CLOSEOUT",
               "authored_sha256": authored, "copied_sha256": files,
               "builder_checkout_modified": False, "source_bytes_identical": True,
               "root_durable_checks": "NOT_RUN", "independent_review": "NOT_RUN",
               "scientific_execution": "NOT_RUN", "final_numeric_access": "NOT_RUN"}
    with (ROOT / relative_folder / "root-snapshot-integration-v1.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "authored": len(authored), "copied": len(files)}))


if __name__ == "__main__":
    main()
