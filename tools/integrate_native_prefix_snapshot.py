"""Copy the closed prefix delivery without changing either historical source set."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"
FOLDER = Path("evidence/ssl-prefix-builder-v1")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    proof = json.loads(
        (BUILDER / FOLDER / "source-proof-final-v1.json").read_text(encoding="utf-8")
    )
    handoff = json.loads((BUILDER / FOLDER / "handoff-v1.json").read_text(encoding="utf-8"))
    authored = handoff["authored_sha256"]
    if authored != proof["authored_sha256"] or not proof["protected_main_unchanged"]:
        raise ValueError("Closed delivery/source preservation proof is incomplete")
    for path, expected in proof["protected_main_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Immutable scientific source changed: {path}")
    files = dict(authored)
    files.update(
        {
            str(p.relative_to(BUILDER)): digest(p)
            for p in (BUILDER / FOLDER).iterdir()
            if p.is_file()
        }
    )
    for relative, expected in files.items():
        if (ROOT / relative).exists() or digest(BUILDER / relative) != expected:
            raise ValueError("Preserve root paths and stable closed builder bytes")
    for relative, expected in files.items():
        raw = (BUILDER / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Delivery changed during integration")
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        if digest(destination) != expected or digest(BUILDER / relative) != expected:
            raise ValueError("Integrated bytes differ")
    receipt = {
        "status": "CLOSED_SNAPSHOT_INTEGRATED_ROOT_CHECKS_PENDING",
        "authored_sha256": authored,
        "copied_sha256": files,
        "source_bytes_identical": True,
        "builder_checkout_modified": False,
        "scientific_execution": "NOT_RUN",
        "independent_review": "NOT_RUN",
        "final_numeric_access": "NOT_RUN",
    }
    with (ROOT / FOLDER / "root-snapshot-integration-v1.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"status": receipt["status"], "authored": len(authored), "copied": len(files)})
    )


if __name__ == "__main__":
    main()
