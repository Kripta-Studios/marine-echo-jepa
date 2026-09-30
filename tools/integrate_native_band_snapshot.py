"""Copy a hash-frozen engineering snapshot; do not approve or fit the revision."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT.parent / "marine-jepa-vnext-builder"


def main():
    source = BUILDER / "evidence/ssl-band-builder-v1"
    proof = json.loads((source / "ast-proof-final-v1.json").read_text(encoding="utf-8"))
    paths = proof["authored_sha256"]
    if proof.get("status") != "VERIFIED" or len(paths) != 8:
        raise ValueError("Require exact eight-file engineering snapshot")
    payloads = {}
    for name, expected in paths.items():
        raw = (BUILDER / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or (ROOT / name).exists():
            raise ValueError("Changed snapshot or occupied integration destination")
        payloads[name] = raw
    destination = ROOT / "evidence/ssl-band-builder-v1"
    if destination.exists():
        raise ValueError("Preserve every earlier integration")
    evidence = {str(path.relative_to(source)): path.read_bytes()
                for path in sorted(source.rglob("*")) if path.is_file()}
    for name, expected in paths.items():
        if hashlib.sha256((BUILDER / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Builder source changed during snapshot")
    for name, raw in payloads.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    for name, raw in evidence.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    receipt = {"status": "ENGINEERING_SNAPSHOT_INTEGRATED_NOT_APPROVED_FOR_FITTING",
               "builder_final_handoff": "PENDING", "authored_sha256": paths,
               "engineering_proof_sha256": hashlib.sha256(evidence["ast-proof-final-v1.json"]).hexdigest(),
               "prior_sources_modified": False, "ledger_modified": False,
               "scientific_optimizer_updates": 0, "prefit_review": "NOT_RUN",
               "budget_resolution": "OWNER_CLARIFICATION_PENDING"}
    with (ROOT / "evidence/ssl-research-v1/band-snapshot-integration.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
