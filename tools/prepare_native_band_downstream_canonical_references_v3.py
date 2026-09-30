"""Expose canonical metadata strings for independently checked compact references."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def main():
    proposal_path = ROOT / "orchestration/native_band_downstream_admission_v1.json"
    proposal = json.loads(proposal_path.read_bytes())
    rows = []
    for job in proposal["jobs"]:
        row = {"id": job["id"]}
        for key in ("approved_config", "runtime_arguments"):
            value = canonical(job["approval"][key])
            row[key + "_canonical_json"] = value
            row[key + "_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        rows.append(row)
    target = ROOT / "orchestration/native_band_downstream_canonical_references_v3.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"status": "METADATA_REFERENCES_NOT_APPROVAL", "proposal_path": str(proposal_path),
                   "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
                   "serialization": "Python JSON sort_keys/compact separators/ensure_ascii/allow_nan=False",
                   "jobs": rows}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "METADATA_REFERENCES_NOT_APPROVAL", "jobs": len(rows)}))


if __name__ == "__main__":
    main()
