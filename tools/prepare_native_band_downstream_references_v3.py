"""Prepare two completed-parent references; no approval, fitting or decoding."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(13, 23), required=True)
    seed = parser.parse_args().seed
    source = ROOT / "orchestration" / f"native_band_replication_downstream_seed{seed}_v3.json"
    proposal = json.loads(source.read_bytes())
    if proposal.get("seed") != seed or len(proposal.get("jobs", [])) != 2:
        raise ValueError("Exactly two completed-parent proposals required")
    rows = []
    for job in proposal["jobs"]:
        config = job["approval"]["approved_config"]
        if config["seed"] != seed or config["method"] != "shared_ssl":
            raise ValueError("Only fixed shared SSL completed-parent endpoints")
        row = {"job_id": job["id"], "kind": "downstream"}
        for key in ("approved_config", "runtime_arguments"):
            value = canonical(job["approval"][key])
            row[key + "_canonical_json"] = value
            row[key + "_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        rows.append(row)
    destination = ROOT / "orchestration" / f"native_band_downstream_references_seed{seed}_v3.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump({"status": "METADATA_REFERENCES_NOT_APPROVAL", "seed": seed,
                   "proposal_path": str(source), "proposal_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "serialization": "Python JSON sort_keys/compact separators/ensure_ascii/allow_nan=False", "jobs": rows}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "METADATA_REFERENCES_NOT_APPROVAL", "seed": seed, "jobs": 2}))


if __name__ == "__main__":
    main()
