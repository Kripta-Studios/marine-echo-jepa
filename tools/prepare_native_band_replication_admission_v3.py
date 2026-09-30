"""Repair missing immutable helper binding in new proposals, preserving prior review inputs."""

import copy
import hashlib
import json
from pathlib import Path

from marine_echo.training import native_band_replication_ssl as core

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    original = ROOT / "orchestration/native_band_replication_admission_v2.json"
    target = ROOT / "orchestration/native_band_replication_admission_v3.json"
    proposal = json.loads(original.read_bytes())
    revised = copy.deepcopy(proposal)
    added = []
    for job in revised["jobs"]:
        required = core.required_sources(core.Config(method=job["approval"]["approved_config"]["method"]))
        bindings = job["approval"]["bindings"]
        missing = [path for path in required if str(path) not in bindings]
        if missing != [ROOT / "src/marine_echo/inference/native_latent.py"]:
            raise ValueError("Expected exactly the independently identified immutable helper omission")
        bindings[str(missing[0])] = digest(missing[0])
        if any(bindings[str(path)] != digest(path) for path in required):
            raise ValueError("Current actual trainer source closure differs")
        added.append({"job_id": job["id"], "path": str(missing[0]), "sha256": digest(missing[0])})
    check = copy.deepcopy(revised)
    for job in check["jobs"]:
        job["approval"]["bindings"].pop(str(ROOT / "src/marine_echo/inference/native_latent.py"))
    if check != proposal:
        raise ValueError("Only the missing helper source binding may change")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(revised, stream, indent=2)
        stream.write("\n")
    receipt = {"status": "FOUR_PROPOSALS_SOURCE_BINDING_REPAIRED_NOT_APPROVAL", "original_path": str(original),
               "original_sha256": digest(original), "revised_path": str(target), "revised_sha256": digest(target),
               "only_added_binding": added, "config_runtime_science_identical": True,
               "source_closure_checked_by_actual_trainer_function": True, "fitting": "NOT_RUN",
               "numerical_data_decoding": "NOT_RUN"}
    with (ROOT / "evidence/ssl-research-v1/band-replication-proposal-source-repair-v3.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "jobs": len(added)}))


if __name__ == "__main__":
    main()
