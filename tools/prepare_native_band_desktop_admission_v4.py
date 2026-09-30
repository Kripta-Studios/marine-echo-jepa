"""Propose six unchanged recipes through the owner-authorized desktop executor."""

import copy
import hashlib
import json
from pathlib import Path

from execute_native_band_operational_job_v4 import source_paths

from marine_echo.training import native_band_operational_ssl as core

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def main():
    folder = ROOT / "evidence/ssl-research-v1"
    fixed_path = ROOT / "orchestration/native_band_replication_admission_v3.json"
    fixed = json.loads(fixed_path.read_bytes())
    direct_path = ROOT / "orchestration/native_band_direct13_ownership_retry_admission_v1.json"
    random_path = ROOT / "orchestration/native_band_random_strong_retry_admission_v1.json"
    strong_path = ROOT / "orchestration/native_band_replication_downstream_seed13_v3.json"
    strong = json.loads(strong_path.read_bytes())
    records = [(job["id"], job["approval"], fixed_path) for job in fixed["jobs"][2:]]
    direct = json.loads(direct_path.read_bytes())
    records.append((direct["id"], direct["approval"], direct_path))
    records.extend((job["id"], job["approval"], strong_path) for job in strong["jobs"])
    random = json.loads(random_path.read_bytes())
    records.append((random["id"], random["approval"], random_path))
    target = ROOT / "orchestration/native_band_desktop_admission_v4.json"
    reference_path = ROOT / "orchestration/native_band_desktop_references_v4.json"
    if target.exists() or reference_path.exists() or len(records) != 6:
        raise ValueError("Six exact prospective records and absent proposal paths required")
    jobs, refs = [], []
    for identifier, original, source in records:
        for name, expected in original["bindings"].items():
            if digest(name) != expected:
                raise ValueError(f"Original scientific binding changed: {name}")
        leaf = copy.deepcopy(original)
        leaf.update(status="PROPOSED_DESKTOP_OPERATIONAL_ADMISSION_NOT_APPROVAL",
                    required_entrypoint=str(ROOT / "tools/execute_native_band_operational_job_v4.py"),
                    approval_scope="PROPOSED: unchanged recipe under owner-authorized VLC desktop policy; renewed distinct prefit required")
        leaf.pop("reviewer_session_id", None)
        leaf.pop("referential_review", None)
        runtime = leaf["runtime_arguments"]
        review_path = folder / (identifier + "-desktop-v4-prefit-review-final.json")
        runtime.update(review=str(review_path), **{"trainer-review": str(review_path)})
        for key in ("output", "receipt", "review"):
            if Path(runtime[key]).exists():
                raise FileExistsError(f"Preserve existing destination: {runtime[key]}")
        required = [*source_paths(ROOT), *core.required_sources(core.Config(method=leaf["approved_config"]["method"], seed=leaf["approved_config"]["seed"])),
                    source, Path(__file__).resolve(), folder / "band-fixed-replication-partial-closeout-v3.json",
                    folder / "band-desktop-operational-source-proof-v3.json",
                    folder / "band-operational-control-admission-proof-v4.json"]
        leaf["bindings"].update({str(path): digest(path) for path in required})
        leaf["operational_revision"] = {
            "owner_resolution_path": str(core.native_resources.OWNER_PATH),
            "owner_resolution_sha256": digest(core.native_resources.OWNER_PATH),
            "scientific_recipe_changed": False, "artifact_schema": "UNCHANGED_TYPED_V2",
            "original_proposal_path": str(source), "original_proposal_sha256": digest(source),
            "historical_sources_unchanged": True,
        }
        # Compare every original science/runtime field, except reviewed authority/routing.
        check = copy.deepcopy(leaf)
        for key in ("status", "approval_scope", "required_entrypoint", "bindings", "operational_revision"):
            check.pop(key, None)
        expected = copy.deepcopy(original)
        for key in ("status", "approval_scope", "required_entrypoint", "bindings", "reviewer_session_id", "referential_review"):
            expected.pop(key, None)
        for key in ("review", "trainer-review"):
            check["runtime_arguments"][key] = expected["runtime_arguments"][key]
        if check != expected:
            raise ValueError("Operational proposal changed scientific fields")
        jobs.append({"id": identifier, "kind": runtime["kind"], "review_path": str(review_path), "approval": leaf})
        row = {"job_id": identifier, "kind": runtime["kind"]}
        for key in ("approved_config", "runtime_arguments"):
            value = canonical(leaf[key])
            row[key + "_canonical_json"] = value
            row[key + "_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
        refs.append(row)
    with target.open("x", encoding="utf-8") as stream:
        json.dump({"status": "SIX_UNCHANGED_RECIPES_OPERATIONAL_PROPOSAL_NOT_APPROVED_OR_FITTED", "jobs": jobs,
                   "screen_recipe_limit": 11, "band_hours_limit": 12, "aggregate_hours_limit": 96}, stream, indent=2)
        stream.write("\n")
    with reference_path.open("x", encoding="utf-8") as stream:
        json.dump({"status": "METADATA_REFERENCES_NOT_APPROVAL", "proposal_path": str(target), "proposal_sha256": digest(target),
                   "serialization": "Python JSON sort_keys/compact separators/ensure_ascii/allow_nan=False", "jobs": refs}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "SIX_UNCHANGED_RECIPES_OPERATIONAL_PROPOSAL_NOT_APPROVED_OR_FITTED", "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
