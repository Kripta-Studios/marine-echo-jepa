"""Write exact prospective review requests; never approve or execute a fit."""

import json
from pathlib import Path

from execute_native_cf_control_job_v1 import IMPLEMENTER, ROOT, catalog, digest, read, write_new


def proposals(manifest, root):
    root = Path(root).resolve()
    if (
        manifest.get("study") != "CF_MATCHED_CONTROLS"
        or manifest.get("status") != "REGISTERED_EXACT_RECIPES_NOT_PREFIT_APPROVAL"
    ):
        raise ValueError("Exact new extension registration required")
    recipes = catalog()
    jobs = manifest.get("jobs", [])
    if len(jobs) != 6 or {j["id"] for j in jobs} != set(recipes):
        raise ValueError("All six distinct fixed control cells required")
    bindings = dict(manifest["bindings"])
    for name, expected in bindings.items():
        if digest(name) != expected:
            raise ValueError("Registered source/input bytes changed")
    train = root / "data/processed/native_ssl_v1/train.npz"
    dev = root / "data/processed/native_ssl_v1/development.npz"
    split = root / "configs/native_ssl_split_v1.json"
    results = {}
    for job in jobs:
        recipe = recipes[job["id"]]
        if (
            job.get("status") != "PREFIT_REQUIRED_NOT_RUN"
            or job.get("resume") is not None
            or job.get("parent") is not None
            or job.get("ssl_updates") != 0
            or read(job["config"]) != recipe
        ):
            raise ValueError("Exact unfitted parent-free registered configuration required")
        output = root / "outputs/native_cf_matched_controls_v1" / job["id"]
        receipt = root / "evidence/ssl-research-v1" / (job["id"] + "-attempt-01")
        review = root / "evidence/ssl-research-v1" / (job["id"] + "-prefit-review-final.json")
        if any(
            Path(job[key]).resolve() != p or p.exists()
            for key, p in (("output", output), ("receipt", receipt), ("review", review))
        ):
            raise ValueError("Exact fresh output/receipt/review destinations required")
        executor = root / "tools/execute_native_cf_control_job_v1.py"
        budget = root / "orchestration/native_cf_control_budget_owner_resolution_v1.json"
        inputs = {
            "train": str(train),
            "dev": str(dev),
            "train_cohort": str(train.with_suffix(".json")),
            "dev_cohort": str(dev.with_suffix(".json")),
            "split": str(split),
            "adr0016": str(root / "docs/adr/0016-native-ssl-selection-and-source-baseline.md"),
            "protocol": str(root / "docs/adr/0025-cf-matched-controls-continuation.md"),
            "config": job["config"],
            "review": str(review),
            "executor": str(executor),
            "budget": str(budget),
        }
        results[job["id"]] = {
            "status": "PROPOSED_CF_CONTROL_PREFIT_NOT_APPROVAL",
            "evidence_kind": "REAL_TRAIN_DEVELOPMENT_FIT",
            "implementer_session_id": IMPLEMENTER,
            "reviewer_session_id": None,
            "root_coordinator_session_id": None,
            "allowed_roles": ["train", "development"],
            "allowed_methods": [recipe["method"]],
            "allowed_modes": [
                "frozen_readout" if recipe["method"] == "cf_random_frozen" else "direct_end_to_end"
            ],
            "allowed_seeds": [recipe["seed"]],
            "approved_config": recipe,
            "train_npz_sha256": digest(train),
            "dev_npz_sha256": digest(dev),
            "split_sha256": digest(split),
            "runtime": {
                "output": str(output),
                "device": "cuda:0",
                "executor": str(executor),
                "budget": str(budget),
            },
            "execution_runtime": {
                **inputs,
                "trainer_review": str(review),
                "output": str(output),
                "receipt": str(receipt),
                "resume": None,
                "device": "cuda:0",
            },
            "bindings": bindings,
            "reserved_numeric_access": False,
            "scientific_review": "NOT_RUN",
            "review_route": "EXISTING_DISTINCT_SESSION_REQUIRES_EXTERNAL_RESTORATION",
        }
    return results


def prepare():
    manifest_path = ROOT / "orchestration/native_cf_matched_controls_v1.json"
    values = proposals(read(manifest_path), ROOT)
    paths = {
        name: ROOT / "evidence/ssl-research-v1" / (name + "-prefit-proposal-v1.json")
        for name in values
    }
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("Preserve previous prospective requests")
    for value in values.values():
        value["bindings"][str(manifest_path)] = digest(manifest_path)
        value["bindings"][str(Path(__file__).resolve())] = digest(__file__)
    for name, value in values.items():
        write_new(paths[name], value)
    return paths


if __name__ == "__main__":
    print(
        json.dumps(
            {"status": "PROPOSED_NOT_APPROVED", "proposals": len(prepare()), "training": "NOT_RUN"}
        )
    )
