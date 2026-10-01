"""Prospective requests contain no approval or execution authority."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import execute_native_cf_control_job_v1 as executor
import prepare_native_cf_control_prefit_v1 as preparation


def fixture(tmp_path):
    root = tmp_path / "SYNTHETIC_CORRECTNESS_ONLY"
    paths = [
        root / "data/processed/native_ssl_v1" / name for name in ("train.npz", "development.npz")
    ]
    paths.append(root / "configs/native_ssl_split_v1.json")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC_CORRECTNESS_ONLY opaque")
    jobs = []
    for name, recipe in executor.catalog().items():
        path = root / "configs/native_cf_controls_v1" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe))
        paths.append(path)
        jobs.append(
            {
                "id": name,
                "status": "PREFIT_REQUIRED_NOT_RUN",
                "resume": None,
                "parent": None,
                "ssl_updates": 0,
                "config": str(path),
                "output": str(root / "outputs/native_cf_matched_controls_v1" / name),
                "receipt": str(root / "evidence/ssl-research-v1" / (name + "-attempt-01")),
                "review": str(
                    root / "evidence/ssl-research-v1" / (name + "-prefit-review-final.json")
                ),
            }
        )
    return root, {
        "study": "CF_MATCHED_CONTROLS",
        "status": "REGISTERED_EXACT_RECIPES_NOT_PREFIT_APPROVAL",
        "jobs": jobs,
        "bindings": {str(p): executor.digest(p) for p in paths},
    }


def test_exact_requests_never_supply_reviewer_or_training_approval(tmp_path):
    root, manifest = fixture(tmp_path)
    before = copy.deepcopy(manifest)
    result = preparation.proposals(manifest, root)
    assert manifest == before
    assert set(result) == set(executor.catalog())
    for name, value in result.items():
        assert value["status"] == "PROPOSED_CF_CONTROL_PREFIT_NOT_APPROVAL"
        assert value["reviewer_session_id"] is None
        assert value["scientific_review"] == "NOT_RUN"
        assert value["allowed_roles"] == ["train", "development"]
        assert value["approved_config"] == executor.catalog()[name]
        assert value["execution_runtime"]["trainer_review"] == value["execution_runtime"]["review"]
        assert value["runtime"]["output"] == value["execution_runtime"]["output"]
        with pytest.raises(ValueError, match="prefit"):
            executor.review_identity(value)
    assert not (root / "outputs").exists()
    assert not (root / "evidence").exists()


@pytest.mark.parametrize(
    "mutation", ["duplicate", "status", "config", "parent", "destination", "source"]
)
def test_changed_registration_cannot_become_prospective_request(tmp_path, mutation):
    root, manifest = fixture(tmp_path)
    job = manifest["jobs"][0]
    if mutation == "duplicate":
        manifest["jobs"][1] = job
    elif mutation == "status":
        manifest["status"] = "APPROVED"
    elif mutation == "config":
        Path(job["config"]).write_text("{}")
        manifest["bindings"][job["config"]] = executor.digest(job["config"])
    elif mutation == "parent":
        job["parent"] = "fitted"
    elif mutation == "destination":
        job["output"] += "-other"
    elif mutation == "source":
        Path(next(iter(manifest["bindings"]))).write_bytes(b"CHANGED")
    with pytest.raises(ValueError):
        preparation.proposals(manifest, root)
    assert not (root / "outputs").exists()
