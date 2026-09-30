"""Actual trainer source admission must be complete before any data or tensor load."""

import json
import os
from pathlib import Path

import pytest

from marine_echo.training import native_band_replication_downstream as downstream
from marine_echo.training import native_band_replication_ssl as core

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("index", range(4))
def test_actual_proposed_leaf_binds_both_trainers_required_source_paths(index):
    version = os.environ.get("NATIVE_REPLICATION_SOURCE_PROPOSAL_VERSION", "v2")
    assert version in ("v2", "v3")
    path = ROOT / f"orchestration/native_band_replication_admission_{version}.json"
    job = json.loads(path.read_bytes())["jobs"][index]
    approved = job["approval"]
    config = core.Config(method=approved["approved_config"]["method"])
    inputs = downstream.RunInputs(**{field: ROOT / "SYNTHETIC_METADATA_ONLY_ABSENT" / field
        for field in ("train", "dev", "train_cohort", "dev_cohort", "split", "adr0016", "protocol", "config", "review")})
    required = set(core.required_sources(config))
    required.update(p for p in downstream.required_paths(inputs, config) if p.is_relative_to(ROOT / "src"))
    missing = [str(p) for p in required if str(p) not in approved["bindings"]]
    assert not missing, f"Trainer-required source bindings missing before decode: {missing}"
    for source in required:
        assert approved["bindings"][str(source)] == core.sha256(source)
