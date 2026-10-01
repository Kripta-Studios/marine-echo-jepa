"""Synthetic CF control assessment; original zero-shot ancestry rules retained."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests/unit"))

import execute_bounded_native_control_assessment_v3 as wrapper
import test_native_cf_controls as control_helpers
import test_native_replication_assessment as helpers

from marine_echo.evaluation import native_assessment_controls_v3 as assessment
from marine_echo.training import native_cf_controls as controls


@pytest.mark.parametrize("module", [assessment, wrapper])
@pytest.mark.parametrize(
    "identity", [assessment.IMPLEMENTER_SESSION_ID, assessment.ORIGINAL_BUILDER_SESSION_ID]
)
def test_both_actual_authors_excluded_from_review(module, identity):
    manifest = {
        "implementer_session_id": "declared-author",
        "root_coordinator_session_id": "declared-coordinator",
    }
    with pytest.raises(ValueError, match="distinct"):
        module._review({"reviewer_session_id": identity}, manifest, "unused", "unused")


@pytest.mark.parametrize("module", [assessment, wrapper])
@pytest.mark.parametrize("method", controls.METHODS)
def test_exact_control_metadata_and_old_kinds(module, method):
    config = controls.Config(method=method).to_dict()
    spec = {
        "kind": controls.INFERENCE_KIND,
        "method": method,
        "seed": 7,
        "mode": controls.Config(method=method).mode,
    }
    module._typed_control_config(
        controls.INFERENCE_KIND, config, spec, "REVIEWED_FROZEN_ASSESSMENT"
    )
    assert controls.INFERENCE_KIND in module.NEURAL_KINDS
    assert "native_prefix_transfer_inference_v1" not in module.NEURAL_KINDS


@pytest.mark.parametrize("module", [assessment, wrapper])
@pytest.mark.parametrize("damage", ["mode", "seed", "updates", "history", "kind", "boolean"])
def test_control_mismatch_rejected_before_tensor_decode(module, damage, monkeypatch):
    config = controls.Config().to_dict()
    spec = {
        "kind": controls.INFERENCE_KIND,
        "method": config["method"],
        "seed": 7,
        "mode": "frozen_readout",
    }
    if damage == "mode":
        spec["mode"] = "full_finetune"
    elif damage == "kind":
        spec["kind"] = "native_ssl_weights_only_inference_v1"
    elif damage == "seed":
        spec["seed"] = 13
    else:
        config[{"updates": "updates", "history": "history", "boolean": "seed"}[damage]] = 1
        if damage == "boolean":
            config["seed"] = True
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Decoded before metadata rejection")
    )
    with pytest.raises(ValueError):
        module._typed_control_config(spec["kind"], config, spec, "REVIEWED_FROZEN_ASSESSMENT")


@pytest.mark.parametrize("method", controls.METHODS)
def test_builtin_exact_forecasts_scalers_and_truthful_kind(method):
    artifact = control_helpers.artifact(method)
    config, statistics = artifact["config"], artifact["scalers"]
    path = ROOT / "evidence/SYNTHETIC_CORRECTNESS_ONLY-in-memory.pt"
    raw = control_helpers.codec(artifact).getvalue()
    spec = {
        "kind": controls.INFERENCE_KIND,
        "method": method,
        "seed": 7,
        "mode": controls.Config(method=method).mode,
        "model_paths": [str(path)],
    }
    before = torch.get_rng_state().clone()
    forecast = assessment._builtin(
        spec,
        {path: raw},
        config,
        statistics,
        "cpu",
        ROOT,
        expected_evidence="SYNTHETIC_CORRECTNESS_ONLY",
    )
    args = control_helpers.context(9)
    expected = control_helpers.api.load_inference(io.BytesIO(raw)).forecast(*args)
    np.testing.assert_array_equal(forecast(*args), expected)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="evidence"):
        assessment._builtin(
            spec,
            {path: raw},
            config,
            statistics,
            "cpu",
            ROOT,
            expected_evidence="REVIEWED_FROZEN_ASSESSMENT",
        )


@pytest.fixture
def control_case(monkeypatch):
    monkeypatch.setattr(helpers, "assessment", assessment)
    monkeypatch.setattr(
        helpers, "TEST_SOURCE_PATHS", assessment.required_sources({"spy": helpers.spy_factory})
    )
    fs = helpers.MemoryFS(monkeypatch)
    case = helpers.Case(fs, learned=True)
    artifact = control_helpers.artifact()
    artifact["config"]["batch_size"] = 8
    artifact["core_config"]["batch_size"] = 8
    case.config.clear()
    case.config.update(artifact["config"])
    case.statistics.clear()
    case.statistics.update(artifact["scalers"])
    case.spec.update(
        method="cf_random_frozen",
        mode="frozen_readout",
        kind=controls.INFERENCE_KIND,
        loader="builtin",
    )
    fs.files[case.base / "model.bin"] = control_helpers.codec(artifact).getvalue()
    case.seal()
    return case


def test_actual_assessment_outputs_matched_control_with_native_truth_masks(control_case):
    case = control_case
    result = case.run()
    assert result["methods"]["fixed"]["feature_training_kind"] == "untrained_control"
    with np.load(io.BytesIO(case.fs.files[case.output / "fixed.npz"]), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["targets"], case.arrays["targets"])
        np.testing.assert_array_equal(saved["observed"], case.arrays["target_observed"])
        np.testing.assert_allclose(saved["query_native_bounds_m"][..., 1], 230, atol=1e-5)
    assert result["scientific_claim"] is None


def test_reserved_fitted_ancestry_still_rejected(control_case, monkeypatch):
    case = control_case
    cohort = case.documents["train-cohort.json"]
    cohort["identities"] = [helpers.identity("test")]
    case.seal()
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No weight decode"))
    with pytest.raises(ValueError):
        case.admit()


def test_control_sources_are_bound_in_both_layers():
    paths = {str(p) for p in assessment.required_sources()}
    bounded = {str(p) for p in wrapper.required_sources()}
    for leaf in ("training/native_cf_controls.py", "inference/native_cf_controls.py"):
        assert str(ROOT / "src/marine_echo" / leaf) in paths & bounded
