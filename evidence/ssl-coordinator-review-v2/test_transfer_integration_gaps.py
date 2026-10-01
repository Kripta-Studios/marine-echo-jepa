"""Coordinator-only regression probes; no data, weights, approvals or fits.

These deliberately fail until separately versioned transfer integrations exist.
They do not authorize modifying historical evaluators or relaxing access gates.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import execute_bounded_native_replication_assessment as supervisor
from marine_echo.evaluation import native_assessment_replication as assessment
from marine_echo.training import native_prefix_desktop_transfer_v3 as prefix


@pytest.mark.parametrize("module", [assessment, supervisor])
def test_reserved_evaluator_accepts_distinct_matched_cf_artifact(module):
    assert "native_cf_control_weights_only_inference_v1" in module.NEURAL_KINDS
    assert {"cf_random_frozen", "cf_direct_supervised"} <= module.METHODS


@pytest.mark.parametrize(
    ("method", "mode"),
    [("cf_random_frozen", "frozen_readout"), ("direct", "scratch_direct")],
)
def test_prefix_configuration_supports_same_cf_backbone_controls(method, mode):
    prefix.PrefixConfig(method=method, family="cf", mode=mode).validate(
        "REVIEWED_PREFIX_TRANSFER"
    )


@pytest.mark.parametrize("module", [assessment, supervisor])
def test_assessment_can_consume_saved_adaptation_artifact(module):
    assert "native_prefix_transfer_inference_v1" in module.NEURAL_KINDS


def test_existing_h96_prefix_boundaries_have_disjoint_warmup_and_common_suffix():
    from datetime import datetime

    for days in (1, 7, 30):
        start, end, suffix = prefix.boundaries("2023-02-01T00:00:00", days)
        assert start == datetime(2023, 2, 5)
        assert (end - start).days == days
        assert suffix == datetime(2023, 3, 14)
        assert (suffix - end).days >= 7
