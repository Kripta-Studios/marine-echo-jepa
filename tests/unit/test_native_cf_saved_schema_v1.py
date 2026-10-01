"""Synthetic actual-writer schemas; no saved real prediction or model access."""

import copy

import numpy as np
import pytest
from test_native_cf_matched_contrasts_v1 import api, fixture

POLICY = "alias_from_all_twelve_aligned_cf_records"


def actual_schemas():
    records = fixture()
    for record in records:
        a = record["arrays"]
        if record["name"] == "lightgbm":
            del a["cutoff"]
            continue
        n = len(a["targets"])
        a.update(
            metadata=np.zeros((n, 4, 9), np.float32),
            context_observed=np.ones((n, 96, 4), bool),
            assessment_support=np.ones((n, 3), bool),
            archive_sha256=np.full(n, "a" * 64),
            query_native_bounds_m=a["query"][..., 3:5] * 250,
            query_frequency_hz=a["query"][..., 0] * 455000,
            query_interval_seconds=a["query"][..., 1] * 3600,
            quantiles=np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
            horizons=np.array([1, 3, 6]),
        )
    return records


def run(records, policy=POLICY):
    return api.compute_cf_matched_contrasts(
        records, role="development", original_campaign_modified=False, lightgbm_cutoff_policy=policy
    )


def test_actual_schemas_preserved_and_cutoff_alias_declared_without_mutation():
    records = actual_schemas()
    original = copy.deepcopy(records)
    result = run(records)
    alias = result["cutoff_provenance"]["lightgbm"]
    assert alias["policy"] == POLICY and alias["original_field_absent"] is True
    assert len(alias["source_records"]) == 12
    assert alias["original_artifact_modified"] is False
    assert "cutoff" not in result["raw_records"]["lightgbm"]["arrays"]
    for record, before in zip(records, original, strict=True):
        assert record["arrays"].keys() == before["arrays"].keys()
        for key in record["arrays"]:
            np.testing.assert_array_equal(record["arrays"][key], before["arrays"][key])
    assert "metadata" in result["raw_records"]["cf_scratch_seed23"]["arrays"]


def test_no_implicit_lightgbm_cutoff_imputation():
    with pytest.raises(ValueError):
        run(actual_schemas(), policy=None)


@pytest.mark.parametrize(
    "damage",
    [
        "one_cutoff",
        "missing_cf",
        "lightgbm_truth",
        "lightgbm_row",
        "bounds",
        "frequency",
        "interval",
        "quantiles",
        "horizons",
        "object",
        "context",
        "metadata",
        "archive",
        "support",
    ],
)
def test_alias_never_corrects_data_or_hides_optional_provenance_mismatch(damage):
    records = actual_schemas()
    a = records[0]["arrays"]
    if damage == "one_cutoff":
        a["cutoff"][0] += 1
    elif damage == "missing_cf":
        del a["cutoff"]
    elif damage == "lightgbm_truth":
        records[-1]["arrays"]["targets"][0, 0] += 1
    elif damage == "lightgbm_row":
        records[-1]["arrays"]["row_id"][0] = "different"
    elif damage == "bounds":
        a["query_native_bounds_m"][0, 0, 1] = 200
    elif damage == "frequency":
        a["query_frequency_hz"][0, 0] = 120000
    elif damage == "interval":
        a["query_interval_seconds"][0, 0] = 1
    elif damage == "quantiles":
        a["quantiles"][0] = 0.1
    elif damage == "horizons":
        a["horizons"][0] = 2
    elif damage == "object":
        a["archive_sha256"] = a["archive_sha256"].astype(object)
    elif damage == "context":
        a["context_observed"][0, 0, 0] = False
    elif damage == "metadata":
        a["metadata"][0, 0, 0] = 1
    elif damage == "archive":
        a["archive_sha256"][0] = "b" * 64
    else:
        a["assessment_support"][0, 0] = False
    with pytest.raises(ValueError):
        run(records)


def test_declared_alias_handles_lightgbm_row_reordering():
    records = actual_schemas()
    records[-1]["arrays"] = {k: v[::-1].copy() for k, v in records[-1]["arrays"].items()}
    result = run(records)
    assert result["status"] == "CALCULATED"
    assert result["cutoff_provenance"]["lightgbm"]["original_field_absent"] is True


def test_float32_arithmetic_policy_is_explicit_and_applied_to_every_arm():
    records = fixture()
    for record in records:
        for key in ("targets", "predictions"):
            record["arrays"][key] = record["arrays"][key].astype(np.float32)
    result = api.compute_cf_matched_contrasts(
        records, role="development", original_campaign_modified=False
    )
    assert result["loss_arithmetic_dtype"] == "float64_all_records_and_ensembles"
    assert (
        result["raw_records"]["cf_learned_frozen_seed7"]["arrays"]["predictions"].dtype
        == np.float32
    )
