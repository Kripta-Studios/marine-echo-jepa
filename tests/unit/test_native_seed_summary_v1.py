"""Synthetic score summaries preserve the fixed campaign and seed estimand."""

from copy import deepcopy

import pytest

from marine_echo.evaluation.native_seed_summary_v1 import (
    CAMPAIGN_METHODS,
    GROUPS,
    summarize,
)


def result():
    return {
        "role": "development",
        "methods": {name: {"metrics": {"primary_pinball_db": 0.5}} for name in CAMPAIGN_METHODS},
    }


def test_seed_mean_and_sample_sd_are_not_an_ensemble_score():
    value = result()
    names = GROUPS["cf_strong_frozen"]
    for name, score in zip(names, [1.0, 2.0, 3.0], strict=True):
        value["methods"][name]["metrics"]["primary_pinball_db"] = score
    report = summarize(value)
    group = report["seed_score_groups"]["cf_strong_frozen"]
    assert group["mean_seed_score_db"] == 2
    assert group["sample_sd_db"] == 1
    assert group["seeds"] == [7, 13, 23]
    assert report["ensemble_scores"] == "NOT_COMPUTED"


@pytest.mark.parametrize("size", [28, 40, 43, 46])
def test_partial_inventory_cannot_be_presented_as_47_method_completion(size):
    value = result()
    value["methods"] = dict(list(value["methods"].items())[:size])
    with pytest.raises(ValueError, match="47"):
        summarize(value)


def test_same_count_with_substituted_or_extension_method_is_rejected():
    value = result()
    item = value["methods"].pop(GROUPS["band_strong_frozen"][2])
    value["methods"]["cf_random_frozen_seed23"] = item
    with pytest.raises(ValueError, match="identities"):
        summarize(value)


def test_missing_seed_score_is_not_dropped_from_the_average():
    value = result()
    value["methods"][GROUPS["cf_strong_frozen"][1]]["metrics"]["primary_pinball_db"] = None
    group = summarize(value)["seed_score_groups"]["cf_strong_frozen"]
    assert group["status"] == "NOT_ASSESSABLE"
    assert group["mean_seed_score_db"] is None
    assert group["sample_sd_db"] is None


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -1, "0.5"])
def test_invalid_scientific_score_is_rejected(score):
    value = result()
    value["methods"]["lightgbm"]["metrics"]["primary_pinball_db"] = score
    with pytest.raises((ValueError, TypeError)):
        summarize(value)


def test_report_keeps_probe_frozen_finetune_and_scratch_groups_separate():
    value = result()
    before = deepcopy(value)
    report = summarize(value)
    assert set(report["seed_score_groups"]) == {
        "cf_short_probe",
        "cf_strong_frozen",
        "cf_full_finetune",
        "shared_scratch",
        "band_short_probe",
        "band_strong_frozen",
        "band_full_finetune",
        "band_scratch",
    }
    assert report["neural_endpoints"] == 43
    assert report["references"] == 4
    assert value == before


def test_final_results_cannot_be_mixed_into_development_summary():
    value = result()
    value["role"] = "final_test"
    with pytest.raises(ValueError, match="development"):
        summarize(value)
