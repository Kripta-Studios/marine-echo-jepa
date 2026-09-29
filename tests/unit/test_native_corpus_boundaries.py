"""Adversarial configuration and clock boundaries around future crops."""

from dataclasses import replace

import numpy as np
import pytest

from marine_echo.data.native_ssl_corpus import issue_windows
from tests.unit.test_native_corpus import slots


def test_intervening_future_configuration_invalidates_longer_targets():
    source = slots()
    source[97].bounds[:] = [0, 220]
    result = issue_windows(source)
    assert result["x"].shape[0] > 0
    assert not result["y_observed"][0, 2]
    assert not result["future_observed"][0, 2].any()


def test_secondary_geometry_is_not_hidden_by_cutoff_metadata():
    source = slots()
    source[50].bounds[1] = [0, 220]
    assert issue_windows(source)["x"].shape[0] == 0


def test_intervening_clock_jump_is_not_erased_by_endpoint_check():
    source = slots()
    source[97] = replace(source[97], timestamp=source[97].timestamp + np.timedelta64(10, "h"))
    result = issue_windows(source)
    assert not result["y_observed"][0, 2]


def test_secondary_processing_change_after_missing_cutoff_is_a_boundary():
    source = slots()
    source[95].observed[1] = False
    source[95] = replace(
        source[95],
        qc=(
            "OBSERVED_CENSORING_UNKNOWN",
            "MISSING_CHANNEL",
            "OBSERVED_CENSORING_UNKNOWN",
            "OBSERVED_CENSORING_UNKNOWN",
        ),
    )
    for index in range(96, 120):
        source[index] = replace(
            source[index],
            processing=("p", "new-processing", "p", "p"),
            ping_counts=(150, 180, 150, 150),
        )
    result = issue_windows(source)
    assert not result["ssl_eligible"][0]
    assert not result["y_observed"][0].any()


@pytest.mark.parametrize("status", ["MISSING_CHANNEL", "PARTIAL_SOURCE_INTERVAL", "DUPLICATE_ROW"])
@pytest.mark.parametrize("change", ["processing", "ping"])
def test_incomplete_cutoff_keeps_established_secondary_configuration(status, change):
    source = slots()
    source[95].observed[1] = False
    source[95] = replace(
        source[95],
        processing=("p", "UNKNOWN", "p", "p"),
        ping_counts=(150, 0, 150, 150),
        qc=(
            "OBSERVED_CENSORING_UNKNOWN",
            status,
            "OBSERVED_CENSORING_UNKNOWN",
            "OBSERVED_CENSORING_UNKNOWN",
        ),
    )
    for index in range(96, 120):
        if change == "processing":
            source[index] = replace(source[index], processing=("p", "new-processing", "p", "p"))
        else:
            source[index] = replace(source[index], ping_counts=(150, 180, 150, 150))
    result = issue_windows(source)
    assert not result["ssl_eligible"][0]
    assert not result["y_observed"][0].any()
