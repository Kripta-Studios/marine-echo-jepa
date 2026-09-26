"""Time and sample-axis expectations for diagnostic raw-count replay."""

from __future__ import annotations

import numpy as np
import pytest

from marine_echo.data.raw_replay import aggregate_raw_count_file


def test_bin_boundary_and_sample_means() -> None:
    times = np.array(
        ["2020-02-17T00:14:59", "2020-02-17T00:15:00"], dtype="datetime64[s]"
    )
    counts = np.array([[[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0]]])
    rows = aggregate_raw_count_file(times, counts, sample_bins=2)
    assert [row["bin_start_utc"] for row in rows] == [
        "2020-02-17T00:00:00Z",
        "2020-02-17T00:15:00Z",
    ]
    assert rows[0]["counts"] == [[3.0, 7.0]]
    assert rows[1]["counts"] == [[11.0, 15.0]]


def test_reject_duplicate_or_decreasing_pings() -> None:
    counts = np.ones((1, 2, 4))
    for times in (
        np.array(["2020-02-17T00:00:00", "2020-02-17T00:00:00"], dtype="datetime64[s]"),
        np.array(["2020-02-17T00:01:00", "2020-02-17T00:00:00"], dtype="datetime64[s]"),
    ):
        with pytest.raises(ValueError):
            aggregate_raw_count_file(times, counts, sample_bins=2)
