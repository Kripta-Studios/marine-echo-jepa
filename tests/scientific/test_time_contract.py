import numpy as np
import pytest

from marine_echo.evaluation.protocol import calendar_split
from marine_echo.features.windows import context_indices, target_indices


def test_split_uses_complete_catalog_days() -> None:
    split = calendar_split("2020-02-16T12:26:44Z", "2020-08-02T10:56:00Z")
    assert split["complete_days"] == 167
    assert split["partitions"]["train"] == {
        "start": "2020-02-17T00:00:00Z",
        "end": "2020-05-27T00:00:00Z",
        "calendar_days": 100,
    }
    assert split["partitions"]["test"]["end"] == "2020-08-02T00:00:00Z"


def test_context_never_changes_when_future_appended() -> None:
    ends = np.arange(
        np.datetime64("2020-02-17T00:15"),
        np.datetime64("2020-02-19T00:15"),
        np.timedelta64(15, "m"),
    )
    cutoff = np.datetime64("2020-02-18T00:00")
    selected = context_indices(ends, cutoff)
    assert len(selected) == 96
    assert selected[-1] == 95
    np.testing.assert_array_equal(selected, context_indices(ends[:96], cutoff))


def test_exact_cutoff_bin_belongs_only_to_context() -> None:
    ends = np.array(
        [
            "2020-02-18T00:00",
            "2020-02-18T00:15",
            "2020-02-18T00:30",
            "2020-02-18T00:45",
            "2020-02-18T01:00",
            "2020-02-18T01:15",
        ],
        dtype="datetime64[m]",
    )
    selected = target_indices(ends, ends[0], 1)
    assert selected.tolist() == [1, 2, 3, 4]


def test_unsupported_horizon_rejected() -> None:
    with pytest.raises(ValueError):
        target_indices(np.array([], dtype="datetime64[m]"), np.datetime64("2020-01-01"), 2)
