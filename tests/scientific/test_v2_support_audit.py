import numpy as np
import pytest

from tools.v2_support_audit import audit_partition


def records():
    return [
        {
            "time": str(np.datetime64("2020-02-17T00") + np.timedelta64(i, "h")),
            "observed": 240,
            "expected": 240,
            "detected": 1080,
            "configuration": "one",
        }
        for i in range(72)
    ]


def test_future_censoring_cannot_change_issuance():
    rows = records()
    before = audit_partition(rows)
    rows[24]["detected"] = 0
    after = audit_partition(rows)
    a = next(x for x in before["rows"] if x["cutoff"] == rows[24]["time"])
    b = next(x for x in after["rows"] if x["cutoff"] == rows[24]["time"])
    assert a["issued"] is b["issued"] is True
    assert a["horizons"]["1"]["index_eligible"] is True
    assert b["horizons"]["1"]["index_eligible"] is False
    assert b["horizons"]["1"]["fraction_eligible"] is True
    assert b["horizons"]["1"]["fraction"] == 0


def test_half_open_support_and_boundary_exclusions():
    result = audit_partition(records())
    issued = [row for row in result["rows"] if row["issued"]]
    assert len(issued) == 43
    assert issued[0]["cutoff"] == "2020-02-18T00"
    assert issued[-1]["cutoff"] == "2020-02-19T18"
    assert result["issue_exclusions"]["partition_boundary"] == 29


def test_missing_and_censored_are_distinct():
    rows = records()
    rows[24]["observed"] = 0
    rows[24]["detected"] = 0
    rows[24]["configuration"] = None
    target = audit_partition(rows)["rows"][24]["horizons"]["1"]
    assert target["reason"] == "insufficient_acquisition"
    assert target["fraction"] is None
    assert target["fraction_eligible"] is False


@pytest.mark.parametrize("mode", ["gap", "reorder"])
def test_wall_clock_gaps_and_reordering_fail(mode):
    rows = records()
    if mode == "gap":
        del rows[30]
    else:
        rows[30], rows[31] = rows[31], rows[30]
    with pytest.raises(ValueError, match="contiguous"):
        audit_partition(rows)


def test_future_configuration_does_not_control_issuance():
    rows = records()
    rows[24]["configuration"] = "changed"
    row = audit_partition(rows)["rows"][24]
    assert row["issued"] is True
    assert row["horizons"]["1"]["reason"] == "target_configuration"
