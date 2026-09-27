import pytest

from tools.v2_native_support import validate_support_review


def approval():
    index = {
        "data_kind": "REAL",
        "study_id": "v2_candidate2",
        "bindings": {"native_freeze": {"sha256": "f"}},
    }
    review = {
        "disposition": "APPROVE_NATIVE_SUPPORT_AUDIT",
        "reviewer_session": "/root/v2_reviewer",
        "driver_sha256": "d",
        "shared_audit_sha256": "s",
        "freeze_sha256": "f",
    }
    return index, review


def test_exact_support_review_and_real_index_required():
    index, review = approval()
    validate_support_review(index, review, "f", "d", "s")
    for field, value in (("reviewer_session", "/root"), ("driver_sha256", "wrong")):
        changed = review.copy()
        changed[field] = value
        with pytest.raises(ValueError, match="review"):
            validate_support_review(index, changed, "f", "d", "s")
    index["data_kind"] = "FIXTURE"
    with pytest.raises(ValueError, match="REAL"):
        validate_support_review(index, review, "f", "d", "s")


def test_index_freeze_mismatch_fails_before_support():
    index, review = approval()
    index["bindings"]["native_freeze"]["sha256"] = "other"
    with pytest.raises(ValueError, match="freeze"):
        validate_support_review(index, review, "f", "d", "s")


def test_native_denominator_boundary_and_future_censoring():
    from tools.v2_support_audit import audit_partition

    rows = [
        {
            "time": str(
                __import__("numpy").datetime64("2020-02-17T00")
                + __import__("numpy").timedelta64(i, "h")
            ),
            "observed": 240,
            "expected": 240,
            "detected": 2160.0,
            "observed_range_ping_m": 21600.0,
            "configuration": "one",
        }
        for i in range(32)
    ]
    original = audit_partition(rows)["rows"][24]
    assert original["issued"] is True
    assert original["horizons"]["1"]["fraction"] == 0.1
    assert original["horizons"]["1"]["index_eligible"] is True
    rows[24]["detected"] = 2159.0
    censored = audit_partition(rows)["rows"][24]
    assert censored["issued"] is True
    assert censored["horizons"]["1"]["index_eligible"] is False
    assert censored["horizons"]["1"]["fraction_eligible"] is True
