"""Raw-code support is a past-only issuance and future-only scoring audit."""

from datetime import UTC, datetime, timedelta

from marine_echo.data.raw_response_support import audit_raw_partition


def hours(count: int, start: datetime) -> list[dict]:
    return [
        {
            "time": (start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00:00Z"),
            "observed": 240,
            "valid": 240,
            "zero_affected_pings": 0,
            "zero_samples": 0,
            "nonfinite_pings": 0,
            "nonfinite_samples": 0,
            "configuration": "A",
        }
        for i in range(count)
    ]


def test_future_configuration_and_counts_cannot_change_issuance() -> None:
    base = hours(72, datetime(2020, 4, 1, tzinfo=UTC))
    original = audit_raw_partition(base, reference_configuration="A")
    changed = [dict(row) for row in base]
    changed[24]["configuration"] = "B"
    changed[24]["valid"] = 0
    changed[24]["zero_affected_pings"] = 240
    altered = audit_raw_partition(changed, reference_configuration="A")
    assert original["rows"][24]["issued"]
    assert altered["rows"][24]["issued"]
    assert altered["rows"][24]["horizons"]["1"]["reason"] == "target_configuration"


def test_exact_valid_fraction_boundary_and_target_time_blocks() -> None:
    data = hours(96, datetime(2020, 4, 1, tzinfo=UTC))
    data[24]["observed"] = 200
    data[24]["valid"] = 190
    data[24]["zero_affected_pings"] = 10
    audit = audit_raw_partition(data, reference_configuration="A")
    target = audit["rows"][24]["horizons"]["1"]
    assert target["valid_fraction"] == 0.95
    assert target["eligible"]
    assert target["block_48h"] == 0
    data[24]["valid"] = 189
    data[24]["zero_affected_pings"] = 11
    failed = audit_raw_partition(data, reference_configuration="A")
    assert failed["rows"][24]["horizons"]["1"]["reason"] == "below_valid_fraction_floor"


def test_missing_target_is_not_a_configuration_mismatch() -> None:
    data = hours(72, datetime(2020, 4, 1, tzinfo=UTC))
    data[24]["observed"] = 0
    data[24]["valid"] = 0
    data[24]["configuration"] = None
    audit = audit_raw_partition(data, reference_configuration="A")
    assert audit["rows"][24]["issued"]
    assert audit["rows"][24]["horizons"]["1"]["reason"] == "missing_target"
