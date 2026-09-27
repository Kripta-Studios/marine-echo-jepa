"""Support audit never opens an incomplete or unreviewed raw corpus."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools.v2_raw_response_support import (
    REFERENCE_CONFIG,
    ROOT,
    RUN_ID,
    load_count_hours,
    sha,
    validate_day_counts,
    validate_review,
)


def review_fields() -> tuple[dict, dict]:
    contract_sha = sha(ROOT / "docs/adr/0007-raw-instrument-response-development.md")
    index = json.loads((ROOT / "evidence/v2/raw-response-development-reviewed/index.json").read_text())
    review = {
        "disposition": "APPROVE_RAW_RESPONSE_SUPPORT_METHOD",
        "reviewer_session": "/root/v2_reviewer",
        "driver_sha256": sha(ROOT / "tools/v2_raw_response_support.py"),
        "audit_sha256": sha(ROOT / "src/marine_echo/data/raw_response_support.py"),
        "contract_sha256": contract_sha,
        "processing_run_id": RUN_ID,
        "reference_configuration": REFERENCE_CONFIG,
        "index_sha256": sha(ROOT / "evidence/v2/raw-response-development-reviewed/index.json"),
        "bindings": index["bindings"],
    }
    return review, index


def test_review_rejects_fixture_wrong_code_or_run_id() -> None:
    review, index = review_fields()
    validate_review(review, index)
    with pytest.raises(ValueError):
        validate_review({**review, "driver_sha256": "0" * 64}, index)
    with pytest.raises(ValueError):
        validate_review({**review, "index_sha256": "0" * 64}, index)
    with pytest.raises(ValueError):
        validate_review(review, {**index, "data_kind": "FIXTURE"})
    with pytest.raises(ValueError):
        validate_review(review, {**index, "processing_run_id": "pilot"})
    for key in index["bindings"]:
        changed = {**index, "bindings": {**index["bindings"], key: "0" * 64}}
        with pytest.raises(ValueError):
            validate_review(review, changed)


def test_incomplete_index_stops_before_shard_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.v2_raw_response_support as driver

    monkeypatch.setattr(driver, "INDEX", tmp_path / "index.json")
    _, index = review_fields()
    index["days"] = {"2020-02-17": {"manifest_path": "missing.json"}}
    with pytest.raises(ValueError, match="58-day"):
        load_count_hours(index)


def test_timestamp_histogram_day_bounds_and_manifest_counts() -> None:
    day = "2020-02-17"
    first = np.datetime64(day, "ns")
    observed = np.zeros(96, dtype=np.int64)
    observed[0] = 2
    valid = observed.copy()
    zeros = np.zeros(96, dtype=np.int64)
    expected = np.full(96, 60, dtype=np.int64)
    starts = first + np.arange(96) * np.timedelta64(15, "m")
    times = first + np.array([1, 2]) * np.timedelta64(1, "s")
    document = {
        "observed_pings": 2, "valid_target_pings": 2,
        "zero_or_undefined_pings": 0, "zero_affected_pings": 0,
        "zero_affected_quarter_hours": 0, "nonfinite_pings": 0,
        "zero_or_undefined_samples": 0, "nonfinite_samples": 0,
    }

    def check(*, used_times: np.ndarray = times, used_expected: np.ndarray = expected, used_doc: dict = document) -> None:
        validate_day_counts(
            day, used_doc, observed, valid, zeros, zeros, zeros, zeros, zeros,
            used_expected, starts, used_times, valid * 180,
        )

    check()
    with pytest.raises(ValueError):
        check(used_times=np.array([first - np.timedelta64(1, "s"), first + np.timedelta64(2, "s")]))
    with pytest.raises(ValueError):
        check(used_times=first + np.array([901, 902]) * np.timedelta64(1, "s"))
    bad_expected = expected.copy()
    bad_expected[0] = 59
    with pytest.raises(ValueError):
        check(used_expected=bad_expected)
    with pytest.raises(ValueError):
        check(used_doc={**document, "valid_target_pings": 1})
