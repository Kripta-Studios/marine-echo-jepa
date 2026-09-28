"""Synthetic-only AEON transfer evaluator tests; no publisher outcomes opened."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from marine_echo.training import aeon_external_evaluator
from marine_echo.training.aeon_corpus import AeonHourlySlot
from marine_echo.training.aeon_external_evaluator import (
    adapter_compatible_rows,
    materialize_candidates,
    score_external_predictions,
)


def _slot(interval: int, *, valid_38: bool = True, valid_125: bool = True) -> AeonHourlySlot:
    values = np.asarray([-70.0 + interval * 0.01, -68.0, -66.0, -64.0], dtype=np.float64)
    mask = np.asarray([valid_38, valid_125, True, True], dtype=bool)
    values[~mask] = np.nan
    return AeonHourlySlot(
        interval_id=interval,
        source_timestamp=np.datetime64("2022-01-01T00:00:00", "us")
        + np.timedelta64(interval - 1, "h"),
        sv_db=values,
        observed_mask=mask,
        qc_status=tuple("OBSERVED_SOURCE_PRODUCT" if item else "INVALID_OR_SPECIAL_SV"
                        for item in mask),
        member_names=("synthetic.csv",),
        archive_sha256="a" * 64,
    )


def _candidate(cutoff: int) -> dict[str, object]:
    stamp = np.datetime64("2022-01-01T00:00:00", "us") + np.timedelta64(cutoff - 1, "h")
    return {
        "cutoff_interval_id": cutoff,
        "cutoff_source_timestamp": str(stamp),
        "cutoff_source_date": str(stamp.astype("datetime64[D]")),
        "row_id": hashlib.sha256(
            f"aeon-external-full-depth:{'a' * 64}:{cutoff}:24:1,3,6".encode("ascii")
        ).hexdigest(),
        "target_interval_ids": [cutoff + 1, cutoff + 3, cutoff + 6],
        "actual_issued_status": "UNKNOWN",
        "target_scoring_status": "UNKNOWN",
    }


def test_only_stage1_candidates_issue_with_past_only_qc_and_external_row_identity() -> None:
    slots = [_slot(interval, valid_38=interval != 1, valid_125=interval != 25)
             for interval in range(1, 34)]
    # Future 31 is missing but must not affect issuance at cutoff 25.
    slots = [slot for slot in slots if slot.interval_id != 31]
    candidates = [_candidate(24), _candidate(25), _candidate(26)]
    windows, not_issued = materialize_candidates(slots, candidates, "a" * 64)
    assert [row.cutoff_interval_id for row in windows] == [25, 26]
    assert list(not_issued) == [candidates[0]["row_id"]]
    assert "PAST_38_INVALID_OR_SPECIAL_SV" in not_issued[candidates[0]["row_id"]]
    assert windows[0].row_id == candidates[1]["row_id"]
    assert windows[0].partition == "external_transfer"
    adapter_rows = adapter_compatible_rows(windows)
    assert adapter_rows[0].partition == "test"
    assert adapter_rows[0].row_id == windows[0].row_id
    assert windows[0].partition == "external_transfer"
    assert not windows[0].context_mask[-1, 1]
    assert not windows[0].target_mask[2]
    assert windows[0].target_qc_status[2] == "MISSING_INTERVAL"


def test_candidate_identity_tamper_fails_closed() -> None:
    slots = [_slot(interval) for interval in range(1, 34)]
    candidate = _candidate(24)
    candidate["target_interval_ids"] = [25, 27, 31]
    with pytest.raises(ValueError, match="candidate identity"):
        materialize_candidates(slots, [candidate], "a" * 64)


def test_empty_candidate_universe_is_ineligible_without_issuance() -> None:
    windows, reasons = materialize_candidates([_slot(1)], [], "a" * 64)
    assert windows == []
    assert reasons == {}


def test_scoring_requires_identical_support_and_90_dates_before_transfer_claim() -> None:
    count = 18 * 2
    truth = np.zeros((count, 3), dtype=np.float64)
    observed = np.ones((count, 3), dtype=bool)
    times = np.empty((count, 3), dtype="datetime64[us]")
    for index in range(count):
        times[index] = np.datetime64("2022-01-01", "us") + np.timedelta64(index // 18, "D")
    direct = np.zeros((count, 3, 5), dtype=np.float64)
    ema = direct.copy()
    result = score_external_predictions(truth, observed, times, direct, ema)
    assert result["eligible_dates_per_horizon"] == [2, 2, 2]
    assert result["cohort_eligible"] is False
    assert result["jepa_value_gate"] == "INELIGIBLE_NOT_A_NEGATIVE_TRANSFER_RESULT"


def test_scoring_preserves_zero_eligible_horizon_as_ineligible() -> None:
    truth = np.zeros((2, 3), dtype=np.float64)
    observed = np.asarray([[True, False, False], [True, False, False]], dtype=bool)
    times = np.full((2, 3), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    times[:, 0] = np.datetime64("2022-01-01", "us")
    predictions = np.zeros((2, 3, 5), dtype=np.float64)
    result = score_external_predictions(truth, observed, times, predictions, predictions)
    assert result["eligible_dates_per_horizon"] == [0, 0, 0]
    assert result["jepa_value_gate"] == "INELIGIBLE_NOT_A_NEGATIVE_TRANSFER_RESULT"


def test_eligible_executed_transfer_uses_new_seed_paired_two_date_gate() -> None:
    count = 90 * 18
    truth = np.zeros((count, 3), dtype=np.float64)
    observed = np.ones((count, 3), dtype=bool)
    times = np.empty((count, 3), dtype="datetime64[us]")
    for index in range(count):
        times[index] = np.datetime64("2022-01-01", "us") + np.timedelta64(index // 18, "D")
    direct = np.ones((count, 3, 5), dtype=np.float64)
    ema = np.full((count, 3, 5), 0.9, dtype=np.float64)
    result = score_external_predictions(truth, observed, times, direct, ema)
    assert result["eligible_dates_per_horizon"] == [90, 90, 90]
    assert result["cohort_eligible"] is True
    assert result["paired_bootstrap_status"] == "COMPLETED_2000_DRAWS_SEED_20260928"
    assert result["jepa_value_gate"] == "PASSED_RETROSPECTIVE_EXTERNAL_TRANSFER"
    assert result["paired_95pct_difference_interval_db"][1] < 0


def test_uneligible_leading_day_does_not_shift_paired_two_date_blocks() -> None:
    count = 18 * 90
    truth = np.zeros((count, 3), dtype=np.float64)
    observed = np.ones((count, 3), dtype=bool)
    times = np.empty((count, 3), dtype="datetime64[us]")
    ema = np.empty((count, 3, 5), dtype=np.float64)
    for index in range(count):
        day = index // 18
        times[index] = np.datetime64("2022-01-01", "us") + np.timedelta64(day, "D")
        ema[index] = 0.7 if day % 3 else 0.9
    direct = np.ones((count, 3, 5), dtype=np.float64)
    reference = aeon_external_evaluator._paired_two_date_bootstrap(
        truth, observed, times, direct, ema
    )
    leading = np.datetime64("2021-12-31", "us")
    with_uneligible_day = aeon_external_evaluator._paired_two_date_bootstrap(
        np.concatenate([np.zeros((1, 3)), truth]),
        np.concatenate([np.ones((1, 3), dtype=bool), observed]),
        np.concatenate([np.full((1, 3), leading, dtype="datetime64[us]"), times]),
        np.concatenate([np.ones((1, 3, 5)), direct]),
        np.concatenate([np.full((1, 3, 5), 0.5), ema]),
    )
    assert np.array_equal(with_uneligible_day, reference)
