"""Native overlap weights define candidate-2 sampled targets without invented values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.v2_cohort import (
    prepare_native_development_cohort,
    verify_support_rows,
)
from marine_echo.training.v2_native import NativeObservationSlot, iter_native_hourly_windows


def _slots() -> list[NativeObservationSlot]:
    start = np.datetime64("2020-02-17T00:00")
    slots = []
    for index in range(120):
        linear = np.full((4, 64), np.nan)
        linear[0, 5:50] = (index + 1) * 1e-8
        weight = np.zeros((4, 64))
        weight[0, 5:50] = 40.0
        slots.append(
            NativeObservationSlot(
                start=start + index * np.timedelta64(15, "m"),
                end=start + (index + 1) * np.timedelta64(15, "m"),
                linear_sv=linear,
                detected_range_ping_m=weight,
                observed_pings=40,
                expected_pings=60,
                configuration_id="a" * 64,
                source_sha256=("b" * 64,),
                processed_sha256="c" * 64,
            )
        )
    return slots


def test_native_hourly_target_uses_range_length_weights() -> None:
    slots = _slots()
    original = next(iter_native_hourly_windows(slots, partition="train"))
    changed = list(slots)
    weight = changed[96].detected_range_ping_m.copy()
    weight[0, 5:50] = 4.0
    changed[96] = replace(changed[96], detected_range_ping_m=weight)
    row = next(iter_native_hourly_windows(changed, partition="train"))
    expected_linear = (97 * 4 + 98 * 40 + 99 * 40 + 100 * 40) / 124 * 1e-8
    assert row.row_id == original.row_id
    assert row.target_db[0] == 10 * np.log10(expected_linear)
    assert row.target_detection_fraction[0] == (124 * 45) / (90 * 160)
    assert row.target_mask.all()
    assert row.context_mask[:, 1:].sum() == 0
    assert row.future_train_db.shape == (3, 4, 4, 64)


def test_native_zero_detection_keeps_fraction_truth_and_missing_index() -> None:
    changed = _slots()
    for index in range(96, 100):
        weight = np.zeros((4, 64))
        changed[index] = replace(changed[index], detected_range_ping_m=weight)
    row = next(iter_native_hourly_windows(changed, partition="train"))
    assert not row.target_mask[0]
    assert np.isnan(row.target_db[0])
    assert row.target_detection_mask[0]
    assert row.target_detection_fraction[0] == 0
    assert row.target_mask[1:].all()
    assert row.target_interval_start[0] == row.cutoff


def test_native_past_index_uses_exact_range_lengths() -> None:
    changed = _slots()
    values = changed[95].linear_sv.copy()
    values[0, 5] = 1000e-8
    weight = changed[95].detected_range_ping_m.copy()
    weight[0, 5] = 4.0
    changed[95] = replace(changed[95], linear_sv=values, detected_range_ping_m=weight)
    row = next(iter_native_hourly_windows(changed, partition="train"))
    expected = ((1000 * 4) + (96 * 44 * 40)) / (4 + 44 * 40) * 1e-8
    assert row.context_index_db is not None
    assert row.context_index_db[-1] == 10 * np.log10(expected)
    assert row.past_source_sha256 == ("b" * 64,)
    assert row.target_source_sha256 == ("b" * 64,)


def test_future_configuration_must_match_issuance_configuration() -> None:
    changed = _slots()
    for index in range(96, 100):
        changed[index] = replace(changed[index], configuration_id="d" * 64)
    row = next(iter_native_hourly_windows(changed, partition="train"))
    assert not row.target_mask[0]
    assert not row.target_detection_mask[0]
    assert row.target_mask[1:].all()


def test_support_cohort_requires_exact_issued_cutoff_and_target_masks() -> None:
    row = next(iter_native_hourly_windows(_slots(), partition="train"))
    entry = {
        "cutoff": str(row.cutoff.astype("datetime64[h]")),
        "issued": True,
        "horizons": {
            str(h): {
                "target_start": str(row.target_interval_start[index].astype("datetime64[h]")),
                "index_eligible": bool(row.target_mask[index]),
                "fraction_eligible": bool(row.target_detection_mask[index]),
                "fraction": float(row.target_detection_fraction[index]),
            }
            for index, h in enumerate((1, 3, 6))
        },
    }
    verify_support_rows([row], [entry])
    entry["horizons"]["3"]["index_eligible"] = False
    with pytest.raises(ValueError, match="support row"):
        verify_support_rows([row], [entry])


def test_failed_support_stops_before_any_native_shard_read(tmp_path: Path) -> None:
    report = tmp_path / "evidence/v2/native-support/eligibility_v2.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({"status": "NATIVE_CANDIDATE_INELIGIBLE_STOP_D1_TARGET_SEARCH"}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="ineligible"):
        prepare_native_development_cohort(
            tmp_path,
            native_index_sha256="a" * 64,
            support_report_sha256=digest,
        )
