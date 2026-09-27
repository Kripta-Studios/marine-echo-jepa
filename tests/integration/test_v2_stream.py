"""Small artificial shards test the real-source reader and causal window boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.v2_stream import (
    CandidateTrainStream,
    WindowPlan,
    iter_hourly_windows,
    iter_windows,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, count: int = 8) -> tuple[Path, Path]:
    day = tmp_path / "2020-02-17.npz"
    starts = np.datetime64("2020-02-17T00:00") + np.arange(count) * np.timedelta64(15, "m")
    linear = np.full((count, 4, 64), 1e-8)
    linear[:, 0, 5:50] = np.arange(1, count + 1)[:, None] * 1e-8
    counts = np.full((count, 4, 64), 20, dtype=np.int64)
    np.savez_compressed(
        day,
        bin_start=starts,
        bin_end=starts + np.timedelta64(15, "m"),
        linear_sv=linear,
        valid_ping_count=counts,
        expected_ping_count=np.full(count, 60, dtype=np.int64),
        observed_ping_count=np.full(count, 40, dtype=np.int64),
        range_edges_m=np.arange(65, dtype=float) * 2,
        frequency_hz=np.array([38000, 125000, 200000, 455000]),
        configuration_boundary=np.zeros(count, dtype=bool),
    )
    report = tmp_path / "processing_manifest.json"
    report.write_text(
        json.dumps(
            {
                "status": "TRAIN_CANDIDATE_PROCESSED_REQUIRES_QC_REVIEW",
                "dataset_id": "mosaic_azfp_down_2020",
                "instrument_id": "AZFP55170",
                "daily_candidate_path": day.name,
                "daily_candidate_sha256": _digest(day),
                "source_hours": [{"source": "source.01A", "source_sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps(
            {
                "status": "COMPLETE_TRAIN_CENSUS_NOT_BENCHMARK",
                "identity": {"calendar": ["2020-02-17"]},
                "artifact_sha256": {
                    report.name: _digest(report),
                    day.name: _digest(day),
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, census


def test_train_reader_verifies_hashes_and_rejects_tampering(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path)
    reader = CandidateTrainStream(root, census, fixture_only=True)
    slots = list(reader)
    assert len(slots) == 8
    assert slots[0].source_sha256 == ("a" * 64,)
    assert slots[0].end == np.datetime64("2020-02-17T00:15")
    day = root / "2020-02-17.npz"
    day.write_bytes(day.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest"):
        list(CandidateTrainStream(root, census, fixture_only=True))


def test_window_uses_fixed_future_slot_and_past_only_context(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path)
    slots = list(CandidateTrainStream(root, census, fixture_only=True))
    plan = WindowPlan(
        context_slots=4, horizon_offsets=(1, 3), target_frequency_hz=38000, range_bins=(5, 50)
    )
    rows = list(iter_windows(slots, plan))
    first = rows[0]
    assert first.cutoff == slots[3].end
    assert first.context.shape == (4, 4, 64)
    assert np.allclose(first.target_db, 10 * np.log10(np.array([5e-8, 7e-8])))
    changed = list(slots)
    changed[4] = changed[4].with_linear_sv(np.full((4, 64), 9e-8))
    altered = next(iter_windows(changed, plan))
    assert np.array_equal(first.context, altered.context, equal_nan=True)
    assert altered.target_db[0] != first.target_db[0]
    assert altered.target_db[1] == first.target_db[1]


def test_missing_scheduled_target_is_masked_without_shifting(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path)
    slots = list(CandidateTrainStream(root, census, fixture_only=True))
    plan = WindowPlan(
        context_slots=4, horizon_offsets=(1, 3), target_frequency_hz=38000, range_bins=(5, 50)
    )
    changed = list(slots)
    changed[4] = changed[4].with_linear_sv(np.full((4, 64), np.nan))
    first = next(iter_windows(changed, plan))
    assert not first.target_mask[0]
    assert np.isnan(first.target_db[0])
    assert first.target_mask[1]
    assert first.target_time[0] == slots[4].end


def test_reader_requires_verified_source_hashes(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path)
    report = root / "processing_manifest.json"
    document = json.loads(report.read_text(encoding="utf-8"))
    document["source_hours"][0]["source_sha256"] = ""
    report.write_text(json.dumps(document), encoding="utf-8")
    census_doc = json.loads(census.read_text(encoding="utf-8"))
    census_doc["artifact_sha256"][report.name] = _digest(report)
    census.write_text(json.dumps(census_doc), encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA-256"):
        list(CandidateTrainStream(root, census, fixture_only=True))


def test_hourly_conditional_target_and_fixed_horizons(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path, count=124)
    slots = CandidateTrainStream(root, census, fixture_only=True)
    rows = list(iter_hourly_windows(slots, partition="train"))
    first = rows[0]
    assert first.context.shape == (96, 4, 64)
    assert first.context_mask[:, 1:].sum() == 0
    assert first.target_db.shape == (3,)
    assert np.allclose(first.target_db, 10 * np.log10(np.array([98.5, 106.5, 118.5]) * 1e-8))
    assert np.allclose(first.target_detection_fraction, 0.5)
    assert np.allclose(first.target_acquisition_fraction, 2 / 3)
    assert first.future_train_db.shape == (3, 4, 4, 64)
    assert first.cutoff == np.datetime64("2020-02-18T00:00")
    assert len(rows) == 2


def test_hourly_target_missingness_does_not_move_to_next_valid_hour(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path, count=120)
    slots = list(CandidateTrainStream(root, census, fixture_only=True))
    changed = list(slots)
    for index in range(96, 100):
        changed[index] = changed[index].with_linear_sv(np.full((4, 64), np.nan))
    first = next(iter_hourly_windows(changed, partition="train"))
    assert not first.target_mask[0]
    assert np.isnan(first.target_db[0])
    assert first.target_mask[1]
    assert first.target_interval_start[0] == np.datetime64("2020-02-18T00:00")
    assert first.target_interval_end[0] == np.datetime64("2020-02-18T01:00")


def test_hourly_target_weights_valid_pings_and_keeps_issued_row(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path, count=120)
    slots = list(CandidateTrainStream(root, census, fixture_only=True))
    original = next(iter_hourly_windows(slots, partition="train"))
    changed = list(slots)
    counts = changed[96].valid_ping_count.copy()
    counts[0, 5:50] = 2
    changed[96] = replace(changed[96], valid_ping_count=counts)
    weighted = next(iter_hourly_windows(changed, partition="train"))
    expected = (97 * 2 + 98 * 20 + 99 * 20 + 100 * 20) / 62 * 1e-8
    assert weighted.row_id == original.row_id
    assert weighted.target_db[0] == pytest.approx(10 * np.log10(expected))
    assert weighted.target_detection_fraction[0] == pytest.approx(62 / 160)
    assert weighted.target_db[1] == original.target_db[1]


def test_hourly_reader_rejects_protected_partition(tmp_path: Path) -> None:
    root, census = _fixture(tmp_path, count=120)
    with pytest.raises(ValueError, match="TRAIN development"):
        next(
            iter_hourly_windows(
                CandidateTrainStream(root, census, fixture_only=True), partition="test"
            )
        )
