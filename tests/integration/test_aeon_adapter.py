"""Synthetic AEON FullDepth contracts; published outcomes are not opened."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from marine_echo.training.aeon_corpus import AeonDevelopmentReader
from marine_echo.training.aeon_windows import AeonWindowPlan, iter_aeon_windows


def _fixture(
    tmp_path: Path,
    *,
    hours: int = 35,
    anomalies: dict[tuple[int, int], str] | None = None,
) -> tuple[Path, Path]:
    anomalies = anomalies or {}
    archive = tmp_path / "AEON3_fixture.zip"
    columns = [
        "Process_ID", "Interval", "Layer", "Sv_mean", "NASC", "Height_mean",
        "Depth_mean", "Layer_depth_min", "Layer_depth_max", "Ping_S", "Ping_E",
        "Dist_M", "Date_M", "Time_M", "Lat_M", "Lon_M", "Noise_Sv_1m",
        "Minimum_Sv_threshold_applied", "Maximum_Sv_threshold_applied",
        "Standard_deviation", "Thickness_mean", "Range_mean",
        "Exclude_below_line_range_mean", "Exclude_above_line_range_mean",
    ]
    members = []
    first = datetime(2024, 3, 6, 0, 52)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for frequency in (38, 125, 200, 455):
            member = f"AEON3_fixture/AEON3_55144_{frequency:03d}_2024_03_60minFullDepth.csv"
            members.append(member)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for hour in range(hours):
                anomaly = anomalies.get((frequency, hour))
                if anomaly == "missing":
                    continue
                moment = first + timedelta(hours=hour)
                row = {
                    "Process_ID": 739, "Interval": 474930 + hour, "Layer": 1,
                    "Sv_mean": -80 + hour / 10 + frequency / 1000,
                    "Layer_depth_min": 0, "Layer_depth_max": 200,
                    "Ping_S": 150 * hour + 1, "Ping_E": 150 * (hour + 1),
                    "Date_M": moment.strftime("%Y%m%d"),
                    "Time_M": moment.strftime(" %H:%M:%S.%f"),
                }
                if anomaly == "sentinel":
                    row["Sv_mean"] = -9999
                elif anomaly == "partial":
                    row["Ping_E"] = 150 * (hour + 1) - 1
                elif anomaly == "geometry":
                    row["Layer_depth_max"] = 199
                elif anomaly == "outside_train_nonnumeric":
                    row["Date_M"] = "20240305"
                    row["Sv_mean"] = "this_would_fail_numeric_conversion"
                elif anomaly == "time_mismatch":
                    row["Time_M"] = (moment + timedelta(minutes=20)).strftime(" %H:%M:%S.%f")
                writer.writerow(row)
                if anomaly == "duplicate":
                    writer.writerow(row)
            zf.writestr(member, stream.getvalue())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "APPROVED_AEON_DEVELOPMENT",
                "data_kind": "SYNTHETIC_FIXTURE",
                "archive_sha256": digest,
                "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED",
                "permitted_members": members,
                "permitted_months": ["2024-03"],
                "development_start": "2024-03-01T00:00:00",
                "development_end_exclusive": "2024-04-01T00:00:00",
            }
        ),
        encoding="utf-8",
    )
    return archive, review


def _reader(archive: Path, review: Path) -> AeonDevelopmentReader:
    return AeonDevelopmentReader(
        archive,
        review_path=review,
        review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
        fixture_only=True,
    )


def _windows(slots: list, partition: str = "development_fit") -> list:
    return list(
        iter_aeon_windows(
            slots,
            plan=AeonWindowPlan(),
            partition=partition,
            partition_start="2024-03-06T00:00:00",
            partition_end_exclusive="2024-03-20T00:00:00",
        )
    )


def test_reader_hashes_archive_and_preserves_four_full_depth_channels(tmp_path: Path) -> None:
    archive, review = _fixture(tmp_path)
    with pytest.raises(ValueError, match="review digest"):
        AeonDevelopmentReader(
            archive, review_path=review, review_sha256="0" * 64, fixture_only=True
        )
    slots = list(_reader(archive, review))
    assert len(slots) == 35
    assert slots[0].source_timestamp == np.datetime64("2024-03-06T00:52")
    assert slots[0].interval_id == 474930
    assert slots[0].sv_db.shape == (4,)
    assert slots[0].observed_mask.all()
    assert len(slots[0].member_names) == 4
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="archive digest"):
        list(_reader(archive, review))


def test_reader_distinguishes_sentinel_partial_duplicate_and_missing(tmp_path: Path) -> None:
    archive, review = _fixture(
        tmp_path,
        anomalies={(38, 24): "sentinel", (125, 25): "partial", (200, 26): "duplicate", (455, 27): "missing", (125, 28): "time_mismatch"},
    )
    slots = list(_reader(archive, review))
    assert slots[24].qc_status[0] == "INVALID_OR_SPECIAL_SV"
    assert slots[25].qc_status[1] == "PARTIAL_SOURCE_INTERVAL"
    assert slots[26].qc_status[2] == "DUPLICATE_ROW"
    assert slots[27].qc_status[3] == "MISSING_ROW"
    assert slots[28].qc_status[1] == "SOURCE_TIME_MISMATCH"
    assert not slots[24].observed_mask[0]
    assert not slots[25].observed_mask[1]


def test_partition_filter_precedes_numeric_outcome_conversion(tmp_path: Path) -> None:
    archive, review = _fixture(
        tmp_path,
        anomalies={(frequency, 0): "outside_train_nonnumeric" for frequency in (38, 125, 200, 455)},
    )
    # This exercises the same source-date gate used by real TRAIN loading.
    slots = list(_reader(archive, review)._iter_partition("train"))
    assert len(slots) == 34
    assert slots[0].interval_id == 474931


def test_windows_use_past_only_and_keep_missing_future_targets(tmp_path: Path) -> None:
    archive, review = _fixture(tmp_path, anomalies={(38, 24): "missing", (125, 10): "partial"})
    slots = list(_reader(archive, review))
    rows = _windows(slots)
    first = rows[0]
    assert first.context_db.shape == (24, 4)
    assert first.context_mask[:, 0].all()
    assert not first.context_mask[10, 1]
    assert np.array_equal(first.target_interval_ids, [474954, 474956, 474959])
    assert not first.target_mask[0]
    assert first.target_qc_status[0] == "MISSING_ROW"
    assert len(rows) >= 1
    changed = list(slots)
    changed[26] = changed[26].with_sv_db(np.array([-10.0, -20.0, -30.0, -40.0]))
    shifted = _windows(changed)[0]
    assert shifted.row_id == first.row_id
    assert np.array_equal(shifted.context_db, first.context_db, equal_nan=True)
    assert shifted.target_db[1] != first.target_db[1]


def test_windows_reject_gapped_past_and_heldout_partition(tmp_path: Path) -> None:
    archive, review = _fixture(tmp_path)
    slots = list(_reader(archive, review))
    assert len(_windows(slots[:10] + slots[11:])) < len(_windows(slots))
    with pytest.raises(ValueError, match="development"):
        _windows(slots, partition="test")


def test_reviewed_development_runs_baseline_and_direct_with_checkpoint(tmp_path: Path) -> None:
    from marine_echo.training.aeon_development import execute_aeon_development

    archive, review = _fixture(
        tmp_path,
        hours=130,
        anomalies={(frequency, 69): "missing" for frequency in (38, 125, 200, 455)},
    )
    slots = list(_reader(archive, review))
    fit = _windows(slots, "development_fit")[:15]
    assess = _windows(slots, "development_assessment")[45:50]
    approved = json.loads(review.read_text(encoding="utf-8"))
    approved.update(
        {
            "window_plan": {"context_hours": 24, "horizons_hours": [1, 3, 6]},
            "development_fit": {
                "start": "2024-03-06T00:00:00", "end_exclusive": "2024-03-07T20:00:00"
            },
            "development_assessment": {
                "start": "2024-03-07T21:00:00", "end_exclusive": "2024-03-11T00:00:00"
            },
            "direct_updates": 8,
            "direct_seed": 7,
            "direct_batch_size": 4,
        }
    )
    review.write_text(json.dumps(approved), encoding="utf-8")
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    output = tmp_path / "synthetic-development"
    result = execute_aeon_development(
        fit, assess, output, plan=AeonWindowPlan(), review_path=review,
        review_sha256=digest, updates=8, batch_size=4, device="cpu", fixture_only=True,
    )
    assert result["status"] == "COMPLETED_SYNTHETIC_FIXTURE"
    assert result["fit_rows"] == 15
    assert result["assessment_rows"] == 5
    assert Path(result["direct"]["checkpoint"]).exists()
    assert Path(result["baseline"]["predictions"]).exists()
    assert result["baseline"]["metrics"]["horizons"][0]["scored_rows"] == 4
