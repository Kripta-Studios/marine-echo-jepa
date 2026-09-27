"""Synthetic-only rehearsal of review-gated AEON CAL/TEST access."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from marine_echo.training.aeon_evaluation_reader import AeonEvaluationReader


def _fixture(tmp_path: Path, partition: str) -> tuple[Path, Path]:
    first = datetime(2024, 12, 1, 0, 52) if partition == "calibration" else datetime(2025, 1, 6, 0, 52)
    outside = first - timedelta(hours=1)
    archive = tmp_path / "evaluation_fixture.zip"
    members = []
    columns = ["Interval", "Layer", "Sv_mean", "Layer_depth_min", "Layer_depth_max",
               "Ping_S", "Ping_E", "Date_M", "Time_M"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for frequency in (38, 125, 200, 455):
            member = (f"AEON3_fixture/AEON3_55144_{frequency:03d}_"
                      f"{first:%Y_%m}_60minFullDepth.csv")
            members.append(member)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, moment in [(-1, outside)] + [
                (hour, first + timedelta(hours=hour)) for hour in range(38)
            ]:
                writer.writerow({
                    "Interval": 500000 + index, "Layer": 1,
                    "Sv_mean": "unapproved_numeric_poison" if index == -1 else (
                        -9999 if index == 25 and frequency == 38 else -80 + frequency / 1000
                    ),
                    "Layer_depth_min": 0, "Layer_depth_max": 200,
                    "Ping_S": 150 * (index + 2) + 1,
                    "Ping_E": 150 * (index + 3),
                    "Date_M": moment.strftime("%Y%m%d"),
                    "Time_M": moment.strftime(" %H:%M:%S.%f"),
                })
            zf.writestr(member, stream.getvalue())
    from marine_echo.training import aeon_evaluation_reader

    review = tmp_path / "access-review.json"
    review.write_text(json.dumps({
        "status": "APPROVED_AEON_EVALUATION_READER_FIXTURE",
        "data_kind": "SYNTHETIC_FIXTURE", "partition": partition,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "reader_code_sha256": hashlib.sha256(
            Path(aeon_evaluation_reader.__file__).read_bytes()
        ).hexdigest(),
        "permitted_members": members,
        "start": first.strftime("%Y-%m-%dT00:00:00"),
        "end_exclusive": (first + timedelta(days=3)).strftime("%Y-%m-%dT00:00:00"),
        "test_access": "FIXTURE_ONLY",
    }), encoding="utf-8")
    return archive, review


@pytest.mark.parametrize("partition", ["calibration", "test"])
def test_evaluation_reader_filters_before_numeric_and_preserves_qc(
    tmp_path: Path, partition: str,
) -> None:
    archive, review = _fixture(tmp_path, partition)
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    reader = AeonEvaluationReader(
        archive, review_path=review, review_sha256=digest,
        partition=partition, fixture_only=True,
    )
    slots = list(reader.iter_slots())
    assert len(slots) == 38
    assert slots[0].interval_id == 500000
    assert slots[25].qc_status[0] == "INVALID_OR_SPECIAL_SV"
    rows = list(reader.iter_windows())
    assert rows
    assert all(row.partition == partition for row in rows)
    assert all(row.context_mask[:, 0].all() for row in rows)
    assert any(not row.target_mask[0] for row in rows)
    assert any("INVALID_OR_SPECIAL_SV" in row.target_qc_status for row in rows)
    with pytest.raises(ValueError, match="review digest"):
        AeonEvaluationReader(archive, review_path=review, review_sha256="0" * 64,
                             partition=partition, fixture_only=True)


def test_evaluation_reader_rejects_wrong_partition_approval(tmp_path: Path) -> None:
    archive, review = _fixture(tmp_path, "calibration")
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="partition"):
        AeonEvaluationReader(archive, review_path=review, review_sha256=digest,
                             partition="test", fixture_only=True)
    with pytest.raises(ValueError, match="exact reviewed split"):
        AeonEvaluationReader(archive, review_path=review, review_sha256=digest,
                             partition="calibration", fixture_only=False)


def test_evaluation_reader_rechecks_source_hash_before_numeric_access(tmp_path: Path) -> None:
    archive, review = _fixture(tmp_path, "test")
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    reader = AeonEvaluationReader(archive, review_path=review, review_sha256=digest,
                                  partition="test", fixture_only=True)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="changed after access gate"):
        list(reader.iter_slots())
