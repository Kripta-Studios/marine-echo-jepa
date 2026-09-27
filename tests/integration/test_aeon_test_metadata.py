"""Synthetic-only AEON retrospective TEST candidate-universe scanner tests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from marine_echo.training.aeon_test_metadata import _scanner_code_sha256, scan_test_metadata


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = tmp_path / "metadata-fixture.zip"
    base = datetime(2025, 1, 6, 0, 52)
    columns = ["Interval", "Layer", "Layer_depth_min", "Layer_depth_max",
               "Ping_S", "Ping_E", "Date_M", "Time_M", "Sv_mean"]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for month in (1, 2):
            for frequency in (38, 125, 200, 455):
                member = (f"AEON3_fixture/AEON3_55144_{frequency:03d}_"
                          f"2025_{month:02d}_60minFullDepth.csv")
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=columns)
                writer.writeheader()
                if month == 1:
                    writer.writerow({
                        "Interval": "NOT_AN_INTERVAL", "Layer": 1,
                        "Layer_depth_min": 0, "Layer_depth_max": 200,
                        "Ping_S": 1, "Ping_E": 150,
                        "Date_M": "20250105", "Time_M": " 23:52:00.000000",
                        "Sv_mean": "DO_NOT_PARSE_OUTSIDE",
                    })
                    for index in range(40):
                        if index == 12 and frequency == 125:
                            continue
                        moment = base + timedelta(hours=index)
                        row = {
                            "Interval": 500000 + index, "Layer": 1,
                            "Layer_depth_min": 0,
                            "Layer_depth_max": 199 if index == 5 and frequency == 38 else 200,
                            "Ping_S": 150 * index + 1,
                            "Ping_E": 150 * (index + 1) - (1 if index == 14 and frequency == 455 else 0),
                            "Date_M": moment.strftime("%Y%m%d"),
                            "Time_M": (moment + timedelta(minutes=20) if index == 13 and frequency == 200
                                       else moment).strftime(" %H:%M:%S.%f"),
                            "Sv_mean": "DO_NOT_PARSE_NONNUMERIC",
                        }
                        writer.writerow(row)
                        if index == 30 and frequency == 38:
                            writer.writerow(row)
                zf.writestr(member, buffer.getvalue())
    source_config = Path(__file__).resolve().parents[2] / "configs/aeon_test_metadata.json"
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["status"] = "SYNTHETIC_FIXTURE_ONLY"
    config["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    config_path = tmp_path / "fixture-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    review_path = tmp_path / "fixture-review.json"
    review_path.write_text(json.dumps({
        "status": "APPROVED_AEON_TEST_METADATA_FIXTURE",
        "scanner_code_sha256": _scanner_code_sha256(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_archive_sha256": config["source_archive_sha256"],
        "numeric_test_outcome_access": "PROHIBITED",
    }), encoding="utf-8")
    return archive, config_path, review_path


def test_metadata_candidate_scanner_never_uses_poison_sv_and_preserves_reasons(
    tmp_path: Path,
) -> None:
    archive, config, review = _fixture(tmp_path)
    output = tmp_path / "result"
    report = scan_test_metadata(archive, config, review, output, fixture_only=True)
    assert report["classification"] == "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
    assert report["interval_count"] == 40
    assert report["candidate_cutoff_interval_ids"] == [500029]
    assert report["actual_issued_rows"] == "UNKNOWN_NUMERIC_QC_NOT_OPENED"
    assert report["actual_scored_rows"] == "UNKNOWN_NUMERIC_QC_NOT_OPENED"
    intervals = report["interval_metadata"]
    assert intervals["500005"]["channels"]["38000"] == "INVALID_GEOMETRY"
    assert intervals["500030"]["channels"]["38000"] == "DUPLICATE_ROW"
    assert intervals["500012"]["channels"]["125000"] == "MISSING_ROW"
    assert intervals["500013"]["channels"]["200000"] == "SOURCE_TIME_MISMATCH"
    assert intervals["500014"]["channels"]["455000"] == "PARTIAL_SOURCE_INTERVAL"
    assert "DO_NOT_PARSE" not in (output / "metadata-candidates.json").read_text(encoding="utf-8")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected_row = hashlib.sha256(
        f"aeon-full-depth:{archive_sha}:500029:24:1,3,6".encode("ascii")
    ).hexdigest()
    expected_hash = hashlib.sha256(
        f"aeon-test-candidates-v1\n500029|{expected_row}|500030,500032,500035\n".encode("ascii")
    ).hexdigest()
    assert report["candidate_sha256"] == expected_hash
    assert report["candidate_rows"][0]["row_id"] == expected_row
    assert report["candidate_source_dates"] == ["2025-01-07"]
    assert report["candidate_source_dates_sha256"] == hashlib.sha256(
        b"2025-01-07\n"
    ).hexdigest()
    assert report["channel_metadata_status_counts"]["38000"]["DUPLICATE_ROW"] == 1


def test_metadata_candidate_scanner_rejects_real_access_without_distinct_review(
    tmp_path: Path,
) -> None:
    archive, _, review = _fixture(tmp_path)
    real_config = Path(__file__).resolve().parents[2] / "configs/aeon_test_metadata.json"
    with pytest.raises(ValueError, match="independent metadata review"):
        scan_test_metadata(archive, real_config, review, tmp_path / "real", fixture_only=False)


def test_scanner_digest_changes_with_corpus_dependency_without_touching_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marine_echo.training import aeon_corpus

    archive, config, review = _fixture(tmp_path)
    dependency = tmp_path / "aeon_corpus.py"
    dependency.write_bytes(Path(aeon_corpus.__file__).read_bytes())
    monkeypatch.setattr(aeon_corpus, "__file__", str(dependency))
    original = _scanner_code_sha256()
    dependency.write_bytes(dependency.read_bytes() + b"\n# synthetic dependency mutation\n")
    assert _scanner_code_sha256() != original
    with pytest.raises(ValueError, match="independent metadata review"):
        scan_test_metadata(
            archive, config, review, tmp_path / "must-not-open", fixture_only=True
        )
