"""Review-gated, metadata-only AEON retrospective TEST candidate universe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from marine_echo.training import aeon_corpus
from marine_echo.training.aeon_corpus import (
    AEON_ADR_SHA256, AEON_SOURCE_SHA256, FREQUENCIES_HZ, _MEMBER, _sha256,
)


_METADATA_FIELDS = (
    "Date_M", "Time_M", "Interval", "Layer", "Layer_depth_min",
    "Layer_depth_max", "Ping_S", "Ping_E",
)
_MONTHS = ("2025-01", "2025-02")
_PAIRS = tuple(
    f"{frequency}:{month}"
    for month in _MONTHS for frequency in ("038", "125", "200", "455")
)
_START_DAY = "20250106"
_END_DAY = "20250301"
_CODE_DEPENDENCIES = ("aeon_test_metadata.py", "aeon_corpus.py")


def _scanner_code_sha256() -> str:
    """Bind the scanner and imported corpus constants/parser grammar as one unit."""
    digest = hashlib.sha256()
    for name, path in (
        ("aeon_test_metadata.py", Path(__file__)),
        ("aeon_corpus.py", Path(aeon_corpus.__file__)),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _config_gate(config: dict[str, Any], *, fixture_only: bool) -> None:
    root = Path(__file__).resolve().parents[3]
    adr_path = root / "docs/adr/0009-aeon-eligible-date-and-evaluation-freeze.md"
    if (
        config.get("schema_version") != "1.0"
        or config.get("study_id") != "aeon3_geb_2024_hourly_sv_v1"
        or config.get("phase") != "retrospective_test_metadata_candidate_universe"
        or config.get("status") != (
            "SYNTHETIC_FIXTURE_ONLY" if fixture_only
            else "PROPOSED_FOR_INDEPENDENT_METADATA_REVIEW"
        )
        or config.get("classification") != "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED"
        or (not fixture_only and config.get("source_archive_sha256") != AEON_SOURCE_SHA256)
        or config.get("protocol_sha256") != AEON_ADR_SHA256
        or config.get("evaluation_adr_sha256") != _sha256(adr_path)
        or config.get("source_time_basis") != "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC"
        or config.get("partition") != "test"
        or config.get("start_source_date_inclusive") != "2025-01-06"
        or config.get("end_source_date_exclusive") != "2025-03-01"
        or config.get("permitted_months") != list(_MONTHS)
        or config.get("required_member_pairs") != list(_PAIRS)
        or config.get("member_rule")
        != "exactly_one_60minFullDepth_csv_per_frequency_month_reviewed_filename_pattern"
        or config.get("metadata_fields") != list(_METADATA_FIELDS)
        or config.get("discard_field_without_use") != "Sv_mean"
        or config.get("context_interval_count") != 24
        or config.get("horizon_interval_steps") != [1, 3, 6]
        or config.get("nominal_complete_pings") != 150
        or config.get("layer_depth_bounds_m") != [0.0, 200.0]
        or config.get("channel_time_tolerance_minutes") != 5
        or config.get("min_step_minutes") != 55
        or config.get("max_step_minutes") != 65
        or config.get("candidate_rule")
        != "exact_24_predecessor_ids_metadata_complete_38khz_contiguous_source_times_and_plus6_within_partition_last_interval"
        or config.get("future_metadata_filter") != "NONE"
        or config.get("error_based_filter") != "PROHIBITED"
        or config.get("numeric_test_outcome_access") != "PROHIBITED"
        or config.get("scanner_code_dependencies") != list(_CODE_DEPENDENCIES)
        or config.get("actual_issued_and_scored_status")
        != "UNKNOWN_UNTIL_SEPARATE_APPROVED_NUMERIC_TEST_MATERIALIZATION"
    ):
        raise ValueError("AEON TEST metadata config differs from fixed candidate rule.")


def _source_time(row: dict[str, str]) -> np.datetime64:
    return np.datetime64(
        datetime.strptime(row["Date_M"] + row["Time_M"].strip(), "%Y%m%d%H:%M:%S.%f"),
        "us",
    )


def _read_metadata(
    archive: Path,
) -> tuple[dict[int, dict[int, list[dict[str, str]]]], list[dict[str, Any]]]:
    """Discard each Sv_mean cell before even parsing its interval metadata."""
    records: dict[int, dict[int, list[dict[str, str]]]] = {}
    member_metadata = []
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError("AEON TEST ZIP repeats member names.")
        selected: dict[str, str] = {}
        for name in names:
            match = _MEMBER.fullmatch(name)
            if match is None:
                continue
            frequency, year, month = match.groups()
            pair = f"{frequency}:{year}-{month}"
            if pair not in _PAIRS:
                continue
            if pair in selected:
                raise ValueError("AEON TEST ZIP repeats an allowed frequency/month member.")
            selected[pair] = name
        if set(selected) != set(_PAIRS):
            raise ValueError("AEON TEST ZIP lacks exact four-channel Jan/Feb FullDepth members.")
        for pair in _PAIRS:
            member = selected[pair]
            info = zf.getinfo(member)
            member_metadata.append({
                "frequency_month": pair, "name": member,
                "compressed_size": info.compress_size, "source_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
            })
            frequency = int(pair.split(":", 1)[0]) * 1000
            with io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header is None or len(header) != len(set(header)) or not set(_METADATA_FIELDS).issubset(header) or "Sv_mean" not in header:
                    raise ValueError("AEON TEST FullDepth metadata columns differ.")
                indices = {field: header.index(field) for field in _METADATA_FIELDS}
                sv_index = header.index("Sv_mean")
                for raw in reader:
                    if len(raw) != len(header):
                        raise ValueError("AEON TEST metadata CSV row width differs.")
                    raw[sv_index] = ""  # Immediately discard acoustic text; never convert or persist.
                    source_day = raw[indices["Date_M"]]
                    if not _START_DAY <= source_day < _END_DAY:
                        continue
                    metadata = {field: raw[position] for field, position in indices.items()}
                    interval_id = int(metadata["Interval"])
                    records.setdefault(interval_id, {}).setdefault(frequency, []).append(metadata)
    return records, member_metadata


def _channel_status(
    rows: list[dict[str, str]], representative_time: np.datetime64,
) -> str:
    if not rows:
        return "MISSING_ROW"
    if len(rows) != 1:
        return "DUPLICATE_ROW"
    row = rows[0]
    difference = _source_time(row) - representative_time
    if difference < -np.timedelta64(5, "m") or difference > np.timedelta64(5, "m"):
        return "SOURCE_TIME_MISMATCH"
    if (
        int(row["Layer"]) != 1
        or float(row["Layer_depth_min"]) != 0.0
        or float(row["Layer_depth_max"]) != 200.0
    ):
        return "INVALID_GEOMETRY"
    if int(row["Ping_E"]) - int(row["Ping_S"]) + 1 != 150:
        return "PARTIAL_SOURCE_INTERVAL"
    return "METADATA_COMPLETE_NUMERIC_SV_UNKNOWN"


def _candidate_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"aeon-test-candidates-v1\n")
    for row in rows:
        target_ids = ",".join(str(number) for number in row["target_interval_ids"])
        digest.update(f"{row['cutoff_interval_id']}|{row['row_id']}|{target_ids}\n".encode("ascii"))
    return digest.hexdigest()


def _candidate_report(
    records: dict[int, dict[int, list[dict[str, str]]]],
    archive_sha256: str,
    member_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    intervals: dict[int, dict[str, Any]] = {}
    source_dates: dict[str, int] = {}
    for interval_id, channels in sorted(records.items()):
        all_rows = [row for channel_rows in channels.values() for row in channel_rows]
        representative = channels.get(38000, [all_rows[0]])[0]
        timestamp = _source_time(representative)
        date = str(timestamp.astype("datetime64[D]"))
        source_dates[date] = source_dates.get(date, 0) + 1
        intervals[interval_id] = {
            "source_timestamp": str(timestamp),
            "source_date": date,
            "channels": {
                str(frequency): _channel_status(channels.get(frequency, []), timestamp)
                for frequency in FREQUENCIES_HZ
            },
        }
    candidate_rows = []
    reasons_by_cutoff = {}
    if intervals:
        first_id, last_id = min(intervals), max(intervals)
        for cutoff_id in sorted(intervals):
            reasons: set[str] = set()
            if cutoff_id + 6 > last_id:
                reasons.add("PLUS6_BEYOND_LAST_SOURCE_INTERVAL")
            prior_ids = range(cutoff_id - 23, cutoff_id + 1)
            if cutoff_id - 23 < first_id:
                reasons.add("CONTEXT_CROSSES_TEST_PARTITION_START")
            prior = [intervals.get(interval_id) for interval_id in prior_ids]
            if any(item is None for item in prior):
                reasons.add("MISSING_PREDECESSOR_INTERVAL")
            if not reasons.intersection({"MISSING_PREDECESSOR_INTERVAL", "CONTEXT_CROSSES_TEST_PARTITION_START"}):
                complete = [item for item in prior if item is not None]
                for item in complete:
                    status = item["channels"]["38000"]
                    if status != "METADATA_COMPLETE_NUMERIC_SV_UNKNOWN":
                        reasons.add("PREDECESSOR_38_" + status)
                for left, right in zip(complete, complete[1:]):
                    delta = np.datetime64(right["source_timestamp"]) - np.datetime64(left["source_timestamp"])
                    if not np.timedelta64(55, "m") <= delta <= np.timedelta64(65, "m"):
                        reasons.add("PREDECESSOR_SOURCE_TIME_DISCONTINUITY")
            if reasons:
                reasons_by_cutoff[str(cutoff_id)] = sorted(reasons)
                continue
            row_id = hashlib.sha256(
                f"aeon-full-depth:{archive_sha256}:{cutoff_id}:24:1,3,6".encode("ascii")
            ).hexdigest()
            candidate_rows.append({
                "cutoff_interval_id": cutoff_id,
                "cutoff_source_timestamp": intervals[cutoff_id]["source_timestamp"],
                "cutoff_source_date": intervals[cutoff_id]["source_date"],
                "row_id": row_id,
                "target_interval_ids": [cutoff_id + step for step in (1, 3, 6)],
                "numeric_issuance_status": "UNKNOWN",
                "target_scoring_status": "UNKNOWN",
            })
    channel_status_counts: dict[str, dict[str, int]] = {}
    for interval in intervals.values():
        for frequency, status in interval["channels"].items():
            counts = channel_status_counts.setdefault(frequency, {})
            counts[status] = counts.get(status, 0) + 1
    candidate_dates = sorted({row["cutoff_source_date"] for row in candidate_rows})
    candidate_date_hash = hashlib.sha256(
        ("\n".join(candidate_dates) + "\n").encode("ascii")
    ).hexdigest()
    return {
        "classification": "METADATA_ONLY_CANDIDATES_NOT_ISSUED_OR_SCORED",
        "numeric_test_outcome_access": "PROHIBITED",
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "source_archive_sha256": archive_sha256,
        "partition_start_source_date_inclusive": "2025-01-06",
        "partition_end_source_date_exclusive": "2025-03-01",
        "members": member_metadata,
        "interval_count": len(intervals),
        "source_interval_ids": sorted(intervals),
        "source_date_interval_counts": dict(sorted(source_dates.items())),
        "interval_metadata": {str(key): value for key, value in intervals.items()},
        "channel_metadata_status_counts": {
            frequency: dict(sorted(counts.items()))
            for frequency, counts in sorted(channel_status_counts.items())
        },
        "candidate_cutoff_interval_ids": [row["cutoff_interval_id"] for row in candidate_rows],
        "candidate_rows": candidate_rows,
        "candidate_sha256": _candidate_hash(candidate_rows),
        "candidate_source_dates": candidate_dates,
        "candidate_source_dates_sha256": candidate_date_hash,
        "candidate_count": len(candidate_rows),
        "metadata_exclusion_reasons_by_cutoff": reasons_by_cutoff,
        "actual_issued_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
        "actual_scored_rows": "UNKNOWN_NUMERIC_QC_NOT_OPENED",
    }


def scan_test_metadata(
    archive: Path,
    config_path: Path,
    review_path: Path,
    output: Path,
    *,
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Run only after exact distinct review; synthetic fixtures use a separate gate."""
    config_sha256 = _sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _config_gate(config, fixture_only=fixture_only)
    code_sha256 = _scanner_code_sha256()
    review_sha256 = _sha256(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("status") != (
            "APPROVED_AEON_TEST_METADATA_FIXTURE" if fixture_only
            else "APPROVED_AEON_TEST_METADATA_SCAN"
        )
        or review.get("scanner_code_sha256") != code_sha256
        or review.get("config_sha256") != config_sha256
        or review.get("source_archive_sha256") != config["source_archive_sha256"]
        or review.get("numeric_test_outcome_access") != "PROHIBITED"
    ):
        raise ValueError("AEON TEST scanner lacks exact independent metadata review.")
    archive = archive.resolve(strict=True)
    archive_sha256 = _sha256(archive)
    if archive_sha256 != config["source_archive_sha256"]:
        raise ValueError("AEON TEST metadata source archive hash differs from review.")
    records, members = _read_metadata(archive)
    report = _candidate_report(records, archive_sha256, members)
    report.update({
        "status": "METADATA_CANDIDATE_UNIVERSE_PENDING_INDEPENDENT_REVIEW",
        "config_sha256": config_sha256,
        "scanner_code_sha256": code_sha256,
        "scanner_code_dependencies": list(_CODE_DEPENDENCIES),
        "prefit_review_sha256": review_sha256,
    })
    output = output.resolve()
    if output.exists():
        raise FileExistsError("AEON TEST metadata candidate output already exists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage.", dir=output.parent))
    try:
        (stage / "metadata-candidates.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    except BaseException:
        if stage.exists() and stage.parent == output.parent:
            (stage / "metadata-candidates.json").unlink(missing_ok=True)
            stage.rmdir()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = scan_test_metadata(
        arguments.archive, arguments.config, arguments.review, arguments.output
    )
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "candidate_sha256": result["candidate_sha256"],
        "actual_issued_rows": result["actual_issued_rows"],
        "actual_scored_rows": result["actual_scored_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
