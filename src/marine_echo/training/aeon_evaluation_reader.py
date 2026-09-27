"""Separate review-gated AEON calibration and retrospective-test source reader."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from marine_echo.training.aeon_corpus import (
    AEON_ADR_SHA256,
    AEON_SOURCE_SHA256,
    FREQUENCIES_HZ,
    SPECIAL_SV,
    AeonHourlySlot,
    _MEMBER,
    _REQUIRED,
    _sha256,
)
from marine_echo.training.aeon_windows import AeonHourlyWindow, AeonWindowPlan, iter_aeon_windows


_BOUNDS = {
    "calibration": ("2024-12-01", "2025-01-06", ("2024-12", "2025-01")),
    "test": ("2025-01-06", "2025-03-01", ("2025-01", "2025-02")),
}


class AeonEvaluationReader:
    """Require a separate exact access review before reading either held-out split."""

    def __init__(
        self,
        archive: Path,
        *,
        review_path: Path,
        review_sha256: str,
        partition: str,
        fixture_only: bool = False,
    ) -> None:
        if partition not in _BOUNDS:
            raise ValueError("AEON evaluation partition must be calibration or test.")
        self.archive = archive.resolve(strict=True)
        self.review_path = review_path.resolve(strict=True)
        if _sha256(self.review_path) != review_sha256:
            raise ValueError("AEON evaluation review digest differs.")
        self.review_sha256 = review_sha256
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        self.archive_sha256 = _sha256(self.archive)
        self.partition = partition
        self.fixture_only = fixture_only
        if (
            review.get("partition") != partition
            or review.get("source_archive_sha256") != self.archive_sha256
            or review.get("reader_code_sha256") != _sha256(Path(__file__))
        ):
            raise ValueError("AEON evaluation partition, archive or reader code review differs.")
        if fixture_only:
            if (
                review.get("status") != "APPROVED_AEON_EVALUATION_READER_FIXTURE"
                or review.get("data_kind") != "SYNTHETIC_FIXTURE"
                or review.get("test_access") != "FIXTURE_ONLY"
            ):
                raise ValueError("AEON evaluation fixture lacks a synthetic-only review.")
            start_text, end_text = review["start"], review["end_exclusive"]
            months = {start_text[:7]}
        else:
            expected_status = (
                "APPROVED_AEON_CALIBRATION_ACCESS" if partition == "calibration"
                else "APPROVED_AEON_RETROSPECTIVE_TEST_ACCESS"
            )
            start_text, end_text, expected_months = _BOUNDS[partition]
            months = set(expected_months)
            if (
                review.get("status") != expected_status
                or review.get("data_kind") != "REAL"
                or review.get("protocol_sha256") != AEON_ADR_SHA256
                or self.archive_sha256 != AEON_SOURCE_SHA256
                or review.get("start") != start_text
                or review.get("end_exclusive") != end_text
                or not isinstance(review.get("selection_review_sha256"), str)
                or len(review["selection_review_sha256"]) != 64
                or review.get("test_access")
                != ("PROHIBITED" if partition == "calibration" else "RETROSPECTIVE_APPROVED")
            ):
                raise ValueError("AEON real evaluation lacks exact reviewed split/access lineage.")
            if partition == "test" and (
                not isinstance(review.get("pretest_freeze_sha256"), str)
                or len(review["pretest_freeze_sha256"]) != 64
                or not isinstance(review.get("calibration_artifact_sha256"), str)
                or len(review["calibration_artifact_sha256"]) != 64
            ):
                raise ValueError("AEON retrospective TEST lacks independent pretest freeze.")
        self.start = np.datetime64(start_text, "us")
        self.end = np.datetime64(end_text, "us")
        if np.isnat(self.start) or np.isnat(self.end) or self.end <= self.start:
            raise ValueError("AEON evaluation source-date bounds are invalid.")
        members = review.get("permitted_members")
        if not isinstance(members, list) or not members or len(set(members)) != len(members):
            raise ValueError("AEON evaluation source member allowlist is invalid.")
        pairs = set()
        for member in members:
            match = _MEMBER.fullmatch(member) if isinstance(member, str) else None
            if match is None:
                raise ValueError("AEON evaluation FullDepth member name is invalid.")
            frequency, year, month = match.groups()
            month_key = f"{year}-{month}"
            if month_key not in months:
                raise ValueError("AEON evaluation member crosses unreviewed source months.")
            pairs.add((frequency, month_key))
        required = {(frequency, month) for frequency in ("038", "125", "200", "455") for month in months}
        if pairs != required:
            raise ValueError("AEON evaluation needs all four reviewed FullDepth channels/month.")
        self.members = tuple(sorted(members))

    def iter_slots(self) -> Iterator[AeonHourlySlot]:
        """Filter by source Date_M before parsing interval IDs or acoustic values."""
        if (
            _sha256(self.archive) != self.archive_sha256
            or _sha256(self.review_path) != self.review_sha256
        ):
            raise ValueError("AEON evaluation archive or review changed after access gate.")
        start_day = str(self.start.astype("datetime64[D]")).replace("-", "")
        end_day = str(self.end.astype("datetime64[D]")).replace("-", "")
        records: dict[int, dict[int, list[tuple[dict[str, str], str]]]] = {}
        with zipfile.ZipFile(self.archive) as zf:
            names = zf.namelist()
            if len(names) != len(set(names)) or not set(self.members).issubset(names):
                raise ValueError("AEON evaluation ZIP has duplicate/missing reviewed members.")
            for member in self.members:
                match = _MEMBER.fullmatch(member)
                assert match is not None
                frequency = int(match.group(1)) * 1000
                with io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames is None or not _REQUIRED.issubset(reader.fieldnames):
                        raise ValueError("AEON evaluation FullDepth columns differ from review.")
                    for row in reader:
                        if not start_day <= row["Date_M"] < end_day:
                            continue
                        interval_id = int(row["Interval"])
                        records.setdefault(interval_id, {}).setdefault(frequency, []).append((row, member))
        previous_time: np.datetime64 | None = None
        for interval_id, channels in sorted(records.items()):
            candidates = [row for channel_rows in channels.values() for row, _ in channel_rows]
            representative = channels.get(38000, [(candidates[0], "")])[0][0]
            timestamp = np.datetime64(datetime.strptime(
                representative["Date_M"] + representative["Time_M"].strip(),
                "%Y%m%d%H:%M:%S.%f",
            ), "us")
            if timestamp < self.start or timestamp >= self.end:
                raise ValueError("AEON evaluation source timestamp exceeds reviewed partition.")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("AEON evaluation source interval timestamps are nonmonotone.")
            previous_time = timestamp
            values = np.full(4, np.nan)
            mask = np.zeros(4, dtype=bool)
            statuses: list[str] = []
            member_names: set[str] = set()
            for index, frequency in enumerate(FREQUENCIES_HZ):
                channel_rows = channels.get(frequency, [])
                member_names.update(member for _, member in channel_rows)
                if not channel_rows:
                    statuses.append("MISSING_ROW")
                    continue
                if len(channel_rows) != 1:
                    statuses.append("DUPLICATE_ROW")
                    continue
                row = channel_rows[0][0]
                channel_time = np.datetime64(datetime.strptime(
                    row["Date_M"] + row["Time_M"].strip(), "%Y%m%d%H:%M:%S.%f"
                ), "us")
                if abs(channel_time - timestamp) > np.timedelta64(5, "m"):
                    statuses.append("SOURCE_TIME_MISMATCH")
                    continue
                if (
                    int(row["Layer"]) != 1
                    or float(row["Layer_depth_min"]) != 0.0
                    or float(row["Layer_depth_max"]) != 200.0
                ):
                    statuses.append("INVALID_GEOMETRY")
                    continue
                if int(row["Ping_E"]) - int(row["Ping_S"]) + 1 != 150:
                    statuses.append("PARTIAL_SOURCE_INTERVAL")
                    continue
                raw = row["Sv_mean"].strip()
                value = float(raw) if raw else float("nan")
                if not np.isfinite(value) or value in SPECIAL_SV:
                    statuses.append("INVALID_OR_SPECIAL_SV")
                    continue
                values[index] = value
                mask[index] = True
                statuses.append("OBSERVED_SOURCE_PRODUCT")
            yield AeonHourlySlot(
                interval_id=interval_id,
                source_timestamp=timestamp,
                sv_db=values,
                observed_mask=mask,
                qc_status=tuple(statuses),  # type: ignore[arg-type]
                member_names=tuple(sorted(member_names)),
                archive_sha256=self.archive_sha256,
            )

    def iter_windows(self) -> Iterator[AeonHourlyWindow]:
        """Use the unchanged 24-past issue rule while retaining CAL/TEST identity."""
        for row in iter_aeon_windows(
            self.iter_slots(), plan=AeonWindowPlan(), partition="development_assessment",
            partition_start=str(self.start), partition_end_exclusive=str(self.end),
        ):
            yield replace(row, partition=self.partition)
