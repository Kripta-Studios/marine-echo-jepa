"""Reviewed AEON hourly FullDepth Sv reader with explicit source-product QC."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


FREQUENCIES_HZ = (38000, 125000, 200000, 455000)
SPECIAL_SV = {-999.0, 999.0, -9999.0, -9.9e37, 9.9e37}
AEON_ADR_SHA256 = "d83392832a1bde3ae3e096de0a664ca9cfc27762a18bb10fb5ace3a97e787435"
AEON_SOURCE_SHA256 = "4e72dd4dbec707b6bf15168e51f380cbe9145a78b595ef886d78cc3806c0ecde"
_MEMBER = re.compile(
    r"(?:[^/]+/)?AEON\d+_\d+_(038|125|200|455)_(\d{4})_(\d{2})_60minFullDepth\.csv"
)
_REQUIRED = {
    "Interval", "Layer", "Sv_mean", "Layer_depth_min", "Layer_depth_max",
    "Ping_S", "Ping_E", "Date_M", "Time_M",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class AeonHourlySlot:
    """One source interval with four channel values and distinct QC reasons."""

    interval_id: int
    source_timestamp: np.datetime64
    sv_db: NDArray[np.float64]
    observed_mask: NDArray[np.bool_]
    qc_status: tuple[str, str, str, str]
    member_names: tuple[str, ...]
    archive_sha256: str

    def with_sv_db(self, values: NDArray[np.float64]) -> AeonHourlySlot:
        if values.shape != (4,):
            raise ValueError("Changed Sv array has a different shape.")
        return replace(self, sv_db=values)


class AeonDevelopmentReader:
    """Read only complete review-allowed months of one hash-bound ZIP.

    The source clock is intentionally unspecified. A separate prefit reviewer
    must authorize every development member; no test member is opened here.
    """

    def __init__(
        self,
        archive: Path,
        *,
        review_path: Path,
        review_sha256: str,
        fixture_only: bool = False,
    ) -> None:
        self.archive = archive.resolve(strict=True)
        self.review_path = review_path.resolve(strict=True)
        if _sha256(self.review_path) != review_sha256:
            raise ValueError("AEON development review digest differs.")
        review = json.loads(self.review_path.read_text(encoding="utf-8"))
        archive_sha256 = _sha256(self.archive)
        if review.get("archive_sha256" if fixture_only else "source_archive_sha256") != archive_sha256:
            raise ValueError("AEON archive digest differs from development review.")
        if fixture_only:
            if (
                review.get("status") != "APPROVED_AEON_DEVELOPMENT"
                or review.get("source_time_basis") != "SOURCE_REPORTED_UNSPECIFIED"
                or review.get("data_kind") != "SYNTHETIC_FIXTURE"
            ):
                raise ValueError("Synthetic AEON fixture review is malformed.")
            members = review.get("permitted_members")
            months = review.get("permitted_months")
            start = np.datetime64(review["development_start"], "us")
            end = np.datetime64(review["development_end_exclusive"], "us")
        else:
            adr_path = Path(__file__).resolve().parents[3] / "docs/adr/0008-aeon-hourly-sv-study.md"
            if (
                archive_sha256 != AEON_SOURCE_SHA256
                or review.get("disposition") != "APPROVE_TRAIN_VALIDATION_DEVELOPMENT_ONLY"
                or review.get("reviewed_adr_sha256") != AEON_ADR_SHA256
                or _sha256(adr_path) != AEON_ADR_SHA256
            ):
                raise ValueError("AEON real development differs from the independent prefit review.")
            months = [f"2024-{month:02d}" for month in range(3, 12)]
            with zipfile.ZipFile(self.archive) as zf:
                members = [
                    name
                    for name in zf.namelist()
                    if (match := _MEMBER.fullmatch(name)) is not None
                    and f"{match.group(2)}-{match.group(3)}" in months
                ]
            start = np.datetime64("2024-03-01", "us")
            end = np.datetime64("2024-12-01", "us")
        if (
            not isinstance(members, list)
            or not members
            or len(set(members)) != len(members)
            or not isinstance(months, list)
            or not months
            or len(set(months)) != len(months)
        ):
            raise ValueError("AEON development member/month allowlist is malformed.")
        if np.isnat(start) or np.isnat(end) or end <= start:
            raise ValueError("AEON development bounds are invalid.")
        if (
            start.astype("datetime64[M]").astype("datetime64[us]") != start
            or end.astype("datetime64[M]").astype("datetime64[us]") != end
        ):
            raise ValueError("AEON development member bounds must cover whole months.")
        seen: set[tuple[str, str]] = set()
        for member in members:
            match = _MEMBER.fullmatch(member) if isinstance(member, str) else None
            if match is None:
                raise ValueError("AEON FullDepth member name is invalid.")
            frequency, year, month = match.groups()
            month_key = f"{year}-{month}"
            month_start = np.datetime64(month_key, "M").astype("datetime64[us]")
            month_end = (np.datetime64(month_key, "M") + 1).astype("datetime64[us]")
            if month_key not in months or month_start < start or month_end > end:
                raise ValueError("AEON member is outside reviewed development months.")
            pair = (frequency, month_key)
            if pair in seen:
                raise ValueError("AEON review repeats a frequency/month member.")
            seen.add(pair)
        required = {(frequency, month) for frequency in ("038", "125", "200", "455") for month in months}
        if seen != required:
            raise ValueError("Every approved AEON month needs all four FullDepth members.")
        self.members = tuple(sorted(members))
        self.start = start
        self.end = end
        self.archive_sha256 = archive_sha256
        self.fixture_only = fixture_only

    def __iter__(self) -> Iterator[AeonHourlySlot]:
        if not self.fixture_only:
            raise ValueError("Real AEON values require an explicit TRAIN or validation partition.")
        return self._iter_partition(None)

    def iter_partition(self, partition: str) -> Iterator[AeonHourlySlot]:
        if self.fixture_only:
            raise ValueError("Synthetic fixtures use the fixture iterator.")
        if partition not in ("train", "validation"):
            raise ValueError("Only TRAIN and validation outcomes are approved for development.")
        return self._iter_partition(partition)

    def _iter_partition(self, partition: str | None) -> Iterator[AeonHourlySlot]:
        bounds = {
            "train": ("20240306", "20241008", "2024-03", "2024-10"),
            "validation": ("20241008", "20241201", "2024-10", "2024-11"),
        }
        records: dict[int, dict[int, list[tuple[dict[str, str], str]]]] = {}
        with zipfile.ZipFile(self.archive) as zf:
            names = zf.namelist()
            if len(names) != len(set(names)) or not set(self.members).issubset(names):
                raise ValueError("AEON ZIP has duplicate or missing reviewed member names.")
            for member in self.members:
                match = _MEMBER.fullmatch(member)
                assert match is not None
                member_month = f"{match.group(2)}-{match.group(3)}"
                if partition is not None and not bounds[partition][2] <= member_month <= bounds[partition][3]:
                    continue
                frequency = int(match.group(1)) * 1000
                with io.TextIOWrapper(zf.open(member), encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames is None or not _REQUIRED.issubset(reader.fieldnames):
                        raise ValueError("AEON FullDepth CSV lacks required source columns.")
                    for row in reader:
                        if partition is not None and not bounds[partition][0] <= row["Date_M"] < bounds[partition][1]:
                            # Month containers can straddle TRAIN/validation; do not
                            # convert or aggregate acoustic values outside this split.
                            continue
                        interval_id = int(row["Interval"])
                        records.setdefault(interval_id, {}).setdefault(frequency, []).append((row, member))
        previous_time: np.datetime64 | None = None
        for interval_id, channels in sorted(records.items()):
            candidates = [row for rows in channels.values() for row, _ in rows]
            representative = channels.get(38000, [(candidates[0], "")])[0][0]
            timestamp = np.datetime64(
                datetime.strptime(
                    representative["Date_M"] + representative["Time_M"].strip(),
                    "%Y%m%d%H:%M:%S.%f",
                ),
                "us",
            )
            if timestamp < self.start or timestamp >= self.end:
                raise ValueError("AEON source timestamp exceeds reviewed development bounds.")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("AEON interval timestamps are nonmonotone.")
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
                channel_time = np.datetime64(
                    datetime.strptime(
                        row["Date_M"] + row["Time_M"].strip(), "%Y%m%d%H:%M:%S.%f"
                    ),
                    "us",
                )
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
