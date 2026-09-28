"""Bounded, metadata-only inspection of a manifest-bound AEON FullDepth ZIP.

This module never converts, retains, returns, or logs an acoustic value. Its
summary is acquisition evidence, not a numeric-QC, eligibility, or score report.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

FREQUENCIES = ("038", "125", "200", "455")
METADATA_FIELDS = (
    "Date_M", "Time_M", "Interval", "Layer", "Layer_depth_min", "Layer_depth_max",
    "Ping_S", "Ping_E",
)
DISCARDED_FIELDS = (
    "Sv_mean", "NASC", "Height_mean", "Depth_mean", "Dist_M", "Lat_M", "Lon_M",
    "Noise_Sv_1m", "Minimum_Sv_threshold_applied",
    "Maximum_Sv_threshold_applied", "Standard_deviation", "Thickness_mean",
    "Range_mean", "Exclude_below_line_range_mean", "Exclude_above_line_range_mean",
)
_MEMBER = re.compile(
    r"(?:[^/]+/)?(AEON\d+_\d+)_(038|125|200|455)_(\d{4})_(\d{2})_60minFullDepth\.csv"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MONTH = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _manifest_members(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    required = {
        "schema_version", "publisher", "publisher_file_id", "publisher_url", "site",
        "deployment", "archive_sha256", "months", "members",
    }
    if set(manifest) != required or manifest.get("schema_version") != "1.0":
        raise ValueError("External AEON manifest schema differs.")
    for field in ("publisher", "publisher_file_id", "publisher_url", "site", "deployment"):
        value = manifest[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 500:
            raise ValueError("External AEON manifest source identity is invalid.")
    if not isinstance(manifest["archive_sha256"], str) or not _SHA256.fullmatch(
        manifest["archive_sha256"]
    ):
        raise ValueError("External AEON manifest archive digest is invalid.")
    months = manifest["months"]
    members = manifest["members"]
    if (
        not isinstance(months, list) or not months or len(months) > 36
        or any(not isinstance(month, str) or not _MONTH.fullmatch(month) for month in months)
        or len(set(months)) != len(months)
        or not isinstance(members, list) or len(members) != 4 * len(months)
        or any(not isinstance(member, str) for member in members)
        or len(set(members)) != len(members)
    ):
        raise ValueError("External AEON manifest months/members are invalid.")
    pairs: set[tuple[str, str]] = set()
    prefixes: set[str] = set()
    for name in members:
        match = _MEMBER.fullmatch(name)
        if match is None:
            raise ValueError("External AEON manifest member name differs from FullDepth grammar.")
        prefix, frequency, year, month = match.groups()
        pairs.add((frequency, f"{year}-{month}"))
        prefixes.add(prefix)
    if len(prefixes) != 1 or pairs != {(frequency, month) for month in months for frequency in FREQUENCIES}:
        raise ValueError("External AEON manifest requires one deployment and four channels per month.")
    return tuple(sorted(members))


def _validate_zip(
    zf: zipfile.ZipFile,
    expected: tuple[str, ...],
    *,
    max_member_bytes: int,
    max_total_bytes: int,
    expected_inventory_sha256: str | None,
) -> tuple[dict[str, zipfile.ZipInfo], str]:
    infos = zf.infolist()
    if len(infos) > 2048:
        raise ValueError("External AEON ZIP has too many members.")
    names: set[str] = set()
    total = 0
    indexed: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        name_components = name[:-1].split("/") if info.is_dir() else name.split("/")
        if (
            not name or "\\" in name or ":" in name or "\x00" in name
            or name.startswith("/") or "//" in name
            or any(part in ("", ".", "..") for part in name_components)
            or (info.is_dir() and info.file_size != 0)
        ):
            raise ValueError("External AEON unsafe ZIP member path.")
        if name in names:
            raise ValueError("External AEON ZIP has duplicate member names.")
        names.add(name)
        file_mode = (info.external_attr >> 16) & 0o170000
        if file_mode == 0o120000 or info.flag_bits & 1 or info.compress_type not in (
            zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
        ):
            raise ValueError("External AEON ZIP has unsafe member type or compression.")
        if info.file_size < 0 or info.file_size > max_member_bytes:
            raise ValueError("External AEON ZIP member size exceeds limit.")
        total += info.file_size
        if total > max_total_bytes:
            raise ValueError("External AEON ZIP total size exceeds limit.")
        if info.compress_size and info.file_size > info.compress_size * 2000:
            raise ValueError("External AEON ZIP compression ratio exceeds limit.")
        indexed[name] = info
    if any("60minFullDepth" in name and _MEMBER.fullmatch(name) is None for name in names):
        raise ValueError("External AEON ZIP has an unrecognized FullDepth member.")
    found_full_depth = {name for name in names if _MEMBER.fullmatch(name)}
    if found_full_depth != set(expected):
        raise ValueError("External AEON ZIP differs from exact four-channel manifest members.")
    inventory_bytes = "".join(
        f"{name}\t{indexed[name].file_size}\t{indexed[name].CRC:08x}\n"
        for name in sorted(found_full_depth)
    ).encode("utf-8")
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    if expected_inventory_sha256 is not None and inventory_sha256 != expected_inventory_sha256:
        raise ValueError("External AEON central inventory digest differs from manifest.")
    return indexed, inventory_sha256


def _timestamp(date: str, time: str) -> datetime:
    for pattern in ("%Y%m%d%H:%M:%S.%f", "%Y%m%d%H:%M:%S"):
        try:
            # Publisher source clock has no established timezone; preserve naive local fields.
            return datetime.strptime(date + time.strip(), pattern)  # noqa: DTZ007
        except ValueError:
            continue
    raise ValueError("External AEON metadata source date/time is invalid.")


def _integer(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("External AEON metadata integer is invalid.") from exc
    if result < 0:
        raise ValueError("External AEON metadata integer is negative.")
    return result


def _geometry(value: str) -> str:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("External AEON metadata geometry is invalid.") from exc
    if not number.is_finite():
        raise ValueError("External AEON metadata geometry is nonfinite.")
    return format(number.normalize(), "f")


def _scan_member(zf: zipfile.ZipFile, name: str, *, max_rows: int) -> dict[str, Any]:
    row_count = 0
    ping_counts: Counter[int] = Counter()
    geometry: Counter[str] = Counter()
    layers: Counter[int] = Counter()
    cadence: Counter[int] = Counter()
    dates: Counter[str] = Counter()
    interval_ids: set[int] = set()
    duplicate_intervals = 0
    nonmonotone_time_steps = 0
    previous: datetime | None = None
    first: datetime | None = None
    last: datetime | None = None
    csv.field_size_limit(65536)
    with io.TextIOWrapper(zf.open(name), encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if (
            header is None or len(header) != len(set(header))
            or not set(METADATA_FIELDS).issubset(header) or "Sv_mean" not in header
        ):
            raise ValueError("External AEON FullDepth metadata header differs.")
        indices = {field: header.index(field) for field in METADATA_FIELDS}
        for raw in reader:
            if len(raw) != len(header):
                raise ValueError("External AEON metadata row width differs.")
            # Project the allowlist immediately; acoustic cells are never interpreted or retained.
            metadata = {field: raw[index] for field, index in indices.items()}
            del raw
            row_count += 1
            if row_count > max_rows:
                raise ValueError("External AEON FullDepth row count exceeds limit.")
            interval_id = _integer(metadata["Interval"])
            if interval_id in interval_ids:
                duplicate_intervals += 1
            interval_ids.add(interval_id)
            timestamp = _timestamp(metadata["Date_M"], metadata["Time_M"])
            if previous is not None:
                seconds = int((timestamp - previous).total_seconds())
                cadence[seconds] += 1
                if seconds <= 0:
                    nonmonotone_time_steps += 1
            previous = timestamp
            first = timestamp if first is None else min(first, timestamp)
            last = timestamp if last is None else max(last, timestamp)
            dates[timestamp.date().isoformat()] += 1
            ping_start = _integer(metadata["Ping_S"])
            ping_end = _integer(metadata["Ping_E"])
            if ping_end < ping_start:
                raise ValueError("External AEON metadata ping interval is reversed.")
            ping_counts[ping_end - ping_start + 1] += 1
            lower = _geometry(metadata["Layer_depth_min"])
            upper = _geometry(metadata["Layer_depth_max"])
            if Decimal(upper) <= Decimal(lower):
                raise ValueError("External AEON metadata layer geometry is reversed.")
            geometry[f"{lower}:{upper}"] += 1
            layers[_integer(metadata["Layer"])] += 1
    highest = max(ping_counts.values(), default=0)
    modes = sorted(count for count, n in ping_counts.items() if n == highest)
    return {
        "member": name,
        "row_count": row_count,
        "distinct_interval_count": len(interval_ids),
        "duplicate_interval_rows": duplicate_intervals,
        "first_source_timestamp": first.isoformat() if first else None,
        "last_source_timestamp": last.isoformat() if last else None,
        "source_date_row_counts": dict(sorted(dates.items())),
        "cadence_seconds_histogram": {str(k): v for k, v in sorted(cadence.items())},
        "nonmonotone_time_steps": nonmonotone_time_steps,
        "ping_count_histogram": {str(k): v for k, v in sorted(ping_counts.items())},
        "ping_count_mode": modes[0] if len(modes) == 1 else None,
        "ping_count_mode_ties": modes if len(modes) > 1 else [],
        "geometry_histogram": dict(sorted(geometry.items())),
        "layer_id_histogram": {str(k): v for k, v in sorted(layers.items())},
    }


def scan_external_metadata(
    archive: Path,
    manifest: Mapping[str, Any],
    *,
    max_member_bytes: int = 512 * 1024 * 1024,
    max_total_bytes: int = 8 * 1024 * 1024 * 1024,
    max_rows_per_member: int = 1_000_000,
    expected_inventory_sha256: str | None = None,
    expected_archive_bytes: int | None = None,
) -> dict[str, Any]:
    """Inspect only approved metadata fields of an exact manifest-bound archive.

    The caller must obtain independent approval before invoking this on a real
    archive. This function does not create an evaluation cohort or open Sv truth.
    """
    expected = _manifest_members(manifest)
    if min(max_member_bytes, max_total_bytes, max_rows_per_member) <= 0:
        raise ValueError("External AEON scan limits must be positive.")
    if expected_archive_bytes is not None and archive.stat().st_size != expected_archive_bytes:
        raise ValueError("External AEON archive byte size differs from manifest.")
    archive_sha256 = _sha256(archive)
    if archive_sha256 != manifest["archive_sha256"]:
        raise ValueError("External AEON archive digest differs from manifest.")
    with zipfile.ZipFile(archive) as zf:
        infos, inventory_sha256 = _validate_zip(
            zf, expected, max_member_bytes=max_member_bytes,
            max_total_bytes=max_total_bytes,
            expected_inventory_sha256=expected_inventory_sha256,
        )
        streams = []
        for name in expected:
            match = _MEMBER.fullmatch(name)
            assert match is not None
            _, frequency, year, month = match.groups()
            stream = _scan_member(zf, name, max_rows=max_rows_per_member)
            info = infos[name]
            stream.update({
                "frequency_khz": int(frequency),
                "source_file_month": f"{year}-{month}",
                "compressed_size_bytes": info.compress_size,
                "uncompressed_size_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
            })
            streams.append(stream)
    return {
        "classification": "METADATA_ONLY_NO_ACOUSTIC_VALUES",
        "source_time_basis": "SOURCE_REPORTED_UNSPECIFIED_NOT_UTC",
        "numeric_sv_access": "PROHIBITED",
        "source": {
            field: manifest[field]
            for field in ("publisher", "publisher_file_id", "publisher_url", "site", "deployment")
        } | {"archive_sha256": archive_sha256},
        "months": list(manifest["months"]),
        "hourly_full_depth_central_inventory_sha256": inventory_sha256,
        "streams": streams,
        "numeric_eligibility": "UNKNOWN_NOT_EVALUATED",
    }


def _month_range(first: str, last: str) -> list[str]:
    if not _MONTH.fullmatch(first.replace("_", "-")) or not _MONTH.fullmatch(
        last.replace("_", "-")
    ):
        raise ValueError("External AEON transfer month bounds are invalid.")
    year, month = (int(part) for part in first.split("_"))
    final_year, final_month = (int(part) for part in last.split("_"))
    months: list[str] = []
    while (year, month) <= (final_year, final_month) and len(months) <= 36:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if not months or len(months) > 36 or months[-1] != last.replace("_", "-"):
        raise ValueError("External AEON transfer month range is invalid.")
    return months


def scan_transfer_source_metadata(
    archive: Path,
    contract: Mapping[str, Any],
    source_role: str,
) -> dict[str, Any]:
    """Bind a post-hoc external transfer source to its exact frozen manifest.

    Invocation on a real archive requires a separate independent Stage-1 approval.
    This function itself is also usable with synthetic manifest fixtures.
    """
    if (
        contract.get("schema_version") != "1.0"
        or contract.get("numeric_external_sv_access") != "PROHIBITED"
        or contract.get("source_clock") != "PUBLISHER_REPORTED_TIMEZONE_UNSPECIFIED"
        or contract.get("hourly_full_depth_central_inventory_serialization")
        != "UTF8_SORTED_FILENAME_TAB_UNCOMPRESSED_DECIMAL_BYTES_TAB_CRC32_8_LOWER_HEX_LF_EACH_MEMBER"
        or contract.get("metadata_fields") != list(METADATA_FIELDS)
        or contract.get("discard_without_use") != list(DISCARDED_FIELDS)
        or not isinstance(contract.get("sources"), list)
    ):
        raise ValueError("External AEON transfer metadata contract differs.")
    matching = [
        source for source in contract["sources"]
        if isinstance(source, dict) and source.get("role") == source_role
    ]
    if len(matching) != 1:
        raise ValueError("External AEON transfer source role is absent or repeated.")
    source = matching[0]
    first = source["first_month"]
    last = source["last_month"]
    months = _month_range(first, last)
    prefix = source["hourly_member_prefix"]
    if not isinstance(prefix, str) or not re.fullmatch(
        r"(?:[^/]+/)?AEON\d+_\d+_", prefix
    ) or not prefix.endswith(f"_{source['filename_serial_identifier']}_") or not prefix.split("/")[-1].startswith(
        source["site"].split("_")[0] + "_"
    ):
        raise ValueError("External AEON transfer member prefix is invalid.")
    members = sorted(
        f"{prefix}{frequency}_{month.replace('-', '_')}_60minFullDepth.csv"
        for month in months for frequency in FREQUENCIES
    )
    if source["hourly_full_depth_member_count"] != len(members):
        raise ValueError("External AEON transfer four-channel member count differs.")
    publisher_url = source["publisher_download_url"]
    if not isinstance(source["file_id"], int) or publisher_url != (
        f"https://ndownloader.figshare.com/files/{source['file_id']}"
    ):
        raise ValueError("External AEON transfer publisher file identity differs.")
    if (
        contract.get("target", {}).get("required_layer_depth_min_m") != 0
        or contract.get("target", {}).get("required_layer_depth_max_m") != 200
        or contract.get("target", {}).get("required_layer_id") != 1
        or contract.get("target", {}).get("required_complete_interval_ping_count") != 150
    ):
        raise ValueError("External AEON transfer metadata target geometry differs.")
    manifest = {
        "schema_version": "1.0",
        "publisher": contract["publisher_article"],
        "publisher_file_id": str(source["file_id"]),
        "publisher_url": publisher_url,
        "site": source["site"],
        "deployment": source["deployment"],
        "archive_sha256": source["archive_sha256"],
        "months": months,
        "members": members,
    }
    report = scan_external_metadata(
        archive, manifest,
        expected_inventory_sha256=source["hourly_full_depth_central_inventory_sha256"],
        expected_archive_bytes=source["archive_bytes"],
    )
    for stream in report["streams"]:
        stream["rows_with_contract_complete_ping_count"] = stream["ping_count_histogram"].get(
            "150", 0
        )
        stream["rows_with_contract_layer_geometry"] = stream["geometry_histogram"].get(
            "0:200", 0
        )
        stream["rows_with_contract_layer_id"] = stream["layer_id_histogram"].get("1", 0)
    report["study_id"] = contract.get("study_id")
    report["source_role"] = source_role
    report["expected_complete_ping_count"] = 150
    report["expected_layer_geometry_m"] = "0:200"
    report["evaluation_status"] = "METADATA_ONLY_NO_NUMERIC_QC_OR_FORECASTS"
    report["source_identity_basis"] = "MANIFEST_ASSERTED_ARCHIVE_SHA256_VERIFIED"
    return report
