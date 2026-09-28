"""Synthetic-only tests for external AEON archive metadata inspection."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from marine_echo.training import aeon_external_metadata
from marine_echo.training.aeon_external_metadata import (
    scan_external_metadata,
    scan_transfer_source_metadata,
)

HEADER = "Interval,Date_M,Time_M,Layer,Layer_depth_min,Layer_depth_max,Ping_S,Ping_E,Sv_mean\n"
MEMBERS = {
    f"AEON4_12345_{frequency}_2022_01_60minFullDepth.csv":
    HEADER
    + "1,20220101,00:00:00.000,1,0,200,1,150,SECRET_ACOUSTIC_VALUE\n"
    + "2,20220101,01:00:00.000,1,0,200,151,300,SECRET_ACOUSTIC_VALUE\n"
    for frequency in ("038", "125", "200", "455")
}


def _archive(path: Path, members: dict[str, str] = MEMBERS) -> dict[str, object]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for name, data in members.items():
            stream.writestr(name, data)
    return {
        "schema_version": "1.0",
        "publisher": "Synthetic publisher",
        "publisher_file_id": "synthetic-file-1",
        "publisher_url": "https://example.invalid/synthetic-file-1",
        "site": "Synthetic site",
        "deployment": "Synthetic deployment",
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "months": ["2022-01"],
        "members": sorted(name for name in members if "60minFullDepth.csv" in name),
    }


def test_scans_only_metadata_and_reports_acquisition_summary(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    manifest = _archive(archive)
    report = scan_external_metadata(archive, manifest)
    assert report["classification"] == "METADATA_ONLY_NO_ACOUSTIC_VALUES"
    assert report["source"]["archive_sha256"] == manifest["archive_sha256"]
    assert len(report["streams"]) == 4
    for stream in report["streams"]:
        assert stream["row_count"] == 2
        assert stream["ping_count_mode"] == 150
        assert stream["ping_count_histogram"] == {"150": 2}
        assert stream["cadence_seconds_histogram"] == {"3600": 1}
        assert stream["geometry_histogram"] == {"0:200": 2}
        assert stream["layer_id_histogram"] == {"1": 2}
    assert "SECRET_ACOUSTIC_VALUE" not in repr(report)


def test_rejects_wrong_digest_before_opening_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "source.zip"
    manifest = _archive(archive)
    manifest["archive_sha256"] = "0" * 64
    monkeypatch.setattr(zipfile, "ZipFile", lambda *_args, **_kwargs: pytest.fail("ZIP opened"))
    with pytest.raises(ValueError, match="digest"):
        scan_external_metadata(archive, manifest)


@pytest.mark.parametrize("bad_name", ["../escape.txt", "/absolute.txt", "a/../../b.txt"])
def test_rejects_unsafe_zip_paths(tmp_path: Path, bad_name: str) -> None:
    archive = tmp_path / "source.zip"
    manifest = _archive(archive, MEMBERS | {bad_name: "x"})
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        scan_external_metadata(archive, manifest)


def test_rejects_missing_channel(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    members = {name: value for name, value in MEMBERS.items() if "_455_" not in name}
    manifest = _archive(archive, members)
    with pytest.raises(ValueError, match="months/members"):
        scan_external_metadata(archive, manifest)


def test_rejects_duplicate_member_name(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, data in MEMBERS.items():
            stream.writestr(name, data)
        stream.writestr(next(iter(MEMBERS)), HEADER)
    manifest = _archive(tmp_path / "normal.zip")
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        scan_external_metadata(archive, manifest)


def test_rejects_oversized_declared_member_before_read(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    manifest = _archive(archive)
    with pytest.raises(ValueError, match="size"):
        scan_external_metadata(archive, manifest, max_member_bytes=100)


def test_rejects_missing_metadata_header(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    members = dict(MEMBERS)
    key = next(iter(members))
    members[key] = members[key].replace("Ping_E", "Ping_End", 1)
    manifest = _archive(archive, members)
    with pytest.raises(ValueError, match="header"):
        scan_external_metadata(archive, manifest)


def test_safe_explicit_directory_entry_is_allowed(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        stream.writestr("publisher_folder/", "")
        for name, data in MEMBERS.items():
            stream.writestr(name, data)
    manifest = _archive(tmp_path / "normal.zip")
    manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(scan_external_metadata(archive, manifest)["streams"]) == 4


def _transfer_contract(archive: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as stream:
        inventory = "".join(
            f"{info.filename}\t{info.file_size}\t{info.CRC:08x}\n"
            for info in sorted(stream.infolist(), key=lambda item: item.filename)
        )
    return {
        "schema_version": "1.0",
        "numeric_external_sv_access": "PROHIBITED",
        "source_clock": "PUBLISHER_REPORTED_TIMEZONE_UNSPECIFIED",
        "hourly_full_depth_central_inventory_serialization":
            "UTF8_SORTED_FILENAME_TAB_UNCOMPRESSED_DECIMAL_BYTES_TAB_CRC32_8_LOWER_HEX_LF_EACH_MEMBER",
        "metadata_fields": [
            "Date_M", "Time_M", "Interval", "Layer", "Layer_depth_min",
            "Layer_depth_max", "Ping_S", "Ping_E",
        ],
        "discard_without_use": list(aeon_external_metadata.DISCARDED_FIELDS),
        "publisher_article": "https://figshare.com/articles/dataset/AZFP/29247113",
        "study_id": "synthetic_external_transfer",
        "target": {
            "required_layer_id": 1,
            "required_layer_depth_min_m": 0,
            "required_layer_depth_max_m": 200,
            "required_complete_interval_ping_count": 150,
        },
        "sources": [{
            "role": "PRIMARY_CROSS_SITE_CONTEMPORANEOUS",
            "site": "AEON4_SYNTHETIC",
            "deployment": "Synthetic deployment",
            "file_id": 1234,
            "publisher_download_url": "https://ndownloader.figshare.com/files/1234",
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "filename_serial_identifier": "12345",
            "hourly_member_prefix": "AEON4_12345_",
            "first_month": "2022_01",
            "last_month": "2022_01",
            "hourly_full_depth_member_count": 4,
            "hourly_full_depth_central_inventory_sha256": hashlib.sha256(
                inventory.encode("utf-8")
            ).hexdigest(),
        }],
    }


def test_transfer_contract_binds_inventory_before_metadata_rows(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "source.zip"
    _archive(archive)
    contract = _transfer_contract(archive)
    result = scan_transfer_source_metadata(archive, contract, "PRIMARY_CROSS_SITE_CONTEMPORANEOUS")
    assert result["source"]["site"] == "AEON4_SYNTHETIC"
    assert result["streams"][0]["rows_with_contract_complete_ping_count"] == 2
    assert result["streams"][0]["rows_with_contract_layer_geometry"] == 2
    assert result["streams"][0]["rows_with_contract_layer_id"] == 2
    contract["sources"][0]["hourly_full_depth_central_inventory_sha256"] = "0" * 64
    monkeypatch.setattr(aeon_external_metadata, "_scan_member", lambda *_args, **_kwargs:
                        pytest.fail("metadata row opened"))
    with pytest.raises(ValueError, match="inventory digest"):
        scan_transfer_source_metadata(archive, contract, "PRIMARY_CROSS_SITE_CONTEMPORANEOUS")
