"""Fail-closed synthetic tests for the external two-source metadata access gate."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from marine_echo.training import aeon_external_metadata as scanner
from marine_echo.training import aeon_external_metadata_run as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _archive(path: Path, station: str) -> tuple[int, str]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for frequency in scanner.FREQUENCIES:
            name = f"{station}_12345_{frequency}_2022_01_60minFullDepth.csv"
            lines = [",".join(scanner.EXACT_CSV_HEADER)]
            for interval in range(1, 32):
                instant = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(hours=interval - 1)
                row = {field: "0" for field in scanner.EXACT_CSV_HEADER}
                row.update({
                    "Process_ID": "PRIVATE_PROCESS_ID",
                    "Interval": str(interval), "Layer": "1", "Sv_mean": "SECRET_SV",
                    "Layer_depth_min": "0", "Layer_depth_max": "200",
                    "Ping_S": str((interval - 1) * 150 + 1), "Ping_E": str(interval * 150),
                    "Date_M": instant.strftime("%Y%m%d"),
                    "Time_M": instant.strftime("%H:%M:%S.%f")[:-3],
                })
                lines.append(",".join(row[field] for field in scanner.EXACT_CSV_HEADER))
            stream.writestr(name, "\n".join(lines) + "\n")
    with zipfile.ZipFile(path) as stream:
        inventory = "".join(
            f"{info.filename}\t{info.file_size}\t{info.CRC:08x}\n"
            for info in sorted(stream.infolist(), key=lambda item: item.filename)
        )
    return path.stat().st_size, hashlib.sha256(inventory.encode()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    source_contract = runner.ROOT / "configs/aeon_external_transfer.json"
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    contract["study_id"] = "synthetic_external_transfer"
    sources = []
    for index, role in enumerate(runner.ROLES, start=4):
        station = f"AEON{index}"
        archive = tmp_path / f"{station}.zip"
        size, inventory_sha = _archive(archive, station)
        sources.append({
            "role": role, "site": f"{station}_SYNTHETIC", "deployment": "Synthetic",
            "file_id": 1000 + index,
            "publisher_download_url": f"https://ndownloader.figshare.com/files/{1000 + index}",
            "archive_path": str(archive), "archive_bytes": size,
            "archive_sha256": _sha(archive), "filename_serial_identifier": "12345",
            "hourly_member_prefix": f"{station}_12345_", "first_month": "2022_01",
            "last_month": "2022_01", "hourly_full_depth_member_count": 4,
            "hourly_full_depth_central_inventory_sha256": inventory_sha,
        })
    contract["sources"] = sources
    paths = {key: tmp_path / f"{key}.json" for key in (
        "contract", "adr", "stage0", "amendment", "review",
    )}
    _write(paths["contract"], contract)
    paths["adr"].write_text("Synthetic ADR\n", encoding="utf-8")
    _write(paths["stage0"], {
        "verdict": "APPROVE_STAGE0_DESIGN_ONLY",
        "reviewer_task": "/root/external_reviewer",
        "reviewed_contracts": {"config_sha256": _sha(paths["contract"])},
    })
    _write(paths["amendment"], {
        "verdict": "APPROVE_STAGE0_WORDING_AMENDMENT_ONLY",
        "reviewer_task": "/root/external_reviewer",
        "prior_stage0_review_sha256": _sha(paths["stage0"]),
        "amended_adr_sha256": _sha(paths["adr"]),
        "unchanged_config_sha256": _sha(paths["contract"]),
        "scanner_source_sha256_after_wording_change": _sha(Path(scanner.__file__)),
    })
    _write(paths["review"], {
        "verdict": "APPROVE_STAGE1_METADATA_RUNNER_FIXTURE",
        "reviewer_task": "/root/external_reviewer",
        "metadata_row_access": "APPROVED_EXACT_TWO_SOURCE_SCAN",
        "numeric_external_sv_access": "PROHIBITED",
        "study_id": contract["study_id"],
        "bindings": {
            "config_sha256": _sha(paths["contract"]),
            "adr_sha256": _sha(paths["adr"]),
            "stage0_review_sha256": _sha(paths["stage0"]),
            "amendment_review_sha256": _sha(paths["amendment"]),
            "scanner_sha256": _sha(Path(scanner.__file__)),
            "runner_sha256": _sha(Path(runner.__file__)),
            "sources": [{
                "role": source["role"], "archive_path": source["archive_path"],
                "archive_sha256": source["archive_sha256"],
                "archive_bytes": source["archive_bytes"],
                "inventory_sha256": source["hourly_full_depth_central_inventory_sha256"],
            } for source in sources],
        },
    })
    return paths


def _invoke(paths: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    return runner.run_stage1(
        contract_path=paths["contract"], adr_path=paths["adr"],
        stage0_path=paths["stage0"], amendment_path=paths["amendment"],
        review_path=paths["review"], output_dir=output_dir, fixture_only=True,
    )


def test_two_source_fixture_reports_are_redacted_and_exclusive(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    outputs = _invoke(paths, tmp_path / "out")
    assert set(outputs) == set(runner.ROLES)
    for output in outputs.values():
        payload = output.read_text(encoding="utf-8")
        assert "SECRET_SV" not in payload and "PRIVATE_PROCESS_ID" not in payload
        value = json.loads(payload)
        assert value["candidate_count"] >= 1
        assert value["actual_issued_rows"] == "UNKNOWN_NUMERIC_QC_NOT_OPENED"
        assert value["access_lineage"]["stage1_review_sha256"] == _sha(paths["review"])
    with pytest.raises(FileExistsError, match="already exists"):
        _invoke(paths, tmp_path / "out")


@pytest.mark.parametrize("change", [
    "missing_review", "wrong_reviewer", "stale_config", "stale_scanner",
    "stale_runner", "stale_source", "existing_second_output",
])
def test_bad_gate_fails_before_any_csv_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    paths = _fixture(tmp_path)
    output_dir = tmp_path / "out"
    if change == "missing_review":
        paths["review"].unlink()
    elif change == "wrong_reviewer":
        review = json.loads(paths["review"].read_text())
        review["reviewer_task"] = "/root/implementer"
        _write(paths["review"], review)
    elif change == "stale_config":
        paths["contract"].write_text(paths["contract"].read_text() + "\n", encoding="utf-8")
    elif change == "stale_scanner":
        review = json.loads(paths["review"].read_text())
        review["bindings"]["scanner_sha256"] = "0" * 64
        _write(paths["review"], review)
    elif change == "stale_runner":
        review = json.loads(paths["review"].read_text())
        review["bindings"]["runner_sha256"] = "0" * 64
        _write(paths["review"], review)
    elif change == "stale_source":
        contract = json.loads(paths["contract"].read_text())
        Path(contract["sources"][1]["archive_path"]).write_bytes(b"changed")
    else:
        output_dir.mkdir()
        (output_dir / "secondary-metadata.json").write_text("user data", encoding="utf-8")
    monkeypatch.setattr(scanner, "_scan_member", lambda *_a, **_k: pytest.fail("CSV row opened"))
    with pytest.raises((FileNotFoundError, ValueError, FileExistsError)):
        _invoke(paths, output_dir)


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_output_directory_must_be_real_directory_before_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    paths = _fixture(tmp_path)
    output_dir = tmp_path / "out"
    if kind == "file":
        output_dir.write_text("user data", encoding="utf-8")
    else:
        target = tmp_path / "redirected"
        target.mkdir()
        try:
            output_dir.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks are unavailable on this host.")
    monkeypatch.setattr(scanner, "_scan_member", lambda *_a, **_k: pytest.fail("CSV row opened"))
    with pytest.raises(ValueError, match="file or symlink"):
        _invoke(paths, output_dir)
