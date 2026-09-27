"""Portable release integrity and isolation checks."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from marine_echo.serving.aeon_portable import build_aeon_portable, verify_package


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_package_has_exact_inventory_and_archive(tmp_path: Path) -> None:
    source = tmp_path / "wheels"
    source.mkdir()
    (source / "dummy-1.0-py3-none-any.whl").write_bytes(b"wheel")
    old_sums = tmp_path / "old-SHA256SUMS"
    old_sums.write_text(
        f"{_digest(source / 'dummy-1.0-py3-none-any.whl')}  wheelhouse/dummy-1.0-py3-none-any.whl\n"
    )
    payload = tmp_path / "payload"
    (payload / "artifacts").mkdir(parents=True)
    (payload / "artifacts/catalog.json").write_text("{}")
    result = build_aeon_portable(
        root=tmp_path, output=tmp_path / "portable", wheelhouse=source,
        upstream_sums=old_sums, payload=payload,
    )
    package = Path(result["output"])
    assert result["archive_sha256"] == _digest(Path(result["archive"]))
    assert verify_package(package) == result["asset_count"]
    assert (package / "Run-AEON-Research.ps1").is_file()
    assert (package / "src/marine_echo/serving/api.py").is_file()
    assert not (package / "src/marine_echo/training").exists()
    assert "development-only" in (package / "README_AEON_RESEARCH.md").read_text().lower()
    with zipfile.ZipFile(result["archive"]) as archive:
        assert "portable/SHA256SUMS" in archive.namelist()
    (package / "artifacts/catalog.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="digest"):
        verify_package(package)


def test_rejects_unverified_wheel_and_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "wheels"
    source.mkdir()
    (source / "dummy.whl").write_bytes(b"unverified")
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("0" * 64 + "  wheelhouse/dummy.whl\n")
    payload = tmp_path / "payload"
    payload.mkdir()
    with pytest.raises(ValueError, match="wheel"):
        build_aeon_portable(tmp_path, tmp_path / "out", source, sums, payload)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        build_aeon_portable(tmp_path, tmp_path / "out", source, sums, payload)
