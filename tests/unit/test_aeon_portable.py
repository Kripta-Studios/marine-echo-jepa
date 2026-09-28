"""Portable release integrity and isolation checks."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from marine_echo.serving.aeon_portable import (
    _write_final_documents,
    build_aeon_portable,
    verify_package,
)


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
    readme = (package / "README_AEON_RESEARCH.md").read_text()
    assert "development-only" in readme.lower()
    assert "provenance/scaling_development/outcome-review.json" in readme
    assert "provenance/expanded_train_development/outcome-review.json" in readme
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


def test_final_cards_are_sourced_and_retain_review_pending_label(tmp_path: Path) -> None:
    study = {
        "source": {"archive_sha256": "a" * 64, "publisher": "Fixture publisher"},
        "target": {"unit": "source-reported Sv dB", "product": "60minFullDepth"},
        "retrospective_test": {
            "issued_rows": 1216, "eligible_days_per_horizon": [49, 50, 51],
            "test_score_sha256": "b" * 64,
            "test_outcome_review_sha256": "c" * 64,
            "models": {"core_direct_equal_three_seed_ensemble": {
                "raw_primary_daily_mean_pinball_db": 0.56},
                "core_ema_equal_three_seed_ensemble": {
                    "raw_primary_daily_mean_pinball_db": 0.53}},
        },
    }
    _write_final_documents(tmp_path, study)
    assert "0.5300" in (tmp_path / "REPORT.md").read_text()
    assert "release review pending" in (tmp_path / "README_RUN.md").read_text().lower()
    assert "not independently field verified" in (tmp_path / "DATA_CARD.md").read_text().lower()
    assert "post hoc" in (tmp_path / "MODEL_CARD.md").read_text().lower()
    assert (tmp_path / "MARINE_DATA_REQUEST.md").is_file()
