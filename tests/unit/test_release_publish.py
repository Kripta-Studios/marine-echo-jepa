from pathlib import Path

import pytest

from marine_echo.serving.release import build_diagnostic


def test_release_rejects_existing_directory_before_reading_sources(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    (output / "stale.js").write_text("old")
    with pytest.raises(FileExistsError):
        build_diagnostic(tmp_path / "missing-source", output)
    assert (output / "stale.js").read_text() == "old"


def test_failed_build_does_not_publish_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "published"
    with pytest.raises(FileNotFoundError):
        build_diagnostic(tmp_path / "missing-source", output)
    assert not output.exists()
