"""Explicit opt-in test of an actual training-period AZFP file."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest


def test_real_azfp_selected_diagnostic_hour_frequency_and_time() -> None:
    root_text = os.environ.get("MARINE_ECHO_REAL_AZFP_HOUR")
    if root_text is None:
        pytest.skip("Set MARINE_ECHO_REAL_AZFP_HOUR to an extracted real AZFP hour directory.")
    import echopype as ep

    root = Path(root_text)
    ed = ep.open_raw(root / "20021614.01A", sonar_model="AZFP", xml_path=root / "20021600.XML")
    beam = ed["Sonar/Beam_group1"]
    assert beam["frequency_nominal"].values.tolist() == [
        38000.0,
        125000.0,
        200000.0,
        455000.0,
    ]
    assert beam.sizes["ping_time"] > 0
    assert str(beam["ping_time"].values[0]).startswith("2020-02-16T14:")


def test_real_archive_records_duplicate_chunks_and_missing_hour_stems() -> None:
    path_text = os.environ.get("MARINE_ECHO_REAL_AZFP_ZIP")
    if path_text is None:
        pytest.skip("Set MARINE_ECHO_REAL_AZFP_ZIP to the verified local archive.")
    with zipfile.ZipFile(path_text) as archive:
        names = set(archive.namelist())
    assert "20021700.01A" in names and "20021700.01B" in names
    assert "20021821.01A" in names
    assert "20021822.01A" not in names and "20021822.01B" not in names
