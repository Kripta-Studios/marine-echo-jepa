"""Pinned Echopype AZFP singleton compatibility on fixed TRAIN source files."""

import hashlib
import json
from pathlib import Path

import echopype as ep
import numpy as np
import pytest
from echopype.convert.api import SONAR_MODELS
from marine_echo.data.candidate_qc import clean_per_ping, sample_edges
from marine_echo.data.range_grid import regrid_linear_sv

from marine_echo.data.azfp_compat import open_raw_azfp_compat

_EXTRACTED = (
    Path(__file__).resolve().parents[2] / "data/raw/pangaea/mosaic_azfp_down_2020_extracted"
)
_XML = _EXTRACTED / "20021600.XML"
_SINGLE = _EXTRACTED / "20030416.01A"
_MULTI = _EXTRACTED / "20030300.01A"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_single_ping_train_file_is_preserved_without_duplication() -> None:
    assert _sha(_SINGLE) == "aa83fe505443235d70d846b898a331c799be9e2fedd7c9fa23f476415993b48a"
    original_parser = SONAR_MODELS["AZFP"]["parser"]
    ed = open_raw_azfp_compat(_SINGLE, _XML)
    beam = ed["Sonar/Beam_group1"]
    vendor = ed["Vendor_specific"]
    assert len(beam.ping_time) == 1
    np.testing.assert_array_equal(beam.frequency_nominal.values, [38000, 125000, 200000, 455000])
    assert beam.backscatter_r.shape[1] == 1
    assert len(vendor.number_of_bins_per_channel) == 4
    assert SONAR_MODELS["AZFP"]["parser"] is original_parser
    assert _sha(_SINGLE) == "aa83fe505443235d70d846b898a331c799be9e2fedd7c9fa23f476415993b48a"


def test_multi_ping_train_file_matches_upstream_exactly() -> None:
    original_parser = SONAR_MODELS["AZFP"]["parser"]
    baseline = ep.open_raw(_MULTI, sonar_model="AZFP", xml_path=_XML)
    converted = open_raw_azfp_compat(_MULTI, _XML)
    for group, fields in {
        "Sonar/Beam_group1": ("ping_time", "frequency_nominal", "backscatter_r"),
        "Vendor_specific": ("number_of_bins_per_channel", "digitization_rate", "lock_out_index"),
    }.items():
        for field in fields:
            np.testing.assert_array_equal(
                baseline[group][field].values, converted[group][field].values
            )
    assert SONAR_MODELS["AZFP"]["parser"] is original_parser


def test_parser_binding_restored_after_open_error(monkeypatch) -> None:
    original_parser = SONAR_MODELS["AZFP"]["parser"]

    def fail(*args, **kwargs):
        assert SONAR_MODELS["AZFP"]["parser"] is not original_parser
        raise RuntimeError("fixture-open-failure")

    monkeypatch.setattr(ep, "open_raw", fail)
    with pytest.raises(RuntimeError, match="fixture-open-failure"):
        open_raw_azfp_compat(_SINGLE, _XML)
    assert SONAR_MODELS["AZFP"]["parser"] is original_parser


def test_singleton_reaches_candidate_calibration_cleaning_and_range_shapes() -> None:
    original_parser = SONAR_MODELS["AZFP"]["parser"]
    root = Path(__file__).resolve().parents[2]
    env = json.loads(
        (
            root.parent
            / "marine-echo-jepa/evidence/continuation/depth_environment_sensitivity.json"
        ).read_text(encoding="utf-8")
    )["nominal"]
    ed = open_raw_azfp_compat(_SINGLE, _XML)
    beam, vendor, platform = ed["Sonar/Beam_group1"], ed["Vendor_specific"], ed["Platform"]
    times_before = beam.ping_time.values.copy()
    counts_before = beam.backscatter_r.values.copy()
    config_before = vendor.number_of_bins_per_channel.values.copy()
    calibrated = ep.calibrate.compute_Sv(ed, env_params=env)
    raw_sv, ranges = calibrated.Sv.values.copy(), calibrated.echo_range.values
    counts = beam.backscatter_r.values
    assert raw_sv.shape[1] == counts.shape[1] == len(beam.ping_time) == 1
    tilt = np.hypot(platform.tilt_x.values, platform.tilt_y.values)
    reasons = np.zeros(raw_sv.shape, dtype=np.uint16)
    reasons[~np.isfinite(raw_sv) | ~np.isfinite(counts)] |= 1
    reasons[ranges < 10.0] |= 2
    reasons[counts >= 65535] |= 4
    reasons[:, ~np.isfinite(tilt) | (tilt >= 30), :] |= 8
    calibrated["Sv"] = calibrated.Sv.where(reasons == 0)
    cleaned = clean_per_ping(calibrated)
    corrected = cleaned.Sv_corrected.values
    assert corrected.shape == raw_sv.shape
    reasons[~np.isfinite(corrected)] |= 16
    grids = []
    for channel in range(4):
        n = int(vendor.number_of_bins_per_channel.values[channel])
        spacing = (
            env["sound_speed"]
            * float(vendor.number_of_samples_per_average_bin.values[channel])
            / (2 * float(vendor.digitization_rate.values[channel]))
        )
        edges = sample_edges(ranges[channel, 0, :n], spacing)
        grid = regrid_linear_sv(
            10 ** (corrected[channel, :, :n] / 10),
            reasons[channel, :, :n] == 0,
            edges,
            np.arange(0.0, 130.0, 2.0),
        )
        assert grid.sv_linear.shape == grid.valid_mask.shape == (1, 64)
        grids.append(grid)
    assert len(grids) == 4
    np.testing.assert_array_equal(beam.ping_time.values, times_before)
    np.testing.assert_array_equal(beam.backscatter_r.values, counts_before)
    np.testing.assert_array_equal(vendor.number_of_bins_per_channel.values, config_before)
    assert SONAR_MODELS["AZFP"]["parser"] is original_parser
