"""Synthetic matching regressions and published TEOS-10 numerical references."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from marine_echo.data.environment import (
    ProfileRecord,
    match_profile,
    parse_iridium_position,
    seawater_properties,
)

T = datetime(2020, 3, 1, tzinfo=UTC)


def record(**changes):
    values = {
        "observed_at": T,
        "available_at": T,
        "latitude": 85.0,
        "longitude": 30.0,
        "quality_verified": True,
        "source_id": "synthetic-profile",
    }
    return ProfileRecord(**(values | changes))


def test_future_and_retrospective_records_cannot_change_predictor_match():
    old = record(observed_at=T - timedelta(hours=12))
    late = record(available_at=T + timedelta(days=1))
    future = record(observed_at=T + timedelta(hours=1), available_at=T + timedelta(hours=1))
    assert match_profile([old, late, future], T, 85, 30, 24, 5) == old


@pytest.mark.parametrize(
    "change",
    [
        {"quality_verified": False},
        {"available_at": None},
        {"observed_at": T - timedelta(days=2)},
        {"longitude": 100.0},
    ],
)
def test_unknown_quality_availability_old_or_remote_fails_closed(change):
    with pytest.raises(ValueError, match="compatible"):
        match_profile([record(**change)], T, 85, 30, 24, 5)


def test_invalid_coordinates_and_naive_dates_rejected():
    with pytest.raises(ValueError):
        record(latitude=float("nan"))
    with pytest.raises(ValueError):
        record(observed_at=T.replace(tzinfo=None))


def test_impossible_availability_and_nonboolean_quality_rejected():
    with pytest.raises(ValueError):
        record(available_at=T - timedelta(seconds=1))
    with pytest.raises(TypeError):
        record(quality_verified="false")


def test_ambiguous_contemporaneous_sources_fail_closed():
    with pytest.raises(ValueError, match="Ambiguous"):
        match_profile([record(source_id="one"), record(source_id="two")], T, 85, 30, 24, 5)


def test_published_teos_pressure_reference_and_inverse():
    # https://www.teos-10.org/pubs/gsw/html/gsw_p_from_z.html
    result = seawater_properties(
        [10, 50, 125], [34.7118, 34.8915, 35.0256], [28.8099, 28.4392, 22.7862], 4, 188
    )
    np.testing.assert_allclose(
        result["pressure_dbar"], [10.055726724518, 50.283543374874, 125.731858435610], atol=1e-9
    )
    assert np.all(result["sound_speed_m_s"] > 1400)
    assert np.all(result["practical_salinity"] < [34.7118, 34.8915, 35.0256])


def test_published_teos_salinity_temperature_and_sound_speed_references():
    # Independent tabulated examples: teos-10.org/pubs/gsw/html/gsw_{z_from_p,
    # SP_from_SA,t_from_CT,sound_speed}.html. Sound-speed example uses different CT.
    depths = [9.9445834469453, 49.7180897012550, 124.2726219409978]
    sa = [34.7118, 34.8915, 35.0256]
    converted = seawater_properties(depths, sa, [28.8099, 28.4392, 22.7862], 4, 188)
    np.testing.assert_allclose(converted["pressure_dbar"], [10, 50, 125], atol=1e-9)
    # The website SP table is v3.05 and differs from the pinned package atlas.
    # Use its shipped MATLAB-reference check fixture, not a generated expectation.
    surface = seawater_properties([0], [34.468236430490606], [20], 11, 142)
    np.testing.assert_allclose(
        surface["practical_salinity"], [34.306287392599714], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        converted["in_situ_temperature_c"],
        [28.785580227725703, 28.432872246163946, 22.810323087627076],
        atol=2e-8,
    )
    speed = seawater_properties(depths, sa, [28.7856, 28.4329, 22.8103], 4, 188)
    np.testing.assert_allclose(
        speed["sound_speed_m_s"],
        [1542.426412426373, 1542.558891663385, 1530.801535436184],
        atol=2e-8,
    )


@pytest.mark.parametrize(
    "depth,salinity,temp",
    [
        ([-10], [34], [-1]),
        ([10], [float("nan")], [-1]),
        ([10, 5], [34, 34], [-1, -1]),
        ([10], [34, 35], [-1]),
    ],
)
def test_invalid_physical_profile_fails_closed(depth, salinity, temp):
    with pytest.raises(ValueError):
        seawater_properties(depth, salinity, temp, 85, 30)


def test_iridium_documented_degrees_minutes_and_signed_longitude():
    line = "1 65 3 2020-8-2 17:10:0 79 6.6 -2 -36.0 0 service"
    stamp, lat, lon = parse_iridium_position(line)
    assert stamp == datetime(2020, 8, 2, 17, 10, tzinfo=UTC)
    assert lat == pytest.approx(79.11)
    assert lon == pytest.approx(-2.6)


@pytest.mark.parametrize(
    "line",
    [
        "0 65 3 2020-8-2 17:10:0 79 6.6 -2 -36.0 0 service",
        "1 65 3 2020-8-2 17:10:0 79 60.6 -2 -36.0 0 service",
        "1 65 3 2020-8-2 17:10:0 79 6.6 -2 36.0 0 service",
        "1 65 3 2020-8-2 17:10:0 79.5 6.6 -2 -36.0 0 service",
    ],
)
def test_iridium_invalid_fix_or_ambiguous_coordinate_rejected(line):
    with pytest.raises(ValueError):
        parse_iridium_position(line)
