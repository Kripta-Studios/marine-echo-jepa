"""Synthetic range-overlap checks; these fixtures are not calibrated field evidence."""

import numpy as np
import pytest

from marine_echo.data.range_grid import regrid_linear_sv


def test_linear_overlap_conserves_integrated_backscatter():
    values = np.array([[1.0, 9.0, 2.0]])
    source = np.array([0.0, 1.0, 3.0, 4.0])
    result = regrid_linear_sv(values, np.ones_like(values, dtype=bool), source, np.arange(5.0))
    np.testing.assert_allclose(result.sv_linear, [[1.0, 9.0, 9.0, 2.0]])
    assert result.valid_mask.all()
    assert np.sum(result.sv_linear) == np.sum(values * np.diff(source))
    coarse = regrid_linear_sv(
        values, np.ones_like(values, dtype=bool), source, np.array([0.0, 4.0])
    )
    assert coarse.sv_linear[0, 0] == 5.25
    assert 10 * np.log10(coarse.sv_linear[0, 0]) != np.mean(10 * np.log10(values))


def test_partial_support_is_recorded_and_never_extrapolated():
    values = np.array([[2.0, np.nan], [4.0, 8.0]])
    mask = np.array([[True, False], [True, True]])
    result = regrid_linear_sv(values, mask, np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0]))
    np.testing.assert_allclose(result.support_fraction, [[0.5, 0.0], [0.5, 0.5]])
    assert not result.supported_range.any()
    assert not result.valid_mask.any()
    assert np.isnan(result.sv_linear).all()
    internal = regrid_linear_sv(values, mask, np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0]))
    assert internal.supported_range.all()
    np.testing.assert_array_equal(internal.valid_mask, [[False], [True]])
    assert np.isnan(internal.sv_linear[0, 0])
    assert internal.sv_linear[1, 0] == 6.0


def test_channel_geometry_is_independent_and_inputs_are_unchanged():
    values = np.ones((2, 2))
    mask = np.ones_like(values, dtype=bool)
    target = np.array([0.0, 2.0, 4.0])
    low = regrid_linear_sv(values, mask, np.array([0.0, 2.0, 4.0]), target)
    high = regrid_linear_sv(values, mask, np.array([0.0, 1.0, 2.0]), target)
    np.testing.assert_array_equal(low.supported_range, [True, True])
    np.testing.assert_array_equal(high.supported_range, [True, False])
    np.testing.assert_array_equal(values, np.ones((2, 2)))
    assert mask.all()


@pytest.mark.parametrize(
    "edge", [np.array([0.0, 2.0, 1.0]), np.array([0.0, 1.0, np.nan]), np.array([-1.0, 0.0, 1.0])]
)
def test_invalid_geometry_is_rejected(edge):
    with pytest.raises(ValueError):
        regrid_linear_sv(np.ones((1, 2)), np.ones((1, 2), dtype=bool), edge, np.array([0.0, 2.0]))


def test_valid_negative_or_nonfinite_values_and_unbounded_shapes_fail():
    for value in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            regrid_linear_sv(
                np.array([[value]]), np.array([[True]]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
            )
    with pytest.raises(ValueError, match="bounded"):
        regrid_linear_sv(
            np.ones((4097, 1)),
            np.ones((4097, 1), dtype=bool),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
