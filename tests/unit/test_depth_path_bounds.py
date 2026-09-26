"""Analytic checks for the diagnostic piecewise physical integrator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from echopype.utils import uwa


def test_vector_absorption_matches_upstream_scalar_cases():
    assay = module()
    temperatures = np.array([-1.93, -1.0, 0.0, 1.78])
    salinities = np.array([0.0, 16.0, 33.0, 35.0])
    pressures = np.array([0.0, 10.0, 60.0, 132.0])
    actual = assay.vector_absorption(temperatures, salinities, pressures)
    expected = np.stack(
        [
            uwa.calc_absorption(
                frequency=assay.FREQUENCIES,
                temperature=t,
                salinity=s,
                pressure=p,
                formula_source="AZFP",
            )
            for t, s, p in zip(temperatures, salinities, pressures)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def module():
    path = Path(__file__).resolve().parents[2] / "tools/depth_environment_sensitivity.py"
    spec = importlib.util.spec_from_file_location("depth_assay", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_homogeneous_time_and_absorption():
    assay = module()
    edges = np.array([0.0, 10.0, 20.0])
    speed = np.array([1500.0, 1500.0])
    alpha = np.array([[0.01], [0.01]])
    distance = assay.distance_for_time(3.0, 0.01, edges, speed, 1500.0)
    assert distance == pytest.approx(15.0)
    assert assay.absorption_integral(
        3.0, distance, edges, alpha, np.array([0.01])
    ) == pytest.approx([0.15])


def test_two_layers_and_global_fallback():
    assay = module()
    edges = np.array([0.0, 10.0, 20.0])
    # 5m in the first layer +10m second layer +3m outside measured depth.
    distance = assay.distance_for_time(
        5.0, 5 / 1000 + 10 / 2000 + 3 / 1500, edges, np.array([1000.0, 2000.0]), 1500.0
    )
    assert distance == pytest.approx(18.0)
    integral = assay.absorption_integral(
        5.0, distance, edges, np.array([[0.01], [0.02]]), np.array([0.03])
    )
    assert integral == pytest.approx([0.34])


def test_boundary_and_zero_time():
    assay = module()
    assert assay.distance_for_time(
        10.0, 0.005, np.array([0.0, 10.0, 20.0]), np.array([1000.0, 2000.0]), 1500.0
    ) == pytest.approx(10.0)
    assert assay.distance_for_time(0.0, 0.0, np.array([0.0, 10.0]), np.array([1000.0]), 1500.0) == 0
