"""Independent hand-calculated oracles for the implementation contract."""
import copy
import json
import math
import unittest
from pathlib import Path
from tools.reference_contracts import (
    linear_mean_db, pinball, target_intervals, validate_forecast, validate_task_dag,
)
ROOT = Path(__file__).resolve().parents[1]

class NumericalTests(unittest.TestCase):
    def test_linear_mean_not_arithmetic_db(self):
        self.assertAlmostEqual(linear_mean_db([-10.0, -20.0]), -12.596373105, places=8)
    def test_single_value(self):
        self.assertEqual(linear_mean_db([-70.0]), -70.0)
    def test_large_values_stable(self):
        self.assertEqual(linear_mean_db([4000.0, 4000.0]), 4000.0)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): linear_mean_db([])
    def test_nonfinite_rejected(self):
        for x in [math.nan, math.inf, -math.inf]:
            with self.subTest(x=x), self.assertRaises(ValueError): linear_mean_db([x])
    def test_pinball_underprediction(self):
        self.assertAlmostEqual(pinball(.9, 10., 8.), 1.8)
    def test_pinball_overprediction(self):
        self.assertAlmostEqual(pinball(.9, 8., 10.), .2)
    def test_pinball_invalid_quantile(self):
        with self.assertRaises(ValueError): pinball(1.1, 1., 2.)
    def test_pinball_nonfinite(self):
        with self.assertRaises(ValueError): pinball(.5, math.nan, 1.)
    def test_windows_not_overlapping_past(self):
        windows = target_intervals('2020-03-01T12:00:00Z')
        self.assertEqual(windows[0], ('2020-03-01T12:00:00Z','2020-03-01T13:00:00Z'))
        self.assertEqual(windows[1], ('2020-03-01T14:00:00Z','2020-03-01T15:00:00Z'))
        self.assertEqual(windows[2], ('2020-03-01T17:00:00Z','2020-03-01T18:00:00Z'))
    def test_timezone_required(self):
        with self.assertRaises(ValueError): target_intervals('2020-03-01T12:00:00')
    def test_offset_normalized(self):
        self.assertEqual(target_intervals('2020-03-01T14:00:00+02:00'),target_intervals('2020-03-01T12:00:00Z'))

class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((ROOT/'tests_handoff/fixtures/forecast.synthetic.json').read_text())
    def test_valid_synthetic_contract(self): validate_forecast(self.data)
    def test_crossing_rejected(self):
        self.data['forecasts'][0]['quantiles']['0.05']=100.
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_nan_rejected(self):
        self.data['forecasts'][1]['quantiles']['0.5']=math.nan
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_future_observation_rejected(self):
        self.data['last_observation_at']='2020-03-01T13:00:00Z'
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_wrong_target_window_rejected(self):
        self.data['forecasts'][0]['target_start']='2020-03-01T11:00:00Z'
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_duplicate_horizons_rejected(self):
        self.data['forecasts'][1]=copy.deepcopy(self.data['forecasts'][0])
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_abstention_must_not_invent_values(self):
        self.data['status']='abstained';self.data['reasons']=['All observations missing']
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_valid_abstention(self):
        self.data['status']='abstained';self.data['reasons']=['All observations missing']
        self.data['last_observation_at']=None;self.data['observation_age_hours']=None
        self.data['support_fraction']=0.
        for p in self.data['forecasts']:
            p['quantiles']={k:None for k in p['quantiles']}
        validate_forecast(self.data)
    def test_synthetic_badge_cannot_be_removed(self):
        self.data['mode']='live_cpu'
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_age_mismatch_rejected(self):
        self.data['observation_age_hours']=0.
        with self.assertRaises(ValueError): validate_forecast(self.data)
    def test_physical_unit_requires_calibration(self):
        self.data['units']='dB re 1 m^-1'
        with self.assertRaises(ValueError): validate_forecast(self.data)

class DagTests(unittest.TestCase):
    def test_actual_graph(self):
        validate_task_dag(json.loads((ROOT/'orchestration/tasks.json').read_text())['tasks'])
    def test_cycle_rejected(self):
        with self.assertRaises(ValueError): validate_task_dag([{'id':'a','depends_on':['b']},{'id':'b','depends_on':['a']}])
    def test_unknown_dependency_rejected(self):
        with self.assertRaises(ValueError): validate_task_dag([{'id':'a','depends_on':['x']}])
    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError): validate_task_dag([{'id':'a','depends_on':[]},{'id':'a','depends_on':[]}])
