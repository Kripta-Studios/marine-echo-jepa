import numpy as np
import pytest

from marine_echo.evaluation.metrics import daily_metrics, interval_widening, paired_blocks, pinball
from marine_echo.features.windows import TrainScaler, past_features, require_training_partition


def test_pinball_hand_calculation_and_abstention_denominator() -> None:
    truth = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    pred = np.full((2, 3, 5), 8.0)
    np.testing.assert_allclose(pinball(truth, pred)[0, 0], [0.1, 0.5, 1, 1.5, 1.9])
    pred[1] = np.nan
    result = daily_metrics(
        truth, pred, np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    )
    assert result["prediction_coverage"] == 0.5
    assert result["eligible_rows"] == 2
    assert result["daily_mean_pinball_db"] is None
    common = daily_metrics(
        truth,
        pred,
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
        shared_support=np.array([True, False]),
    )
    assert common["daily_mean_pinball_db"] == 1


def test_days_have_equal_weight() -> None:
    truth = np.array([[0.0], [0.0], [10.0]])
    pred = np.zeros((3, 1, 5))
    result = daily_metrics(
        truth,
        pred,
        np.array(
            ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-02T00:00"], dtype="datetime64[m]"
        ),
    )
    assert result["daily_mean_pinball_db"] == 2.5


def test_nonnegative_widening_and_partition_gate() -> None:
    truth = np.array([[0.0], [4.0]])
    pred = np.array([[[-1.0, 0.0, 0.0, 0.0, 1.0]], [[-1.0, 0.0, 0.0, 0.0, 1.0]]])
    assert interval_widening(truth, pred, "calibration").tolist() == [3.0]
    with pytest.raises(ValueError):
        interval_widening(truth, pred, "test")


def test_block_draws_preserve_model_pairing() -> None:
    times = np.arange(
        np.datetime64("2020-01-01"), np.datetime64("2020-01-05"), np.timedelta64(1, "h")
    )
    errors = np.stack([np.arange(96), np.arange(96) + 2], axis=1).astype(float)
    chosen, means = paired_blocks(times, errors)
    assert chosen.shape == (2000, 2)
    np.testing.assert_allclose(means[:, 1] - means[:, 0], 2.0)


def test_scaler_and_ssl_cannot_fit_protected_partitions() -> None:
    for split in ("validation", "calibration", "test"):
        with pytest.raises(ValueError):
            TrainScaler().fit(np.ones((2, 2)), split)
        with pytest.raises(ValueError):
            require_training_partition(["train", split])


def test_future_value_mutation_preserves_features() -> None:
    times = np.arange(
        np.datetime64("2020-01-01T00:15"),
        np.datetime64("2020-01-03T00:15"),
        np.timedelta64(15, "m"),
    )
    values = np.arange(len(times), dtype=float)[:, None]
    cutoff = np.datetime64("2020-01-02T00:00")
    original = past_features(values, times, cutoff)
    values[times > cutoff] = -999999
    np.testing.assert_array_equal(original, past_features(values, times, cutoff))
