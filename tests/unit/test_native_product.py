import numpy as np
import pytest

from marine_echo.evaluation.native_product import native_scores


def test_equal_deployment_weighting_and_daily_floor():
    targets = np.zeros((60, 3))
    pred = np.concatenate([np.ones((20, 3, 5)), np.ones((40, 3, 5)) * 3])
    dates = np.full((60, 3), "2023-01-01")
    deployment = np.asarray(["first"] * 20 + ["second"] * 40)
    result = native_scores(pred, targets, np.ones_like(targets, dtype=bool), dates, deployment)
    assert result["primary_pinball_db"] == 1.0
    assert result["scored_per_horizon"] == [60, 60, 60]
    # Missing support is explicit, never a zero-valued metric.
    result = native_scores(
        pred[:10], targets[:10], np.ones((10, 3), dtype=bool), dates[:10], deployment[:10]
    )
    assert result["primary_pinball_db"] is None


def test_geometry_is_not_normalized_by_depth_or_score():
    targets = np.ones((18, 3)) * -70
    prediction = np.repeat(targets[..., None], 5, axis=-1)
    result = native_scores(
        prediction,
        targets,
        np.ones((18, 3), dtype=bool),
        np.full((18, 3), "2023-01-01"),
        np.full(18, "native230"),
    )
    assert result["primary_pinball_db"] == 0
    with pytest.raises(ValueError, match="shape"):
        native_scores(
            prediction[:-1],
            targets,
            np.ones((18, 3), dtype=bool),
            np.full((18, 3), "2023-01-01"),
            np.full(18, "native230"),
        )


@pytest.mark.parametrize("invalid", ["forecast", "observed_target"])
def test_invalid_forecast_or_observed_target_must_not_change_assessment_support(invalid):
    target = np.full((20, 3), -70.0)
    prediction = np.repeat(target[..., None], 5, axis=-1)
    if invalid == "forecast":
        prediction[0, 0, 0] = np.nan
    else:
        target[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native_scores(
            prediction,
            target,
            np.ones((20, 3), bool),
            np.full((20, 3), "2023-01-01"),
            np.full(20, "native230"),
        )
