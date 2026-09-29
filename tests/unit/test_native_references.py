import numpy as np
import pytest

from marine_echo.training.native_references import feature_matrix, run


def test_reference_features_are_causal_and_masked():
    context = {
        "x": np.ones((2, 96, 4)),
        "observed": np.ones((2, 96, 4), dtype=bool),
        "metadata": np.zeros((2, 4, 10)),
        "y": np.zeros((2, 3)),
    }
    context["observed"][0, 0, 1] = False
    first = feature_matrix(context, 96)
    context["x"][0, 0, 1] = 1e20
    context["y"][:] = 1e20
    np.testing.assert_array_equal(first, feature_matrix(context, 96))


def test_reference_fitting_is_blocked_before_any_corpus_access(tmp_path):
    review = tmp_path / "review.json"
    review.write_text('{"status":"REQUEST_CHANGES"}')
    with pytest.raises(ValueError, match="prefit"):
        run(
            tmp_path / "missing_train.npz",
            tmp_path / "missing_dev.npz",
            tmp_path / "out",
            review,
            "lightgbm",
        )
    assert not (tmp_path / "out").exists()
