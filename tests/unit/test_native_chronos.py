import numpy as np

from marine_echo.training.native_chronos import contexts, native_inputs, predict_rows


def test_chronos_uses_matched_context_and_masks_only():
    data = {
        "x": np.ones((2, 96, 4)),
        "observed": np.ones((2, 96, 4), dtype=bool),
        "y": np.zeros((2, 3)),
    }
    data["observed"][0, -1, 1] = False
    inputs = contexts(data, 96)
    data["y"][:] = 1e10
    data["x"][0, -1, 1] = 1e10
    np.testing.assert_array_equal(inputs, contexts(data, 96))
    assert np.isnan(inputs[0, 1, -1])
    assert contexts(data, 24).shape == (2, 4, 24)


def test_chronos_horizons_are_native_one_three_six():
    class Pipeline:
        def predict_quantiles(self, **kwargs):
            assert not kwargs["cross_learning"]
            assert kwargs["context_length"] == 96
            prediction = np.broadcast_to(np.arange(6)[None, :, None], (4, 6, 5))
            return [prediction, prediction], None

    output = predict_rows(Pipeline(), np.ones((2, 4, 96)), 96)
    assert output.shape == (2, 3, 5)
    np.testing.assert_array_equal(output[0, :, 0], [0, 2, 5])


def test_chronos_known_metadata_covariates_preserve_native_product():
    metadata = np.zeros((2, 4, 10), dtype=np.float32)
    metadata[:, :, 4] = 230 / 250
    data = {
        "x": np.ones((2, 96, 4)),
        "observed": np.ones((2, 96, 4), dtype=bool),
        "metadata": metadata,
        "y": np.zeros((2, 3)),
    }
    inputs = native_inputs(data, 96)
    assert inputs[0]["target"].shape == (1, 96)
    np.testing.assert_array_equal(
        inputs[0]["future_covariates"]["query_lower_bound"], np.full(6, 230 / 250, dtype=np.float32)
    )
    assert not any("acoustic" in key for key in inputs[0]["future_covariates"])
    data["y"][:] = 1e20
    other = native_inputs(data, 96)
    np.testing.assert_array_equal(inputs[0]["target"], other[0]["target"])
