"""SYNTHETIC_CORRECTNESS_ONLY adapted suffix admission must be separate."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-transfer-integration-builder-v1")
)
from test_support import prefix_module

prefix = prefix_module()


def test_distinct_suffix_adapter_and_no_zero_shot_kind_alias():
    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    assert module.ROLE == "adapted_suffix"
    assert module.NEURAL_KIND == "native_prefix_matched_transfer_inference_v4"
    assert (
        prefix.selected_kind(prefix.PrefixConfig(method="cf_random_frozen", family="cf"))
        == "native_cf_random_control_encoder_v1"
    )


@pytest.mark.parametrize("damage", ["dual_conflict", "dual_type", "context_only"])
def test_context_never_substitutes_for_forecast_label_mask(damage):
    import numpy as np

    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    arrays = {"observed": np.ones((2, 96, 4), bool)}
    if damage != "context_only":
        arrays["target_observed"] = np.ones((2, 3), bool)
        arrays["y_observed"] = (
            np.zeros((2, 3), bool) if damage == "dual_conflict" else np.ones((2, 3), int)
        )
    with pytest.raises(ValueError):
        module._label_masks(arrays, 2)


def test_equal_forecast_aliases_preserve_provenance():
    import numpy as np

    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    mask = np.ones((2, 3), bool)
    actual, names = module._label_masks(
        {"observed": mask.copy(), "target_observed": mask.copy()}, 2
    )
    np.testing.assert_array_equal(actual, mask)
    assert names == ["target_observed", "observed"]


def test_missing_approval_precedes_all_numeric_decode():
    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    with pytest.raises(ValueError):
        module.verify_review(
            {},
            {"role": "adapted_suffix"},
            "APPROVED_PREFIX_SUFFIX_ASSESSMENT_EXECUTION",
            "prefix_suffix_assessment_execution",
        )


def test_enumerated_references_receive_no_assessment_values_and_use_pinned_offsets():
    import numpy as np

    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    data = {"x": np.arange(2 * 96 * 4, dtype=np.float32).reshape(2, 96, 4)}
    for name, expected in [
        ("persistence", np.repeat(data["x"][:, -1, 0, None], 3, axis=1)),
        ("seasonal24", data["x"][:, -25 + np.array([1, 3, 6]), 0]),
    ]:
        actual = module._forecast({"kind": name}, None, data, "cpu")
        np.testing.assert_array_equal(actual, np.repeat(expected[..., None], 5, axis=-1))


def test_native_daily_floor_unsupported_support_never_fabricates_intervals():
    import numpy as np

    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    prediction = np.zeros((17, 3, 5))
    data = {
        "observed": np.ones((17, 3), bool),
        "target_dates": np.full((17, 3), "2026-02-11"),
        "deployment": np.full(17, "synthetic-site"),
    }
    metric = module.native_scores(
        prediction,
        np.zeros((17, 3)),
        data["observed"],
        data["target_dates"],
        data["deployment"],
        minimum_daily_rows=18,
    )
    assert metric["primary_pinball_db"] is None
    uncertainty, draws = module.comparison._bootstrap(
        data, ["constant"], {"constant": {"metrics": metric}}
    )
    assert draws is None and uncertainty["interval_status"] == "NOT_ASSESSABLE"


def test_target_free_neural_forecast_call_is_exactly_four_inputs(monkeypatch):
    import numpy as np

    module = importlib.import_module("marine_echo.evaluation.native_prefix_suffix_assessment_v1")
    data = {
        k: object()
        for k in ("x", "context_observed", "metadata", "query", "targets", "future", "observed")
    }
    calls = []

    class Spy:
        def __init__(self, weights, device):
            assert device == "cpu"

        def forecast(self, *inputs):
            calls.append(inputs)
            return np.zeros((1, 3, 5))

    monkeypatch.setattr(module.prefix, "PrefixPredictor", Spy)
    module._forecast({"kind": module.NEURAL_KIND}, b"SYNTHETIC", data, "cpu")
    assert calls == [tuple(data[k] for k in ("x", "context_observed", "metadata", "query"))]
