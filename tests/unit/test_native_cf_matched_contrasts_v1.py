"""SYNTHETIC_CORRECTNESS_ONLY in-memory CF numerical contract checks."""

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER if BUILDER.name == "marine-echo-jepa" else BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
import marine_echo
import marine_echo.evaluation

marine_echo.__path__ = [str(MAIN / "src/marine_echo")]
marine_echo.evaluation.__path__ = [str(MAIN / "src/marine_echo/evaluation")]
from marine_echo.evaluation import native_suffix_reconstruction_v1 as reconstruction

if BUILDER != MAIN:
    marine_echo.evaluation.__path__.insert(0, str(BUILDER / "src/marine_echo/evaluation"))
api = importlib.import_module("marine_echo.evaluation.native_cf_matched_contrasts_v1")


def fixture(*, identical=False, zero_span=False):
    """Unequal deployment/date/horizon counts and a preserved calendar gap."""
    group = [
        ("A", "2026-01-01", 18),
        ("A", "2026-01-02", 24),
        ("A", "2026-01-06", 20),
        ("B", "2026-01-01", 40),
        ("B", "2026-01-02", 18),
        ("B", "2026-01-07", 22),
    ]
    deps, dates, truth, masks = [], [], [], []
    for g, (dep, day, count) in enumerate(group):
        deps.extend([dep] * count)
        dates.extend([["2026-01-01" if zero_span else day] * 3] * count)
        truth.extend([[-50.0 + g + h for h in range(3)]] * count)
        observed = np.ones((count, 3), bool)
        if g == 0:
            observed[:6, 2] = False
        if g == 3:
            observed[:25, 1] = False
        masks.extend(observed)
    n = len(deps)
    query = np.zeros((n, 3, 10), np.float64)
    query[..., 0], query[..., 1], query[..., 2] = 38000 / 455000, 1, 1
    query[..., 4], query[..., 9] = 225 / 250, [1, 3, 6]
    base = {
        "targets": np.asarray(truth, np.float64),
        "observed": np.array(masks),
        "deployment": np.asarray(deps),
        "row_id": np.asarray([f"row-{i:04d}" for i in range(n)]),
        "target_dates": np.asarray(dates),
        "query": query,
        "cutoff": np.arange(n, dtype=np.int64),
    }
    offsets = {
        "learned_frozen": [-6, 0, 6],
        "learned_full": [-3, 0, 3],
        "random_frozen": [2, 4, 6],
        "scratch": [6, 5, 4],
        "lightgbm": [1],
    }
    records = []
    for name, spec in api.prespecified_specs().items():
        arrays = {k: v.copy() for k, v in base.items()}
        offset = (
            0
            if identical
            else offsets[spec["arm"]][
                [7, 13, 23].index(spec["seed"]) if spec["arm"] != "lightgbm" else 0
            ]
        )
        arrays["predictions"] = np.repeat((arrays["targets"] + offset)[..., None], 5, -1)
        records.append(
            {
                "name": name,
                "spec": spec,
                "arrays": arrays,
                "lineage": {
                    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                    "source": "caller-authored-unverified",
                },
            }
        )
    return records


def run(records, **kwargs):
    return api.compute_cf_matched_contrasts(
        records, role="development", original_campaign_modified=False, **kwargs
    )


def test_nonlinear_seed_score_mean_is_not_forecast_ensemble_score():
    result = run(fixture())
    statistics = result["arm_statistics"]["learned_frozen"]
    assert statistics["seed_score_mean_db"] == pytest.approx(2.0)
    ensemble = result["ensembles"]["cf_learned_frozen_forecast_mean_ensemble"]
    assert ensemble["metrics"]["primary_pinball_db"] == pytest.approx(0.0)
    assert ensemble["metrics"]["primary_pinball_db"] != statistics["seed_score_mean_db"]


def test_every_seed_direction_sd_and_independent_hierarchical_arithmetic():
    result = run(fixture())
    assert result["extension"] == "CF_MATCHED_CONTROLS" and len(result["raw_records"]) == 13
    expected = {
        "learned_frozen": [3, 0, 3],
        "learned_full": [1.5, 0, 1.5],
        "random_frozen": [1, 2, 3],
        "scratch": [3, 2.5, 2],
    }
    for arm, values in expected.items():
        stats = result["arm_statistics"][arm]
        assert stats["seeds"] == [7, 13, 23]
        assert [s["primary_pinball_db"] for s in stats["seed_scores"]] == pytest.approx(values)
        assert stats["seed_score_mean_db"] == pytest.approx(np.mean(values))
        assert stats["seed_score_sample_sd_db"] == pytest.approx(np.std(values, ddof=1))
    paired = result["paired_seed_contrasts"]["learned_frozen_minus_random_frozen"]
    assert [x["difference_db"] for x in paired["per_seed"]] == pytest.approx([2, -2, 0])
    assert paired["mean_difference_db"] == 0
    assert result["paired_seed_contrasts"]["learned_full_minus_scratch"][
        "mean_difference_db"
    ] == pytest.approx(-1.5)
    for arm in ("learned_frozen", "learned_full"):
        assert result["ensemble_minus_lightgbm"]["cf_" + arm + "_forecast_mean_ensemble"][
            "difference_db"
        ] == pytest.approx(-0.5)
    assert (
        result["scientific_claim"] is None
        and result["lineage_status"] == "CALLER_AUTHORED_UNVERIFIED"
    )


def test_reordering_and_masked_fills_do_not_change_scores_or_bootstrap():
    records = fixture()
    first = run(records)
    shuffled = copy.deepcopy(records)
    for i, record in enumerate(shuffled):
        order = np.random.default_rng(i).permutation(len(record["arrays"]["targets"]))
        record["arrays"] = {k: v[order] for k, v in record["arrays"].items()}
        record["arrays"]["targets"][~record["arrays"]["observed"]] = np.nan if i % 2 else np.inf
    second = run(shuffled[::-1])
    for key in ("arm_statistics", "paired_seed_contrasts", "ensemble_minus_lightgbm", "bootstrap"):
        assert first[key] == second[key]
    assert first["common_support"]["row_id"].tolist() == second["common_support"]["row_id"].tolist()


def test_independent_daily_weights_unequal_counts_and_canonical_raw_retention():
    records = fixture()
    first = records[0]
    # Error varies by deployment/day/horizon; a row-weighted mean differs.
    delta = np.linspace(-8, 1, len(first["arrays"]["targets"]))[:, None]
    first["arrays"]["predictions"] = np.repeat(
        (first["arrays"]["targets"] + delta)[..., None], 5, -1
    )
    result = run(records)
    a = first["arrays"]
    losses = 0.5 * np.abs(delta[:, 0])
    deployment_scores = []
    for dep in ["A", "B"]:
        horizons = []
        for h in range(3):
            daily = []
            for day in np.unique(a["target_dates"][:, h]):
                mask = (
                    (a["deployment"] == dep)
                    & (a["target_dates"][:, h] == day)
                    & a["observed"][:, h]
                )
                if mask.sum() >= 18:
                    daily.append(losses[mask].mean())
            horizons.append(np.mean(daily))
        deployment_scores.append(np.mean(horizons))
    expected = np.mean(deployment_scores)
    actual = result["records"][first["name"]]["metrics"]["primary_pinball_db"]
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(losses.mean())
    np.testing.assert_array_equal(
        result["raw_records"][first["name"]]["arrays"]["predictions"], a["predictions"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "duplicate",
        "seed",
        "backbone",
        "mode",
        "exposure",
        "head",
        "history",
        "extraction",
    ],
)
def test_prespecified_inventory_cannot_substitute_or_drop_records(damage):
    records = fixture()
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records[-1] = records[0]
    else:
        key, value = {
            "seed": ("seed", 13),
            "backbone": ("backbone", "SharedTemporalEncoder"),
            "mode": ("mode", "direct_end_to_end"),
            "exposure": ("supervised_updates", 3000),
            "head": ("head_policy", "inherited_head"),
            "history": ("history", 24),
            "extraction": ("forecast_extraction", "sampled_SSL_crops"),
        }[damage]
        records[0]["spec"][key] = value
    with pytest.raises(ValueError):
        run(records)


@pytest.mark.parametrize(
    "damage",
    [
        "row",
        "duplicate",
        "mask",
        "context_mask",
        "truth",
        "dates",
        "query",
        "cutoff",
        "nonfinite",
        "mask_dtype",
        "order",
        "geometry",
        "role",
    ],
)
def test_exact_common_support_and_numeric_guards(damage):
    records = fixture()
    a = records[1]["arrays"]
    if damage == "row":
        a["row_id"][0] = "foreign"
    elif damage == "duplicate":
        a["row_id"][1], a["deployment"][1] = a["row_id"][0], a["deployment"][0]
    elif damage == "mask":
        a["observed"][0, 0] = False
    elif damage == "context_mask":
        a["context_observed"] = np.ones((len(a["targets"]), 96, 4), bool)
        del a["observed"]
    elif damage == "truth":
        a["targets"][0, 0] += 1
    elif damage == "dates":
        a["target_dates"][0, 0] = "2026-02-01"
    elif damage == "query":
        a["query"][0, 0, 6] = 1
    elif damage == "cutoff":
        a["cutoff"][0] += 1
    elif damage == "nonfinite":
        a["predictions"][0, 0, 0] = np.nan
    elif damage == "mask_dtype":
        a["observed"] = a["observed"].astype(np.int8)
    elif damage == "order":
        a["predictions"][0, 0, 0] += 1
    elif damage == "geometry":
        for r in records:
            r["arrays"]["query"][..., 4] = 200 / 250
    else:
        a["corpus_role"] = np.array("final_test")
    with pytest.raises(ValueError):
        run(records)


@pytest.mark.parametrize("role", ["final_test", "adapted_suffix", "prefix", "future", "train"])
def test_only_development_without_campaign_edits(role):
    with pytest.raises(ValueError):
        api.compute_cf_matched_contrasts(fixture(), role=role, original_campaign_modified=False)
    with pytest.raises(ValueError):
        api.compute_cf_matched_contrasts(
            fixture(), role="development", original_campaign_modified=True
        )


def test_identical_contrasts_zero_and_unsupported_or_zero_span_null():
    result = run(fixture(identical=True))
    for contrast in result["paired_seed_contrasts"].values():
        assert contrast["mean_difference_db"] == 0 and contrast["mean_difference_ci95_db"] == [0, 0]
        assert all(r["interval95_db"] == [0, 0] for r in contrast["per_seed"])
    for contrast in result["ensemble_minus_lightgbm"].values():
        assert contrast["difference_db"] == 0 and contrast["interval95_db"] == [0, 0]
    zero = run(fixture(identical=True, zero_span=True))
    assert zero["bootstrap"]["interval_status"] == "NOT_ASSESSABLE"
    assert all(x["interval95_db"] is None for x in zero["ensemble_minus_lightgbm"].values())
    records = fixture()
    for record in records:
        record["arrays"]["observed"][:, 2] = False
    unsupported = run(records)
    assert unsupported["status"] == "NOT_ASSESSABLE"
    assert all(
        a["seed_score_mean_db"] is None and a["seed_score_sample_sd_db"] is None
        for a in unsupported["arm_statistics"].values()
    )
    assert all(
        p["mean_difference_db"] is None and p["mean_difference_ci95_db"] is None
        for p in unsupported["paired_seed_contrasts"].values()
    )


def test_no_mutation_global_rng_access_grant_or_io(monkeypatch):
    records = fixture()
    before = copy.deepcopy(records)
    rng = np.random.get_state()

    def forbidden(*args, **kwargs):
        pytest.fail("Pure numerical function performed loading or gated assessment")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(reconstruction, "reconstruct", forbidden)
    result = run(records)
    after = np.random.get_state()
    assert rng[0] == after[0] and rng[2:] == after[2:]
    np.testing.assert_array_equal(rng[1], after[1])
    for old, new in zip(before, records, strict=True):
        assert old["spec"] == new["spec"] and old["lineage"] == new["lineage"]
        for key in old["arrays"]:
            np.testing.assert_array_equal(old["arrays"][key], new["arrays"][key])
    result["raw_records"][records[0]["name"]]["arrays"]["targets"][0, 0] += 1
    np.testing.assert_array_equal(before[0]["arrays"]["targets"], records[0]["arrays"]["targets"])
    assert (
        result["access_authority"] is False and result["independent_scientific_approval"] is False
    )


@pytest.mark.parametrize(
    "damage",
    [
        "head_seed",
        "cf_width",
        "cf_latent",
        "cf_blocks",
        "zero_ssl",
        "float_exposure",
        "method",
        "reference_seed",
    ],
)
def test_typed_spec_dimensions_and_control_identity_are_exact(damage):
    records = fixture()
    spec = records[-1]["spec"] if damage == "reference_seed" else records[0]["spec"]
    key, value = {
        "head_seed": ("head_seed", 7),
        "cf_width": ("cf_width", 192),
        "cf_latent": ("cf_latent", 64),
        "cf_blocks": ("cf_blocks", 4),
        "zero_ssl": ("zero_ssl", True),
        "float_exposure": ("supervised_updates", 2000.0),
        "method": ("method", "direct"),
        "reference_seed": ("seed", 23),
    }[damage]
    spec[key] = value
    with pytest.raises(ValueError):
        run(records)


@pytest.mark.parametrize(
    "damage",
    [
        "truth_nonfinite",
        "dates_missing",
        "dates_type",
        "identity_type",
        "query_horizon",
        "query_frequency",
        "cutoff_bool",
        "cutoff_negative",
        "shape",
        "integer_forecasts",
        "future_array",
        "dual_mask",
    ],
)
def test_finite_native_saved_array_schema_is_strict(damage):
    records = fixture()
    arrays = records[0]["arrays"]
    if damage == "truth_nonfinite":
        arrays["targets"][0, 0] = np.inf
    elif damage == "dates_missing":
        arrays["target_dates"][0, 0] = ""
    elif damage == "dates_type":
        arrays["target_dates"] = arrays["target_dates"].astype("S10")
    elif damage == "identity_type":
        arrays["row_id"] = arrays["row_id"].astype("S10")
    elif damage == "query_horizon":
        arrays["query"][..., 9] = [1, 2, 6]
    elif damage == "query_frequency":
        arrays["query"][..., 0] = 455000 / 455000
    elif damage == "cutoff_bool":
        arrays["cutoff"] = arrays["cutoff"].astype(bool)
    elif damage == "cutoff_negative":
        arrays["cutoff"][0] = -1
    elif damage == "shape":
        arrays["predictions"] = arrays["predictions"][..., :4]
    elif damage == "integer_forecasts":
        arrays["predictions"] = arrays["predictions"].astype(np.int64)
    elif damage == "future_array":
        arrays["future"] = np.zeros((len(arrays["targets"]), 3, 4, 4))
    else:
        arrays["target_observed"] = ~arrays["observed"]
    with pytest.raises(ValueError):
        run(records)


def test_mask_alias_native_geometry_gap_and_caller_lineage_remain_explicit():
    records = fixture(identical=True)
    for record in records:
        record["arrays"]["target_observed"] = record["arrays"].pop("observed")
        record["lineage"] = {
            "claimed_status": "APPROVED",
            "ancestor_path": "Z:/inaccessible/not-opened.pt",
        }
    result = run(records)
    assert len(result["records"]) == 13 and len(result["ensembles"]) == 4
    assert (
        result["records"][records[0]["name"]]["mask_provenance"]["canonicalized_from"]
        == "target_observed"
    )
    for record in result["raw_records"].values():
        assert record["lineage"]["claimed_status"] == "APPROVED"
        assert record["lineage_status"] == "CALLER_AUTHORED_UNVERIFIED"
    assert result["native_geometry"]["unique_products"][0]["lower_m"] == pytest.approx(225)
    assert result["bootstrap"]["blocks"][0]["observed_calendar_block_ids"] == [0, 2]
    assert result["bootstrap"]["blocks"][1]["observed_calendar_block_ids"] == [0, 3]
    assert result["bootstrap"]["seed"] == 20260929
    assert result["bootstrap"]["generated_replicates"] == 2000
    assert result["bootstrap"]["block_hours"] == 48 and result["bootstrap"]["block_days"] == 2
    assert result["scientific_claim"] is None and result["independent_scientific_approval"] is False


def test_immutable_prescription_copies_and_all_raw_records_retained():
    first = api.prespecified_specs()
    first["cf_learned_frozen_seed7"]["horizons"][1] = 2
    first["cf_scratch_seed23"]["head_seed"] = 23
    fresh = api.prespecified_specs()
    assert fresh["cf_learned_frozen_seed7"]["horizons"] == [1, 3, 6]
    assert fresh["cf_scratch_seed23"]["head_seed"] == 100023
    records = fixture()
    result = run(records)
    assert result["original_campaign_modified"] is False
    assert result["original_campaign_requirements_unchanged"] == {
        "neural_endpoints": 43,
        "total_methods": 47,
    }
    for record in records:
        saved = result["raw_records"][record["name"]]
        assert saved["spec"] == record["spec"] and saved["lineage"] == record["lineage"]
        assert saved["arrays"].keys() == record["arrays"].keys()
        for key in record["arrays"]:
            np.testing.assert_array_equal(saved["arrays"][key], record["arrays"][key])


def test_readonly_arrays_work_but_array_callbacks_are_not_inputs():
    records = fixture(identical=True)
    for record in records:
        for value in record["arrays"].values():
            value.setflags(write=False)
    assert run(records)["status"] == "CALCULATED"

    class UnsafeArray(np.ndarray):
        def __array_ufunc__(self, *args, **kwargs):
            pytest.fail("Array callback must not execute")

    records = fixture()
    records[0]["arrays"]["targets"] = records[0]["arrays"]["targets"].view(UnsafeArray)
    with pytest.raises(ValueError, match="callbacks"):
        run(records)


def test_seed_statistics_fail_closed_on_arithmetic_overflow():
    with pytest.raises(ValueError, match="overflow"):
        api._mean_sd([1e307, 0.0, 1e307])


def test_unsupported_bootstrap_draws_are_not_filtered_into_an_interval():
    records = fixture()
    for record in records:
        a = record["arrays"]
        lacking = (a["deployment"] == "A") & (a["target_dates"][:, 2] != "2026-01-06")
        a["observed"][lacking, 2] = False
    result = run(records)
    assert result["status"] == "CALCULATED"
    assert result["bootstrap"]["generated_replicates"] == 2000
    assert 0 < result["bootstrap"]["unsupported_replicates"] < 2000
    assert result["bootstrap"]["interval_status"] == "NOT_ASSESSABLE"
    assert all(
        p["mean_difference_ci95_db"] is None for p in result["paired_seed_contrasts"].values()
    )
    assert all(p["interval95_db"] is None for p in result["ensemble_minus_lightgbm"].values())
    assert result["common_support"]["issued_rows"] == len(records[0]["arrays"]["targets"])
