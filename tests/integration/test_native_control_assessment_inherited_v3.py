"""SYNTHETIC_CORRECTNESS_ONLY CPU; in-memory codecs, no scientific fits."""

from __future__ import annotations

import copy
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

BUILDER = Path(__file__).resolve().parents[2]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "native_control_assessment_test_support_v3",
    BUILDER / "tests/unit/test_native_control_assessment_inherited_v3.py",
)
helpers = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = helpers
HELPER_SPEC.loader.exec_module(helpers)
assessment = helpers.assessment
memory_case = helpers.memory_case


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_actual_v2_safe_forecast_replay_geometry_and_inaccessible_ancestry(seed):
    import torch

    from marine_echo.models.native_band_temporal import NativeBandTemporalModel
    from marine_echo.training.native_band_replication_ssl import Config

    torch.set_num_threads(1)
    config = Config(seed=seed, width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NativeBandTemporalModel(8, 4, 1, 2).eval()
    stats = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    artifact = {
        "kind": "native_band_replication_ssl_weights_only_inference_v2",
        "architecture": "nonlinear_frequency_conditioned_v1",
        "config": config,
        "scalers": stats,
        "model": model.state_dict(),
        "bindings": {"Z:/unavailable/TRAIN/ancestry.json": "a" * 64},
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "correctness_smoke": True,
    }
    stream = io.BytesIO()
    torch.save(artifact, stream)
    path = BUILDER / "evidence/ssl-replication-assessment-prefix-builder-v4/virtual-portable.pt"
    spec = {
        "kind": artifact["kind"],
        "mode": "core_frozen_readout",
        "method": "shared_ssl",
        "seed": seed,
        "model_paths": [str(path)],
    }
    before_rng = torch.get_rng_state().clone()
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    replay = assessment._builtin(spec, {path: stream.getvalue()}, config, stats, "cpu", path.parent)
    data = helpers.synthetic_arrays()
    x = np.where(data["context_observed"], data["x"], 0).astype(np.float32)
    with torch.no_grad():
        expected = model.forecast(
            *(
                torch.from_numpy(v)
                for v in (x, data["context_observed"], data["metadata"], data["query"])
            )
        ).numpy()
    actual = replay(data["x"], data["context_observed"], data["metadata"], data["query"])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
    np.testing.assert_allclose(data["query"][..., 4] * 250, 230, atol=1e-5)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before_state.items())
    altered = data["x"].copy()
    altered[~data["context_observed"]] = 1e10
    np.testing.assert_array_equal(
        actual, replay(altered, data["context_observed"], data["metadata"], data["query"])
    )


def test_context_only_clean_and_robustness_preserve_scored_support(memory_case):
    case = memory_case
    case.manifest["methods"]["same"] = copy.deepcopy(case.spec)
    case.seal()
    clean = case.run()
    with np.load(io.BytesIO(case.fs.files[case.output / "fixed.npz"]), allow_pickle=False) as z:
        clean_arrays = {k: z[k].copy() for k in z.files}
    case.output = case.base / "robust-out"
    case.manifest["robustness"] = dict(assessment.ROBUSTNESS)
    case.seal()
    robust = case.run()
    assert clean["status"] == "COMPLETED_FORECASTS"
    assert clean["native_geometry"]["lower_m"] == robust["native_geometry"]["lower_m"] == 230
    assert robust["context_diagnostics"]["dropped_observations"] > 0
    assert clean["support"] == robust["support"]
    assert clean["methods"]["fixed"]["metrics"] == robust["methods"]["fixed"]["metrics"]
    assert robust["methods"]["fixed"]["metrics"] == robust["methods"]["same"]["metrics"]
    assert robust["scientific_claim"] is None
    assert robust["paired_intervals"].startswith("NOT_RUN")
    with np.load(io.BytesIO(case.fs.files[case.output / "fixed.npz"]), allow_pickle=False) as z:
        for k in ("targets", "observed", "row_id", "deployment", "target_dates", "query"):
            np.testing.assert_array_equal(z[k], clean_arrays[k])
        np.testing.assert_array_equal(z["predictions"], clean_arrays["predictions"])
        assert not {"x", "future", "future_observed"} & set(z.files)
    with np.load(
        io.BytesIO(case.fs.files[case.output / "context-diagnostics.npz"]), allow_pickle=False
    ) as z:
        positions, dropped = z["drop_positions"], z["dropped"]
    np.testing.assert_array_equal(positions, np.argwhere(dropped))
    assert not dropped[..., 0].any()
    assert robust["resources"]["elapsed_seconds"] >= 0
    assert robust["resources"]["process_cuda_peak_allocated_bytes"] == 0


def test_no_future_decode_and_strict_four_argument_calls(memory_case, monkeypatch):
    original = np.lib.npyio.NpzFile.__getitem__
    calls = []

    def checked(saved, key):
        calls.append(key)
        assert key not in {"future", "future_observed"}
        return original(saved, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", checked)
    result = memory_case.run()
    assert result["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    assert "targets" in calls
    assert "future" not in calls
    assert set(result["mask_provenance"]["original_fields"]) == {"target_observed"}


def test_missing_support_not_faked_as_completed_assessment(memory_case):
    case = memory_case
    case.arrays["target_observed"][:] = False
    case.arrays["targets"][:] = np.nan
    case.seal()
    result = case.run()
    assert result["status"] == "COMPLETED_FORECASTS_NOT_ASSESSABLE"
    assert result["methods"]["fixed"]["status"] == "NOT_ASSESSABLE"
    assert result["methods"]["fixed"]["metrics"]["primary_pinball_db"] is None


def test_protected_outputs_fail_before_reloading(memory_case, monkeypatch):
    case = memory_case
    case.run()
    monkeypatch.setattr(
        assessment.np, "load", lambda *a, **k: pytest.fail("Output collision decoded corpus")
    )
    with pytest.raises(FileExistsError):
        case.run()


@pytest.mark.parametrize("bad", ["nonfinite", "shape", "unordered"])
def test_invalid_forecasts_never_get_completion_receipt(memory_case, monkeypatch, bad):
    def factory(spec, snapshots, config, statistics, device, base):
        def forecast(x, observed, metadata, query):
            p = np.zeros((len(x), 3, 5), dtype=np.float32)
            if bad == "nonfinite":
                p[0, 0, 0] = np.nan
            elif bad == "shape":
                p = p[:, :, :4]
            else:
                p[..., 0] = 1
            return p

        return forecast

    # Callable source file is independently part of the static source closure.
    case = memory_case
    case.review["bindings"].update(
        {
            str(p): assessment.sha(p.read_bytes())
            for p in assessment.required_sources({"spy": factory})
        }
    )
    case.fs.files[case.review_path] = helpers._encoded(case.review)
    with pytest.raises(ValueError, match="forecast"):
        assessment.execute_assessment(
            case.manifest_path, case.review_path, case.output, loaders={"spy": factory}
        )
    assert case.output / "completion.json" not in case.fs.files


def test_exclusive_write_failure_propagates_once_without_retry(memory_case, monkeypatch):
    case = memory_case
    original = Path.open
    denied, attempted = case.output / "context-diagnostics.npz", []

    def opened(path, *args, **kwargs):
        if path == denied:
            attempted.append(path)
            raise PermissionError("Synthetic explicit denial; not an OS operation.")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(PermissionError, match="Synthetic"):
        case.run()
    assert attempted == [denied]
    assert case.output / "completion.json" not in case.fs.files


def test_safe_native_load_matches_real_backbone_without_rng_or_optimizer():
    import torch

    from marine_echo.models.native_temporal import NativeTemporalModel
    from marine_echo.training.native_ssl import Config

    torch.set_num_threads(1)
    config = Config(width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
    model = NativeTemporalModel(8, 4, 1, 2).eval().requires_grad_(False)
    scalers = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    artifact = {
        "kind": "native_ssl_weights_only_inference_v1",
        "config": config,
        "scalers": scalers,
        "model": model.state_dict(),
    }
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    p = Path("/synthetic-owned-model.pt").resolve()
    spec = {"kind": artifact["kind"], "model_paths": [str(p)], "mode": "core_frozen_readout"}
    state = torch.random.get_rng_state().clone()
    forecast = assessment._builtin(spec, {p: buffer.getvalue()}, config, scalers, "cpu", p.parent)
    assert torch.equal(state, torch.random.get_rng_state())
    data = helpers.synthetic_arrays(3)
    x, mask, _ = assessment.secondary_dropout(data["x"], data["context_observed"], None)
    with torch.inference_mode():
        expected = model.forecast(
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(data["metadata"]),
            torch.from_numpy(data["query"]),
        ).numpy()
    before = {k: v.clone() for k, v in forecast.__self__.model.state_dict().items()}
    actual = forecast(x, mask, data["metadata"], data["query"])
    np.testing.assert_array_equal(actual, expected)
    assert all(torch.equal(v, forecast.__self__.model.state_dict()[k]) for k, v in before.items())
    assert all(not p.requires_grad and p.grad is None for p in forecast.__self__.model.parameters())


@pytest.mark.parametrize("tamper", ["kind", "scaler", "tensor"])
def test_owned_weights_only_codec_rejects_wrong_state(tamper):
    import torch

    from marine_echo.models.native_temporal import NativeTemporalModel
    from marine_echo.training.native_ssl import Config

    config = Config(width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
    scalers = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    model = NativeTemporalModel(8, 4, 1, 2)
    artifact = {
        "kind": "native_ssl_weights_only_inference_v1",
        "config": config,
        "scalers": copy.deepcopy(scalers),
        "model": model.state_dict(),
    }
    if tamper == "kind":
        artifact["kind"] = "native_ssl_selected_encoder_v1"
    elif tamper == "scaler":
        artifact["scalers"]["channel_std"][0] = 0
    else:
        artifact["model"].pop(next(iter(artifact["model"])))
    stream = io.BytesIO()
    torch.save(artifact, stream)
    p = Path("/synthetic-owned-model.pt").resolve()
    spec = {
        "kind": "native_ssl_weights_only_inference_v1",
        "model_paths": [str(p)],
        "mode": "core_frozen_readout",
    }
    with pytest.raises(ValueError):
        assessment._builtin(spec, {p: stream.getvalue()}, config, scalers, "cpu", p.parent)


def test_load_only_references_preserve_source_period_recipe():
    from marine_echo.training.native_references import feature_matrix

    data = helpers.synthetic_arrays(3)
    x, mask, _ = assessment.secondary_dropout(data["x"], data["context_observed"], None)
    p = Path("/synthetic-reference-recipe.json").resolve()
    for kind in ("persistence", "seasonal24"):
        spec = {"kind": kind, "model_paths": [str(p)]}
        forecast = assessment._builtin(
            spec, {p: b"frozen recipe"}, {"history": 96}, {"fit_role": "train"}, "cpu", p.parent
        )
        actual = forecast(x, mask, data["metadata"], data["query"])
        expected = (
            np.repeat(x[:, -1, 0, None], 3, axis=1)
            if kind == "persistence"
            else x[:, [-24, -22, -19], 0]
        )
        np.testing.assert_array_equal(actual, np.repeat(expected[..., None], 5, axis=-1))
    altered = x.copy()
    altered[~mask] = 1e9
    np.testing.assert_array_equal(
        feature_matrix({"x": x, "observed": mask, "metadata": data["metadata"]}, 96),
        feature_matrix({"x": altered, "observed": mask, "metadata": data["metadata"]}, 96),
    )


def test_independent_daily_horizon_deployment_weights_and_masked_fills(memory_case):
    case = memory_case
    case.arrays = helpers.synthetic_arrays(104)
    a = case.arrays
    a["x"][:, :, 0] = 0
    a["deployment"] = a["deployment"].astype("<U32")
    a["deployment"][64:] = "synthetic-test2"
    a["target_observed"][:] = True
    a["target_observed"][0] = False
    a["targets"][:24] = 0
    a["targets"][24:64] = 4
    a["targets"][64:] = 10
    a["targets"][0] = np.nan
    a["target_dates"][24:64] = "2026-01-02"
    a["target_dates"][84:] = "2026-01-02"
    case.documents["split.json"]["reserved_test"].append(helpers.identity("test2"))
    case.documents["cohort.json"]["identities"].append(helpers.identity("test2"))
    case.documents["cohort.json"]["rows"] = list(
        map(list, zip(a["deployment"], a["row_id"], strict=True))
    )
    case.seal()
    metrics = case.run()["methods"]["fixed"]["metrics"]
    # Constant0: mean five-quantile loss is half the positive target. A's daily
    # losses0/2 average1, B's5/5 average5; equal deployments give3, not row mean.
    assert metrics["primary_pinball_db"] == pytest.approx(3)
    assert metrics["per_deployment"]["synthetic-test"]["primary_pinball_db"] == pytest.approx(1)
    assert metrics["per_deployment"]["synthetic-test2"]["primary_pinball_db"] == pytest.approx(5)
    assert metrics["scored_per_horizon"] == [103] * 3
    assert metrics["eligible_source_dates_per_horizon"] == [4] * 3
    assert len(metrics["daily_rows"]) == 12
    assert metrics["primary_pinball_db"] != pytest.approx((40 * 4 + 40 * 10) / 103 / 2)


def test_target_and_unopened_future_changes_cannot_change_forecasts(memory_case):
    case = memory_case
    first = case.run()
    with np.load(io.BytesIO(case.fs.files[case.output / "fixed.npz"]), allow_pickle=False) as z:
        before = z["predictions"].copy()
    case.output = case.base / "target-perturbed-out"
    mask = case.arrays["target_observed"]
    case.arrays["targets"][mask] += 123
    case.arrays["future"] = np.asarray([{"never": "decode"}], dtype=object)
    case.seal()
    second = case.run()
    with np.load(io.BytesIO(case.fs.files[case.output / "fixed.npz"]), allow_pickle=False) as z:
        np.testing.assert_array_equal(before, z["predictions"])
    assert first["methods"]["fixed"]["metrics"] != second["methods"]["fixed"]["metrics"]


def test_existing_comparison_decoder_accepts_saved_synthetic_outputs(memory_case):
    from marine_echo.evaluation import native_comparison

    case = memory_case
    case.run()
    decoded, trace, optional = native_comparison._load(
        case.fs.files[case.output / "fixed.npz"], "final_test", "SYNTHETIC_CORRECTNESS_ONLY"
    )
    assert decoded["query"].shape == (24, 3, 10)
    assert trace["original_fields"] == ["observed"]
    assert "query_native_bounds_m" in optional


@pytest.mark.parametrize("family", ["cf", "band"])
def test_cf_ema_and_band_owned_inference_replay_frozen_state(family):
    import torch

    from marine_echo.models.native_temporal import CFNativeModel
    from marine_echo.training.native_ssl import Config

    scalers = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    if family == "cf":
        config = Config(
            method="cf_jepa", cf_width=8, cf_latent=4, cf_blocks=1, batch_size=8
        ).to_dict()
        model = CFNativeModel(8, 4, 1).eval()
        kind = "native_ssl_weights_only_inference_v1"
        extra = {}
    else:
        from marine_echo.models.native_band_temporal import ARCHITECTURE, NativeBandTemporalModel
        from marine_echo.training.native_band_ssl import Config as BandConfig

        config = BandConfig(width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
        model = NativeBandTemporalModel(8, 4, 1, 2).eval()
        kind = "native_band_ssl_weights_only_inference_v1"
        extra = {
            "architecture": ARCHITECTURE,
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "correctness_smoke": True,
            "bindings": {"opaque-ancestral-source": "0" * 64},
        }
    artifact = {
        "kind": kind,
        "config": config,
        "scalers": scalers,
        "model": model.state_dict(),
        **extra,
    }
    stream = io.BytesIO()
    torch.save(artifact, stream)
    p = Path("/synthetic-owned-family.pt").resolve()
    spec = {"kind": kind, "model_paths": [str(p)], "mode": "core_frozen_readout"}
    rng = torch.random.get_rng_state().clone()
    forecast = assessment._builtin(spec, {p: stream.getvalue()}, config, scalers, "cpu", p.parent)
    assert torch.equal(rng, torch.random.get_rng_state())
    data = helpers.synthetic_arrays(3)
    x, mask, _ = assessment.secondary_dropout(data["x"], data["context_observed"], None)
    with torch.inference_mode():
        expected = model.forecast(
            torch.from_numpy(x),
            torch.from_numpy(mask),
            torch.from_numpy(data["metadata"]),
            torch.from_numpy(data["query"]),
        ).numpy()
    before = {k: v.clone() for k, v in forecast.__self__.model.state_dict().items()}
    np.testing.assert_array_equal(forecast(x, mask, data["metadata"], data["query"]), expected)
    assert all(torch.equal(v, forecast.__self__.model.state_dict()[k]) for k, v in before.items())
    assert not forecast.__self__.model.training
    assert all(not p.requires_grad and p.grad is None for p in forecast.__self__.model.parameters())


def test_fifteen_utf8_boosters_load_only_with_exact_feature_schema(monkeypatch):
    import json
    from types import SimpleNamespace

    from marine_echo.training.native_references import feature_matrix

    data = helpers.synthetic_arrays(3)
    x, mask, _ = assessment.secondary_dropout(data["x"], data["context_observed"], None)
    count = feature_matrix({"x": x, "observed": mask, "metadata": data["metadata"]}, 96).shape[1]
    texts, received = [], []

    class Booster:
        def __init__(self, *, model_str):
            received.append(model_str)
            self.value = json.loads(model_str)["value"]

        def num_feature(self):
            return count

        def predict(self, features):
            assert features.shape == (3, count)
            return np.full(3, self.value, dtype=float)

    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(Booster=Booster))
    slots = [[h, float(q)] for h in (1, 3, 6) for q in assessment.native_product.QUANTILES]
    paths = [Path(f"/synthetic-booster-{i}.txt").resolve() for i in range(15)]
    for i in range(15):
        texts.append(
            json.dumps(
                {"label": "SYNTHETIC_CORRECTNESS_ONLY Unicode á", "value": i}, ensure_ascii=False
            )
        )
    snapshots = {p: t.encode("utf-8") for p, t in zip(paths, texts, strict=True)}
    spec = {"kind": "lightgbm15_utf8", "model_paths": list(map(str, paths)), "booster_slots": slots}
    forecast = assessment._builtin(
        spec, snapshots, {"history": 96}, {"fit_role": "train"}, "cpu", paths[0].parent
    )
    np.testing.assert_array_equal(
        forecast(x, mask, data["metadata"], data["query"]),
        np.broadcast_to(np.arange(15).reshape(3, 5), (3, 3, 5)),
    )
    assert received == texts
    # The fake library has no fit/refit/training API: any such call would fail.


def test_safe_kind_error_precedes_corpus_decode(memory_case, monkeypatch):
    import torch

    case = helpers.Case(memory_case.fs, learned=True)
    case.spec["loader"] = "builtin"
    stream = io.BytesIO()
    torch.save({"kind": "native_ssl_selected_encoder_v1"}, stream)
    case.fs.files[case.base / "model.bin"] = stream.getvalue()
    case.seal()
    monkeypatch.setattr(
        assessment.np,
        "load",
        lambda *a, **k: pytest.fail("Corpus decoded before safe artifact kind check"),
    )
    with pytest.raises(ValueError, match="kind"):
        case.run()
    assert not case.output.exists()


def test_missing_downstream_config_cannot_be_reinterpreted_as_supported_forecast():
    import torch

    from marine_echo.models.native_temporal import NativeTemporalModel
    from marine_echo.training.native_ssl import Config

    config = Config(width=8, latent=4, blocks=1, heads=2, batch_size=8).to_dict()
    scalers = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    artifact = {
        "kind": "native_ssl_weights_only_inference_v1",
        "config": config,
        "scalers": scalers,
        "model": NativeTemporalModel(8, 4, 1, 2).state_dict(),
        "supervised_ancestry": {"mode": "full_finetune", "ssl_only": False},
    }
    stream = io.BytesIO()
    torch.save(artifact, stream)
    p = Path("/synthetic-supervised.pt").resolve()
    spec = {"kind": artifact["kind"], "model_paths": [str(p)], "mode": "full_finetune"}
    with pytest.raises(ValueError, match="downstream config"):
        assessment._builtin(spec, {p: stream.getvalue()}, config, scalers, "cpu", p.parent)


@pytest.mark.parametrize("mode", ["frozen_readout", "full_finetune", "direct_end_to_end"])
def test_supported_supervised_forecasts_require_encoded_parent_graph_and_counters(mode):
    import torch

    from marine_echo.models.native_temporal import NativeTemporalModel
    from marine_echo.training.native_ssl import Config

    config = Config(
        method="direct" if mode == "direct_end_to_end" else "shared_ssl",
        width=8,
        latent=4,
        blocks=1,
        heads=2,
        batch_size=8,
    ).to_dict()
    scalers = {
        "channel_mean": [0.0] * 4,
        "channel_std": [1.0] * 4,
        "target_mean": [0.0] * 3,
        "target_std": [1.0] * 3,
    }
    parents = set() if mode == "direct_end_to_end" else {"a" * 64, "b" * 64}
    ancestry = {
        "mode": mode,
        "ssl_only": False,
        "supervised_updates": 3000,
        "selected_supervised_step": 750,
        "ancestor_encoder_sha256": None if not parents else "a" * 64,
        "ancestor_run_sha256": None if not parents else "b" * 64,
    }
    downstream = {k: config[k] for k in ("method", "seed", "history", "batch_size")}
    downstream.update(mode=mode, updates=3000)
    artifact = {
        "kind": "native_ssl_weights_only_inference_v1",
        "config": config,
        "scalers": scalers,
        "model": NativeTemporalModel(8, 4, 1, 2).state_dict(),
        "supervised_ancestry": ancestry,
        "downstream_config": downstream,
    }
    p = Path("/synthetic-supervised.pt").resolve()
    spec = {"kind": artifact["kind"], "model_paths": [str(p)], "mode": mode}
    stream = io.BytesIO()
    torch.save(artifact, stream)
    forecast = assessment._builtin(
        spec,
        {p: stream.getvalue()},
        config,
        scalers,
        "cpu",
        p.parent,
        ancestor_artifact_hashes=parents,
    )
    data = helpers.synthetic_arrays(3)
    x, mask, _ = assessment.secondary_dropout(data["x"], data["context_observed"], None)
    assert forecast(x, mask, data["metadata"], data["query"]).shape == (3, 3, 5)
    if parents:
        with pytest.raises(ValueError, match="recursive graph"):
            assessment._builtin(
                spec,
                {p: stream.getvalue()},
                config,
                scalers,
                "cpu",
                p.parent,
                ancestor_artifact_hashes=set(),
            )
    artifact["supervised_ancestry"]["selected_supervised_step"] = 3001
    stream = io.BytesIO()
    torch.save(artifact, stream)
    with pytest.raises(ValueError, match="update/selection"):
        assessment._builtin(
            spec,
            {p: stream.getvalue()},
            config,
            scalers,
            "cpu",
            p.parent,
            ancestor_artifact_hashes=parents,
        )
