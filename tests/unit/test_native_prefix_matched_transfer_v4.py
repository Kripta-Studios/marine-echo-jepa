"""SYNTHETIC_CORRECTNESS_ONLY matched configuration and source behavior."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-transfer-integration-builder-v1")
)
from test_support import control_parent_metadata, controls, legacy, metadata_helper, prefix_module

prefix = prefix_module()
torch.set_num_threads(2)


@pytest.mark.parametrize(
    "method,mode",
    [
        ("cf_random_frozen", "frozen_readout"),
        ("cf_direct_supervised", "frozen_readout"),
        ("direct", "scratch_direct"),
    ],
)
def test_matched_cf_alternatives_admitted(method, mode):
    config = prefix.PrefixConfig(method=method, family="cf", mode=mode)
    config.validate("REVIEWED_PREFIX_TRANSFER")


def test_conventional_cell_is_explicit_separate_feature_recipe():
    prefix.PrefixConfig(method="lightgbm", family="reference", mode="conventional").validate(
        "REVIEWED_PREFIX_TRANSFER"
    )


def cfg(method="cf_random_frozen", mode="frozen_readout", seed=7):
    return prefix.PrefixConfig(
        method=method,
        family="cf",
        mode=mode,
        seed=seed,
        correctness_smoke=True,
        updates=4,
        cadence=1,
        batch_size=4,
    )


STATS = {
    "channel_mean": [-50.0] * 4,
    "channel_std": [5.0] * 4,
    "target_mean": [-50.0] * 3,
    "target_std": [5.0] * 3,
}


def selected(method="cf_random_frozen", seed=7):
    c = controls.Config(
        method=method, seed=seed, width=8, latent=8, blocks=1, updates=4, cadence=1, batch_size=4
    )
    source = controls.prepare_model(c)
    ancestry = {
        "mode": c.mode,
        "ssl_only": False,
        "ssl_updates": 0,
        "ancestor_encoder_sha256": None,
        "ancestor_run_sha256": None,
        "initialization_seed": seed,
        "readout_initialization_seed": seed + 100000,
        "supervised_updates": 4,
        "selected_supervised_step": 1,
        "encoder_supervised_updates": 0 if method == "cf_random_frozen" else 4,
    }
    return {
        "kind": controls.ENCODER_KINDS[method],
        "config": c.to_dict(),
        "core_config": controls.core_config(c).to_dict(),
        "encoder": prefix.core.cpu_state(source.encoder),
        "scalers": STATS,
        "bindings": {"unavailable-parent.py": "a" * 64},
        "supervised_ancestry": ancestry,
        "evidence_kind": prefix.EVIDENCE,
    }, source


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_actual_cf_control_heads_frozen_buffers_and_scratch_gradient(seed):
    heads = []
    for method in controls.METHODS:
        artifact, source = selected(method, seed)
        model = prefix.prepare_model(cfg(method, seed=seed), artifact["config"], artifact, STATS)
        heads.append(prefix.core.cpu_state(model.head))
        assert all(
            torch.equal(v, source.readout.state_dict()[k])
            for k, v in model.head.state_dict().items()
        )
        initial = prefix.core.cpu_state(model.encoder)
        registry = metadata_helper.metadata(cutoffs=(0, 1, 2, 3))
        data = metadata_helper.arrays(registry["rows"], registry)
        b = prefix._batch(data, np.arange(4), prefix._scalers(STATS), "cpu", labels=True)
        loss = prefix.core.pinball(
            model.forecast(b["x"], b["observed"], b["metadata"], b["query"]),
            b["y"],
            b["y_observed"],
        )
        loss.backward()
        assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in initial.items())
        assert all(p.grad is None for p in model.encoder.parameters())
        assert model.feature_ancestor["parent_head_reused"] is False
        assert model.feature_ancestor["training_kind"] == controls.TRAINING_KINDS[method]
    assert all(torch.equal(v, heads[1][k]) for k, v in heads[0].items())
    raw = controls.core_config(
        controls.Config(seed=seed, width=8, latent=8, blocks=1, updates=4, cadence=1, batch_size=4)
    ).to_dict()
    scratch_cfg = cfg("direct", "scratch_direct", seed)
    scratch = prefix.prepare_model(scratch_cfg, raw, None, STATS)
    assert all(torch.equal(v, scratch.head.state_dict()[k]) for k, v in heads[0].items())
    scratch.encoder.train()
    prediction = scratch.forecast(b["x"], b["observed"], b["metadata"], b["query"], frozen=False)
    prefix.core.pinball(prediction, b["y"], b["y_observed"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in scratch.encoder.parameters())
    for step in range(4):
        np.testing.assert_array_equal(
            prefix.sample_indices(18, scratch_cfg, step),
            prefix.sample_indices(18, cfg(seed=seed), step),
        )


@pytest.mark.parametrize("damage", ["ssl", "kind", "seed", "scalers", "head_inheritance"])
def test_control_feature_identity_fails_closed(damage):
    value, _ = selected()
    if damage == "ssl":
        value["supervised_ancestry"]["ssl_updates"] = 6000
    elif damage == "kind":
        value["kind"] = "native_ssl_selected_encoder_v1"
    elif damage == "seed":
        value["config"]["seed"] = 13
    elif damage == "scalers":
        value["scalers"] = {}
    else:
        value["config"]["method"] = "cf_jepa"
    with pytest.raises((ValueError, TypeError)):
        prefix.prepare_model(cfg(), value["config"], value, STATS)


def test_boundaries_one_day_feasible_and_same_suffix_independent_of_values():
    raw = metadata_helper.metadata(cutoffs=tuple(range(18)) + tuple(range(888, 906)))
    parts = [prefix.partitions(raw, days) for days in (1, 7, 30)]
    assert len(parts[0]["fit_rows"]) == 18
    assert all(p["suffix_support_sha256"] == parts[0]["suffix_support_sha256"] for p in parts)
    assert prefix.boundaries("2026-01-01", 1)[2].isoformat() == "2026-02-11T00:00:00"
    assert prefix.partitions(raw, 1) == legacy.partitions(raw, 1)


def test_native_multiconfiguration_partitions_stay_byte_semantic_equal():
    raw = metadata_helper.native_configuration_metadata()[0]
    for days in (1, 7, 30):
        assert prefix.partitions(raw, days) == legacy.partitions(raw, days)


def admit_control_metadata(value):
    import json

    prefix._admit_cf_parent(
        value.config,
        value.manifest,
        value.parent,
        value.backbone,
        value.encoder_raw,
        b"SYNTHETIC opaque inference",
        json.dumps(value.membership).encode(),
        set(),
        {prefix.sha(value.bound(value.inputs.train)): value.inputs.train},
        {("synthetic-TRAIN-deployment", "synthetic-TRAIN-archive")},
        value.inputs.split,
        value.base,
        value.bound,
        value.doc,
    )


def test_actual_control_synthetic_parent_status_is_not_real_approval(monkeypatch):
    value = control_parent_metadata()
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Parent metadata decoded tensors")
    )
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Parent metadata decoded arrays"))
    before = torch.get_rng_state().clone()
    admit_control_metadata(value)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("damage", ["ssl", "foreign_train", "samples", "source", "scaler_ancestor"])
def test_typed_control_parent_metadata_is_bound_before_decode(damage):
    value = control_parent_metadata()
    if damage == "ssl":
        value.parent["supervised_ancestry"]["ssl_updates"] = 1
    elif damage == "foreign_train":
        value.membership["train_deployments"][0] = "reserved-site"
    elif damage == "samples":
        value.membership["sequence"][0]["indices"][0] = -1
    elif damage == "source":
        value.parent["bindings"][str(Path(controls.__file__))] = "a" * 64
    else:
        value.parent["fitted_weight_ancestors"] = ["unknown-local-parent"]
    with pytest.raises(ValueError):
        admit_control_metadata(value)


@pytest.mark.parametrize("method", controls.METHODS)
def test_actual_control_encoder_and_full_forecast_codec_have_same_features(method):
    artifact, source = selected(method)
    inference = {k: v for k, v in artifact.items() if k not in ("encoder", "kind")}
    inference.update(kind=controls.INFERENCE_KIND, model=prefix.core.cpu_state(source))
    parent = {
        "bindings": artifact["bindings"],
        "supervised_ancestry": artifact["supervised_ancestry"],
    }
    before = torch.get_rng_state().clone()
    prefix._check_control_artifacts(
        artifact, inference, parent, cfg(method), artifact["config"], STATS
    )
    assert torch.equal(before, torch.get_rng_state())
    changed = {**artifact, "encoder": {k: v.clone() for k, v in artifact["encoder"].items()}}
    key = next(k for k, v in changed["encoder"].items() if v.dtype == torch.float32)
    changed["encoder"][key].add_(0.125)
    with pytest.raises(ValueError):
        prefix._check_control_artifacts(
            changed, inference, parent, cfg(method), artifact["config"], STATS
        )


def test_missing_prefix_labels_supply_no_gradient_or_value_dependence():
    predictions = torch.zeros((2, 3, 5), requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    target = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    first = prefix.core.pinball(predictions, target, mask)
    first.backward()
    assert not predictions.grad[~mask].any()
    changed = target.clone()
    changed[~mask] = 1e20
    assert torch.equal(first, prefix.core.pinball(predictions.detach(), changed, mask))
