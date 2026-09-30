"""SYNTHETIC_CORRECTNESS_ONLY CPU integration; no public-data execution.

Optimizer fixtures require explicit ROOT enablement in its integrated checkout.
The builder never retries previously denied optimizer/cache operations.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BUILDER = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_prefix_unit_support", BUILDER / "tests/unit/test_native_prefix_transfer.py"
)
helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helpers
SPEC.loader.exec_module(helpers)
prefix, support = helpers.prefix, helpers.support
memory_case = helpers.memory_case

ROOT_OPTIMIZERS = (
    os.environ.get("NATIVE_PREFIX_ROOT_RUNNER_CHECKS") == "1" and BUILDER.name == "marine-echo-jepa"
)
root_optimizer = pytest.mark.skipif(
    not ROOT_OPTIMIZERS,
    reason="NOT_RUN builder: no denied optimizer/cache retry; root integrated synthetic CPU fixture only",
)


@pytest.mark.parametrize("seed", [7, 13, 23])
@root_optimizer
def test_replication_root_optimizer_resume_frozen_and_scratch(seed):
    args = list(trajectory_inputs("scratch_direct", "band"))
    args[0] = prefix.PrefixConfig(**{**args[0].to_dict(), "band_artifact_version": 2, "seed": seed})
    from marine_echo.training.native_band_replication_ssl import Config

    args[1] = Config(
        method="direct", seed=seed, width=8, latent=4, blocks=1, heads=2, pretrain_updates=3000
    ).to_dict()
    uninterrupted = prefix._Trajectory(*args)
    initial = prefix.core.cpu_state(uninterrupted.model.encoder)
    initial_head = prefix.core.cpu_state(uninterrupted.model.head)
    rng = prefix.core.rng_state()
    uninterrupted.advance()
    partial = prefix._Trajectory(*args)
    prefix.core.restore_rng(rng)
    partial.advance(2)
    restored = prefix._Trajectory(*args)
    restored.restore(prefix.decode_checkpoint(prefix.encode_checkpoint(partial.checkpoint())))
    restored.advance()
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in uninterrupted.model.state_dict().items()
    )
    assert any(
        not torch.equal(v, uninterrupted.model.encoder.state_dict()[k]) for k, v in initial.items()
    )
    frozen_args = list(args)
    frozen_args[0] = prefix.PrefixConfig(**{**args[0].to_dict(), "mode": "frozen_readout"})
    frozen_args[2] = {
        "kind": prefix.REPLICATION_SUPERVISED_KIND,
        "architecture": prefix.ARCHITECTURES["band"],
        "config": args[1],
        "scalers": args[3],
        "encoder": initial,
        "supervised_ancestry": support.direct_ancestry(),
    }
    frozen = prefix._Trajectory(*frozen_args)
    assert all(torch.equal(v, frozen.model.head.state_dict()[k]) for k, v in initial_head.items())
    frozen.advance()
    assert all(torch.equal(v, frozen.model.encoder.state_dict()[k]) for k, v in initial.items())
    assert frozen.samples == uninterrupted.samples
    assert [v["step"] for v in frozen.candidates] == [1, 2, 3, 4]


def trajectory_inputs(mode="scratch_direct", family="core"):
    config = prefix.PrefixConfig(
        method="direct",
        family=family,
        mode=mode,
        correctness_smoke=True,
        updates=4,
        cadence=1,
        batch_size=8,
    )
    backbone = prefix.core.Config(method="direct", width=8, latent=4, blocks=1, heads=2).to_dict()
    if family == "band":
        backbone["architecture"] = prefix.ARCHITECTURES["band"]
    stats = {
        "channel_mean": [-50.0] * 4,
        "channel_std": [5.0] * 4,
        "target_mean": [-50.0] * 3,
        "target_std": [5.0] * 3,
    }
    registry = support.metadata(cutoffs=tuple(range(18)) + (888,))
    partition = prefix.partitions(registry, 1)
    keys = set(partition["fit_rows"])
    rows = [r for r in registry["rows"] if (r["deployment"], r["row_id"]) in keys]
    train = support.arrays(rows, registry)
    train["target_observed"][0, 1] = False
    train["targets"][0, 1] = np.nan
    dev = support.arrays([], registry, "development", count=18)
    selected = None
    if mode == "frozen_readout":
        scratch_config = prefix.PrefixConfig(**{**config.to_dict(), "mode": "scratch_direct"})
        model = prefix.prepare_model(scratch_config, backbone, None, stats)
        selected = {
            "kind": prefix.selected_kind(config),
            "supervised_ancestry": support.direct_ancestry(),
            "config": backbone,
            "scalers": stats,
            "encoder": prefix.core.cpu_state(model.encoder),
        }
        if family == "band":
            selected["architecture"] = prefix.ARCHITECTURES["band"]
    return config, backbone, selected, stats, train, dev, {"synthetic_input": "a" * 64}


def test_fit_bad_gate_never_initializes_optimizer_or_decodes(memory_case, monkeypatch):
    case = memory_case
    case.review["bindings"].pop(str(case.base / "stats.json"))
    case.fs.files[case.review_path] = support.encoded(case.review)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Optimizer before gate"))
    monkeypatch.setattr(prefix.np, "load", lambda *a, **k: pytest.fail("Decode before gate"))
    monkeypatch.setattr(torch, "manual_seed", lambda *a, **k: pytest.fail("RNG before gate"))
    with pytest.raises(ValueError, match="binding"):
        prefix.fit(case.manifest_path, case.review_path, case.output)


def test_suffix_numerical_values_never_available_to_fitter(memory_case):
    c = memory_case
    admission = prefix.admit(c.manifest_path, c.review_path, c.output)
    admission.manifest["_base"] = str(c.base)
    decoded = prefix._decode(admission, "prefix_npz", "prefix")
    # An entirely unbound numeric suffix archive must remain outside accepted inputs.
    c.fs.files[c.base / "unrequested-suffix.npz"] = support.npz({"targets": np.full((6, 3), 1e12)})
    again = prefix._decode(admission, "prefix_npz", "prefix")
    for key in decoded:
        np.testing.assert_array_equal(again[key], decoded[key])
    assert not any("suffix.npz" in p for p in admission.identities)


@pytest.mark.parametrize("family", ["core", "band"])
def test_batched_frozen_replay_matches_actual_encoder_head_and_masked_fill(family):
    c, backbone, selected, stats, train, _dev, identities = trajectory_inputs(
        "frozen_readout", family
    )
    model = prefix.prepare_model(c, backbone, selected, stats).eval()
    artifact = {
        "kind": "native_prefix_transfer_inference_v1",
        "config": c.to_dict(),
        "backbone_config": backbone,
        "scalers": stats,
        "architecture": prefix.ARCHITECTURES[family],
        "model": prefix.core.cpu_state(model),
        "selected_step": 1,
        "zero_shot": False,
        "identities": identities,
    }
    replay = prefix.PrefixPredictor(io.BytesIO(prefix.encode_checkpoint(artifact)))
    before = prefix.core.cpu_state(replay.model)
    expected = prefix.predict(model, train, prefix._scalers(stats), c)
    got = replay.forecast(train["x"], train["context_observed"], train["metadata"], train["query"])
    np.testing.assert_array_equal(got, expected)
    filled = train["x"].copy()
    filled[~train["context_observed"]] = 1e10
    np.testing.assert_array_equal(
        got, replay.forecast(filled, train["context_observed"], train["metadata"], train["query"])
    )
    assert all(torch.equal(t, replay.model.state_dict()[k]) for k, t in before.items())
    np.testing.assert_allclose(train["query"][..., 4] * 250, 230)
    assert artifact["scalers"] == stats


def test_cf_ema_backbone_replay_freezes_bn_and_reports_head_size():
    c = prefix.PrefixConfig(
        method="cf_jepa", family="cf", correctness_smoke=True, updates=4, cadence=1
    )
    backbone = prefix.core.Config(method="cf_jepa", cf_width=8, cf_latent=4, cf_blocks=1).to_dict()
    stats = {
        "channel_mean": [-50.0] * 4,
        "channel_std": [5.0] * 4,
        "target_mean": [-50.0] * 3,
        "target_std": [5.0] * 3,
    }
    encoder = prefix.CFTemporalEncoder(8, 4, 1).eval()
    selected = {
        "kind": "native_ssl_selected_encoder_v1",
        "config": backbone,
        "scalers": stats,
        "encoder": prefix.core.cpu_state(encoder),
    }
    model = prefix.prepare_model(c, backbone, selected, stats)
    before = prefix.core.cpu_state(model.encoder)
    data = support.arrays([], {}, "development", count=3)
    prefix.predict(model, data, prefix._scalers(stats), c)
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())
    assert not model.encoder.training
    assert sum(p.numel() for p in prefix.QueryHead(64, 5, 64).parameters()) == 5189
    assert sum(p.numel() for p in prefix.QueryHead(128, 5, 128).parameters()) == 18565


@pytest.mark.parametrize("mode", ["scratch_direct", "frozen_readout"])
@root_optimizer
def test_actual_optimizer_changes_only_declared_encoder_and_four_dev_choices(mode):
    args = trajectory_inputs(mode)
    t = prefix._Trajectory(*args)
    original_scalers = copy.deepcopy(t.statistics)
    before = prefix.core.cpu_state(t.model.encoder)
    t.advance()
    changed = any(not torch.equal(v, t.model.encoder.state_dict()[k]) for k, v in before.items())
    assert changed == (mode == "scratch_direct")
    assert t.statistics == original_scalers
    assert [v["step"] for v in t.candidates] == [1, 2, 3, 4]
    assert {v["role"] for v in t.candidates} == {"original_development"}
    assert t.step == 4 and len(t.samples) == 4
    assert len(t.prefix["x"]) == 18
    assert t.prefix["target_observed"].sum(0).tolist() == [18, 17, 18]
    replay = prefix.PrefixPredictor(io.BytesIO(prefix.encode_checkpoint(t.inference_artifact())))
    assert replay.model.encoder.training is False


@pytest.mark.parametrize("mode", ["scratch_direct", "frozen_readout"])
@root_optimizer
def test_exact_resume_cpu_with_optimizer_scheduler_rng_samples_and_portable_replay(mode):
    args = trajectory_inputs(mode)
    full = prefix._Trajectory(*copy.deepcopy(args))
    rng = prefix.core.rng_state()
    full.advance()
    interrupted = prefix._Trajectory(*copy.deepcopy(args))
    prefix.core.restore_rng(rng)
    interrupted.advance(2)
    blob = prefix.encode_checkpoint(interrupted.checkpoint())
    resumed = prefix._Trajectory(*copy.deepcopy(args))
    resumed.restore(prefix.decode_checkpoint(blob))
    resumed.advance()
    assert full.samples == resumed.samples
    assert full.selected_step == resumed.selected_step
    assert full.candidates == resumed.candidates
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert all(
        torch.equal(v, resumed.model.state_dict()[k]) for k, v in full.model.state_dict().items()
    )
    assert all(torch.equal(v, resumed.selected_state[k]) for k, v in full.selected_state.items())
    raw = prefix.encode_checkpoint(resumed.inference_artifact())
    prediction = prefix.PrefixPredictor(io.BytesIO(raw))
    data = args[4]
    np.testing.assert_array_equal(
        prediction.forecast(data["x"], data["context_observed"], data["metadata"], data["query"]),
        prefix.predict(prediction.model, data, prediction.scalers, args[0]),
    )
    with pytest.raises(ValueError, match="identities"):
        changed = prefix._Trajectory(*copy.deepcopy(args))
        saved = prefix.decode_checkpoint(blob)
        saved["identities"]["wrong"] = "b" * 64
        changed.restore(saved)


@root_optimizer
def test_matched_modes_labels_samples_and_masked_values_no_gradient_or_selection():
    args = trajectory_inputs("scratch_direct")
    frozen_args = trajectory_inputs("frozen_readout")
    np.testing.assert_array_equal(args[4]["target_observed"], frozen_args[4]["target_observed"])
    a, b = prefix._Trajectory(*args), prefix._Trajectory(*frozen_args)
    a.advance()
    b.advance()
    assert a.samples == b.samples
    changed_args = copy.deepcopy(args)
    changed_args[4]["targets"][~changed_args[4]["target_observed"]] = 1e15
    changed = prefix._Trajectory(*changed_args)
    changed.advance()
    assert all(
        torch.equal(v, changed.model.state_dict()[k]) for k, v in a.model.state_dict().items()
    )
    assert a.candidates == changed.candidates


@root_optimizer
def test_protected_virtual_fit_completion_and_actual_safe_readout(memory_case, monkeypatch):
    case = memory_case

    # Explicit private transport from the first call; no failed OS writes rerouted.
    def save(path, value):
        case.fs.files[path] = prefix.encode_checkpoint(value)

    monkeypatch.setattr(prefix.core, "atomic_checkpoint", save)
    result = prefix.fit(case.manifest_path, case.review_path, case.output)
    assert result["status"] == "COMPLETED"
    assert result["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    assert result["updates"] == 4
    assert result["encoder_unchanged"] is False
    assert result["original_train_scalers"] == case.documents["stats.json"]
    assert result["suffix_numerical_access"] is False and result["zero_shot"] is False
    raw = case.fs.files[case.output / "inference.pt"]
    assert prefix.decode_checkpoint(raw)["kind"] == "native_prefix_transfer_inference_v1"
    with pytest.raises(FileExistsError):
        prefix.fit(case.manifest_path, case.review_path, case.output)


@pytest.mark.parametrize(
    "family,method", [("core", "direct"), ("band", "direct"), ("core", "shared_ssl")]
)
def test_actual_typed_parent_validates_encoder_before_prefix_decode(monkeypatch, family, method):
    c = support.frozen_case(prefix, monkeypatch, BUILDER, family=family, method=method)

    class AcceptedParent(Exception):
        pass

    def stop_before_arrays(admission, key, role):
        assert key == "prefix_npz" and role == "prefix"
        assert admission.identities[str(c.base / "parent-inference.pt")]
        raise AcceptedParent

    monkeypatch.setattr(prefix, "_decode", stop_before_arrays)
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("Parent check created optimizer")
    )
    monkeypatch.setattr(
        torch, "manual_seed", lambda *a, **k: pytest.fail("Parent check initialized RNG")
    )
    with pytest.raises(AcceptedParent):
        prefix.fit(c.manifest_path, c.review_path, c.output)
    assert c.output not in c.fs.dirs


@pytest.mark.parametrize(
    "bad",
    [
        "encoder_equivalence",
        "fake_ssl_kind",
        "full_finetune",
        "old_head_missing",
        "wrong_band_kind",
        "tensor_lineage",
    ],
)
def test_forged_supervised_tensor_parent_never_reaches_arrays_or_optimizer(monkeypatch, bad):
    family = "band" if bad == "wrong_band_kind" else "core"
    c = support.frozen_case(prefix, monkeypatch, BUILDER, family=family)
    if bad == "encoder_equivalence":
        key = next(k for k, t in c.selected["encoder"].items() if t.is_floating_point())
        c.selected["encoder"][key] = c.selected["encoder"][key] + 1
    elif bad == "fake_ssl_kind":
        c.selected["kind"] = "native_ssl_selected_encoder_v1"
    elif bad == "full_finetune":
        c.selected["supervised_ancestry"] = {
            **c.selected["supervised_ancestry"],
            "mode": "full_finetune",
        }
    elif bad == "old_head_missing":
        key = next(k for k in c.parent_inference["model"] if k.startswith("readout."))
        c.parent_inference["model"].pop(key)
    elif bad == "wrong_band_kind":
        c.selected["kind"] = "native_downstream_supervised_encoder_v1"
    else:
        c.parent_inference["downstream_config"] = {
            **c.parent_inference["downstream_config"],
            "seed": 13,
        }
    support.refresh_parent_artifacts(c, prefix)
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Forged tensors reached arrays")
    )
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("Forged tensors reached optimizer")
    )
    with pytest.raises(ValueError):
        prefix.fit(c.manifest_path, c.review_path, c.output)


def test_bound_structural_absence_real_npz_codec_masks_dates_and_gradients(memory_case):
    c = memory_case
    row = c.registry["rows"][0]
    support.absent_target(c.registry, row, 1, "PING_QUARANTINE")
    support.bound_gaps(c, prefix)
    c.prefix_arrays["target_observed"][0, 1] = False
    c.prefix_arrays["target_dates"][0, 1] = ""
    c.prefix_arrays["targets"][0, 1] = np.nan
    c.seal()
    admission = prefix.admit(c.manifest_path, c.review_path, c.output)
    admission.manifest["_base"] = str(c.base)
    data = prefix._decode(admission, "prefix_npz", "prefix")
    assert len(data["x"]) == 18 and not data["target_observed"][0, 1]
    assert data["target_dates"][0, 1] == ""
    model = prefix.prepare_model(c.config, c.backbone, None, c.documents["stats.json"])
    b = prefix._batch(
        data, np.arange(3), prefix._scalers(c.documents["stats.json"]), "cpu", labels=True
    )
    pred = model.forecast(b["x"], b["observed"], b["metadata"], b["query"], frozen=False)
    pred.retain_grad()
    prefix.core.pinball(pred, b["y"], b["y_observed"]).backward()
    assert pred.grad[0, 1].abs().sum() == 0
    data["targets"][0, 1] = 1e20
    b2 = prefix._batch(
        data, np.arange(3), prefix._scalers(c.documents["stats.json"]), "cpu", labels=True
    )
    assert torch.equal(b["y"], b2["y"])
    assert admission.partition["boundaries"][row["deployment"]][2:] == [
        "2026-01-05T00:00:00",
        "2026-01-06T00:00:00",
        "2026-02-11T00:00:00",
    ]


@pytest.mark.parametrize("bad", ["receipt", "gap_hash", "source_hash", "observed"])
def test_missing_or_stale_independent_source_gap_proof_before_numeric_decode(
    memory_case, monkeypatch, bad
):
    c = memory_case
    support.absent_target(c.registry, c.registry["rows"][0], 1)
    support.bound_gaps(c, prefix)
    if bad == "receipt":
        c.manifest.pop("source_gaps")
    elif bad == "gap_hash":
        c.documents["source-gaps.json"]["gaps"] = {}
    elif bad == "source_hash":
        c.documents["gap-source-metadata.json"]["reader_gap_fixture"] = False
    else:
        c.registry["rows"][0]["target_observed"][1] = True
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Unbound gap decoded arrays")
    )
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Unbound gap decoded tensor"))
    with pytest.raises((KeyError, ValueError)):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_absent_suffix_support_keeps_all_rows_explicitly_not_assessable(memory_case, monkeypatch):
    c = memory_case
    for row in c.registry["rows"]:
        if row["row_id"].endswith(tuple("-" + str(i) for i in range(888, 906))):
            support.absent_target(c.registry, row, 2, "CONFIGURATION_BOUNDARY")
    support.bound_gaps(c, prefix)
    c.seal()
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Unsupported support decoded arrays")
    )
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("Unsupported support started optimizer")
    )
    receipt = prefix.fit(c.manifest_path, c.review_path, c.output)
    assert receipt["status"] == "NOT_ASSESSABLE"
    assert receipt["evidence_kind"] == prefix.EVIDENCE
    assert len(receipt["partition"]["suffix_rows"]) == 18
    assert all(
        r["target_ids"][2] == -1 and r["target_timestamps"][2] is None
        for r in receipt["partition"]["reserved_suffix_support"]
    )
    assert c.output not in c.fs.dirs


@root_optimizer
@pytest.mark.parametrize("family", ["core", "band"])
def test_root_only_typed_direct_frozen_receipt_and_exact_resume(family):
    args = trajectory_inputs("frozen_readout", family)
    full = prefix._Trajectory(*copy.deepcopy(args))
    full.advance()
    partial = prefix._Trajectory(*copy.deepcopy(args))
    partial.advance(2)
    resumed = prefix._Trajectory(*copy.deepcopy(args))
    resumed.restore(prefix.decode_checkpoint(prefix.encode_checkpoint(partial.checkpoint())))
    resumed.advance()
    assert full.samples == resumed.samples
    assert full.candidates == resumed.candidates
    assert all(
        torch.equal(v, resumed.model.state_dict()[k]) for k, v in full.model.state_dict().items()
    )
    artifact = resumed.inference_artifact()
    assert artifact["feature_ancestor"]["selected_kind"] == prefix.SUPERVISED_KINDS[family]
    assert artifact["feature_ancestor"]["supervised_ancestry"]["ssl_only"] is False
    assert artifact["feature_ancestor"]["parent_head_reused"] is False


def test_configuration_map_cannot_relabel_unchanged_bound_native_source_metadata(monkeypatch):
    c = support.native_case(prefix, monkeypatch, BUILDER)
    catalog = c.registry["configuration_map"]
    catalog["sources"]["synthetic-site"]["configurations"]["native-180"]["channel_bounds_m"][1] = [
        0,
        200,
    ]
    for record in c.registry["intervals"].values():
        if record["configuration"] == "native-180" and record["channel"] == 1:
            record["native_bounds_m"] = [0, 200]
    for row in c.registry["rows"]:
        if row["configuration"] == "native-180":
            for meta in row["target_slot_metadata"]:
                meta[1][4] = 200 / 250
    c.seal()  # Rebind only private derived artifacts; actual source metadata stays225.
    monkeypatch.setattr(
        prefix.np,
        "load",
        lambda *a, **k: pytest.fail("Configuration mismatch decoded numerical data"),
    )
    with pytest.raises(ValueError, match="configuration metadata"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_native_map_admission_is_metadata_only_before_decode_rng_or_model(monkeypatch):
    c = support.native_case(prefix, monkeypatch, BUILDER)
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Native metadata admission decoded NPZ")
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Native metadata admission decoded tensors")
    )
    monkeypatch.setattr(
        torch, "manual_seed", lambda *a, **k: pytest.fail("Native metadata admission reset RNG")
    )
    a = prefix.admit(c.manifest_path, c.review_path, c.output)
    assert len(a.partition["fit_rows"]) == len(a.partition["suffix_rows"]) == 18
    assert set(a.partition["configuration_map"]["sources"]) == {"synthetic-site"}


@pytest.mark.parametrize(
    "key",
    ["native-configurations.json", "native-source-metadata.json", "raw.json", "source-gaps.json"],
)
def test_stale_native_metadata_bindings_block_before_all_numeric_parsing(monkeypatch, key):
    c = support.native_case(prefix, monkeypatch, BUILDER)
    c.fs.files[c.base / key] += b"tampered-source-metadata"
    monkeypatch.setattr(
        prefix.np, "load", lambda *a, **k: pytest.fail("Stale native metadata decoded NPZ")
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Stale native metadata decoded tensors")
    )
    with pytest.raises(ValueError, match="binding"):
        prefix.admit(c.manifest_path, c.review_path, c.output)


def test_exact_bound_context_mask_and_issued_processing_survive_private_codec(monkeypatch):
    c = support.native_case(prefix, monkeypatch, BUILDER)
    a = prefix.admit(c.manifest_path, c.review_path, c.output)
    a.manifest["_base"] = str(c.base)
    data = prefix._decode(a, "prefix_npz", "prefix")
    assert len(data["x"]) == 18
    assert data["context_observed"].all()
    np.testing.assert_allclose(data["query"][..., 4] * 250, 230)
    assert np.all(data["metadata"][..., 6] == 1)


@pytest.mark.parametrize("damage", ["mask", "processing", "geometry"])
def test_numeric_context_cannot_override_bound_configuration_provenance(monkeypatch, damage):
    c = support.native_case(prefix, monkeypatch, BUILDER)
    if damage == "mask":
        c.prefix_arrays["context_observed"][0, 10, 1] = False
    elif damage == "processing":
        c.prefix_arrays["metadata"][0, 1, 6] = 0
    else:
        c.prefix_arrays["metadata"][0, 1, 4] = 200 / 250
    c.seal()
    a = prefix.admit(c.manifest_path, c.review_path, c.output)
    a.manifest["_base"] = str(c.base)
    with pytest.raises(ValueError, match="conflicts"):
        prefix._decode(a, "prefix_npz", "prefix")
