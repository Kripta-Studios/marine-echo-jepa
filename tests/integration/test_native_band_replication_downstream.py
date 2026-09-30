"""SYNTHETIC_CORRECTNESS_ONLY CPU gradients/gates; optimizer fixtures ROOT-only."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
from replication_test_support import MAIN, gate_fixture

from marine_echo.training import native_band_replication_downstream as downstream
from marine_echo.training import native_band_replication_ssl as core

ROOT_RUNNER_CHECKS = os.environ.get("NATIVE_BAND_ROOT_RUNNER_CHECKS") == "1" and Path(
    core.__file__
).resolve().is_relative_to(MAIN / "src")


def test_typed_original_v1_parent_keeps_kind_source_scaler_and_exact_encoder(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-builder-v1"))
    from band_test_support import gate_fixture as original_fixture

    fs, _p, cfg, _review, inputs, dcfg, review, *_ = original_fixture(monkeypatch)
    review["allowed_seeds"] = [7]
    review["bindings"].update(
        {str(path): core.sha256(path) for path in downstream.required_paths(inputs, cfg)}
    )
    fs.json(inputs.review, review)
    downstream.check_prefit(inputs, dcfg, cfg, correctness_smoke=True)
    selected = downstream._load_ancestor(inputs, cfg)
    assert selected["kind"] == "native_band_ssl_selected_encoder_v1"
    before = {k: v.clone() for k, v in selected["encoder"].items()}
    model = downstream.prepare_model(dcfg, cfg, selected)
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())
    assert selected["bindings"] != review["bindings"]
    forged = dict(selected, kind="native_band_replication_ssl_selected_encoder_v2")
    fs.checkpoint(inputs.encoder, forged)
    with pytest.raises(ValueError, match="kind"):
        downstream._load_ancestor(inputs, cfg)


@pytest.mark.parametrize("mode", downstream.MODES)
def test_frozen_state_or_full_direct_gradients_and_matched_head_sampler(monkeypatch, mode):
    method = "direct" if mode == "direct_end_to_end" else "shared_ssl"
    _fs, p, cfg, _, inputs, dcfg, _, train, _ = gate_fixture(monkeypatch, mode=mode, method=method)
    downstream.check_prefit(inputs, dcfg, cfg, correctness_smoke=True)
    selected = downstream._load_ancestor(inputs, cfg)
    model = downstream.prepare_model(dcfg, cfg, selected)
    random = core.initialize_model(dcfg.seed, **core.model_dimensions(cfg), method=method)
    assert all(
        torch.equal(v, random.readout.state_dict()[k])
        for k, v in model.readout.state_dict().items()
    )
    scalers = core.Scalers.fit(train, split_path=p["split"])
    state = core.cpu_state(model.encoder)
    ids = downstream.supervised_indices(np.arange(20), dcfg, 0)
    for other in downstream.MODES:
        np.testing.assert_array_equal(
            ids, downstream.supervised_indices(np.arange(20), replace(dcfg, mode=other), 0)
        )
    batch = core.tensor_batch(train, ids, scalers, 96, "cpu")
    loss = core.pinball(
        downstream._forecast_train(model, batch, mode), batch["y"], batch["y_observed"]
    )
    loss.backward()
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in state.items())
    assert model.readout.net[0].weight.grad.abs().sum() > 0
    if mode == "frozen_readout":
        assert not model.encoder.training
        assert all(not p.requires_grad and p.grad is None for p in model.encoder.parameters())
    else:
        assert model.encoder.training
        assert model.encoder.patch_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.predictor.parameters())


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "self",
        "role",
        "architecture",
        "source",
        "data",
        "cohort",
        "ancestor_source",
        "ancestor_config",
        "ancestor_membership",
        "method",
    ],
)
def test_downstream_role_source_and_ancestry_denied_before_numeric_load(monkeypatch, change):
    fs, p, cfg, _, inputs, dcfg, review, _, _ = gate_fixture(monkeypatch)
    if change == "status":
        review["status"] = "APPROVED_PREFIT"
    elif change == "self":
        review["reviewer_session_id"] = core.IMPLEMENTER_SESSION_ID.upper()
    elif change == "role":
        review["allowed_roles"] = ["final_test"]
    elif change == "architecture":
        review["architecture"] = "legacy"
    elif change == "source":
        review["bindings"][str(Path(core.__file__).resolve())] = "0" * 64
    elif change in ("data", "cohort"):
        fs.files[p["dev" if change == "data" else "dev_cohort"]] += b"stale"
    elif change in ("ancestor_source", "ancestor_config", "ancestor_membership"):
        parent = inputs.encoder.parent
        record = json.loads((parent / "run.json").read_text())
        if change == "ancestor_source":
            record["bindings"][str(Path(core.__file__).resolve())] = "0" * 64
        elif change == "ancestor_config":
            record["config"]["architecture"] = "legacy"
        else:
            membership = json.loads((parent / "membership.json").read_text())
            membership["train_deployments"][0] = "synthetic-development"
            fs.json(parent / "membership.json", membership)
            record["membership_sha256"] = core.sha256(parent / "membership.json")
            review["bindings"][str(parent / "membership.json")] = core.sha256(
                parent / "membership.json"
            )
        fs.json(parent / "run.json", record)
        review["bindings"][str(parent / "run.json")] = core.sha256(parent / "run.json")
    else:
        review["allowed_methods"] = []
    fs.json(inputs.review, review)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Rejected ancestry reached numerical data.")
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Rejected ancestry reached weight loading.")
    )
    with pytest.raises(ValueError):
        downstream.run(
            inputs,
            dcfg,
            output=fs.base / "denied",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=cfg,
        )


def test_legacy_ancestor_kind_and_changed_scalers_are_rejected_with_safe_codecs(monkeypatch):
    fs, _, cfg, _, inputs, _, _, _, _ = gate_fixture(monkeypatch)
    saved = torch.load(inputs.encoder, weights_only=True)
    saved["kind"] = "native_ssl_selected_encoder_v1"
    fs.checkpoint(inputs.encoder, saved)
    with pytest.raises(ValueError, match="kind"):
        downstream._load_ancestor(inputs, cfg)
    saved["kind"] = "native_band_replication_ssl_selected_encoder_v2"
    saved["scalers"]["channel_std"][0] = 0
    fs.checkpoint(inputs.encoder, saved)
    with pytest.raises(ValueError, match="scaler"):
        downstream._load_ancestor(inputs, cfg)


@pytest.mark.skipif(
    not ROOT_RUNNER_CHECKS,
    reason="NOT_RUN builder: prior optimizer cache denial; ROOT-only synthetic CPU fixture",
)
@pytest.mark.parametrize("mode", downstream.MODES)
@pytest.mark.parametrize("seed", [13, 23])
def test_root_only_resume_weights_changes_frozen_identity_counts_and_forecast(
    monkeypatch, mode, seed
):
    method = "direct" if mode == "direct_end_to_end" else "shared_ssl"
    fs, _p, cfg, _, inputs, dcfg, _, _train, dev = gate_fixture(
        monkeypatch, mode=mode, method=method, seed=seed
    )
    monkeypatch.setattr(core, "atomic_checkpoint", fs.checkpoint)
    original = core.cpu_state(
        downstream.prepare_model(dcfg, cfg, downstream._load_ancestor(inputs, cfg)).encoder
    )

    def run(output, resume=None, stop_after=None):
        return downstream.run(
            inputs,
            dcfg,
            output=output,
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=cfg,
            resume=resume,
            stop_after=stop_after,
        )

    whole, resumed = fs.base / "whole-downstream", fs.base / "resumed-downstream"
    report = run(whole)
    run(resumed, stop_after=1)
    run(resumed, resume=resumed / "latest.pt")
    a, b = (torch.load(folder / "latest.pt", weights_only=True) for folder in (whole, resumed))
    assert a["kind"] == b["kind"] == "native_band_replication_downstream_resume_v2"
    assert a["state"]["sequence"] == b["state"]["sequence"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    changed = any(not torch.equal(v, a["model"]["encoder." + k]) for k, v in original.items())
    assert changed == (mode != "frozen_readout")
    if mode == "frozen_readout":
        assert fs.files[whole / "selected_encoder.pt"] == fs.files[inputs.encoder]
    else:
        encoder = torch.load(whole / "selected_encoder.pt", weights_only=True)
        assert encoder["kind"] == "native_band_replication_downstream_supervised_encoder_v2"
        assert encoder["supervised_ancestry"]["ssl_only"] is False
    assert report["optimizer_steps_total"] == report["supervised_updates"] == 4
    assert report["sample_presentations"] == 16
    assert report["label_observations_processed"] == 48
    infer = downstream.load_inference(whole / "inference.pt")
    with np.load(whole / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(
            infer.forecast(dev["x"], dev["observed"], dev["metadata"], dev["query"]),
            saved["predictions"],
            atol=2e-5,
            rtol=1e-5,
        )
    assert all(not p.requires_grad for p in infer.model.parameters())
