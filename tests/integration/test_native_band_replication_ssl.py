"""SYNTHETIC_CORRECTNESS_ONLY CPU gates/codecs; optimizer checks ROOT-only.

NATIVE_BAND_ROOT_RUNNER_CHECKS=1 explicitly enables the synthetic runner fixture
in ROOT's authorized worktree. Builder never retries the previously denied cache.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "evidence/ssl-band-replication-builder-v2")
)
from replication_test_support import MAIN, codec, gate_fixture

from marine_echo.training import native_band_replication_ssl as band
from marine_echo.training import native_ssl as legacy

ROOT_RUNNER_CHECKS = os.environ.get("NATIVE_BAND_ROOT_RUNNER_CHECKS") == "1" and Path(
    band.__file__
).resolve().is_relative_to(MAIN / "src")


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "self",
        "case_self",
        "root",
        "architecture",
        "role",
        "method",
        "source",
        "inherited_source",
        "data",
        "split",
        "protocol",
        "runtime_config",
        "kind",
    ],
)
def test_exact_prefit_denials_before_any_numeric_loading(monkeypatch, change):
    fs, p, cfg, review, *_ = gate_fixture(monkeypatch)
    if change == "status":
        review["status"] = "APPROVED_COMPARISON_RECONSTRUCTION"
    elif change in ("self", "case_self", "root"):
        review["reviewer_session_id"] = (
            band.IMPLEMENTER_SESSION_ID.upper()
            if change == "case_self"
            else band.IMPLEMENTER_SESSION_ID
            if change == "self"
            else review["root_coordinator_session_id"]
        )
    elif change == "architecture":
        review["architecture"] = "legacy"
    elif change == "role":
        review["allowed_roles"] = ["train", "final_test"]
    elif change == "method":
        review["allowed_methods"] = []
    elif change in ("source", "inherited_source"):
        path = Path(band.__file__) if change == "source" else Path(legacy.__file__)
        review["bindings"][str(path.resolve())] = "0" * 64
    elif change in ("data", "split", "protocol"):
        fs.files[p["train" if change == "data" else change]] += b"stale"
    elif change == "runtime_config":
        changed = cfg.to_dict()
        changed["sigreg_weight"] = 0.04
        fs.json(p["config"], changed)
        review["bindings"][str(p["config"])] = band.sha256(p["config"])
    else:
        review["evidence_kind"] = "REAL_TRAIN_DEVELOPMENT_FIT"
    fs.json(p["review"], review)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Rejected prefit reached numerical loading.")
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("Rejected prefit reached weights.")
    )
    with pytest.raises(ValueError):
        band.run(
            p["train"],
            p["dev"],
            p["split"],
            p["protocol"],
            p["config"],
            p["review"],
            output=fs.base / "rejected",
            config=cfg,
            device="cpu",
            correctness_smoke=True,
        )


def test_transitive_source_closure_and_train_only_scalers_and_paired_sampling(monkeypatch):
    _fs, p, cfg, review, _, _, _, train, dev = gate_fixture(monkeypatch)
    sources = {q.name for q in band.required_sources(cfg)}
    assert {
        "native_band_temporal.py",
        "native_band_replication_ssl.py",
        "native_band_replication_downstream.py",
        "native_band_replication_acoustic.py",
        "native_temporal.py",
        "native_ssl.py",
        "native_encoder.py",
        "sigreg.py",
        "native_resources.py",
        "native_ssl_corpus.py",
        "aeon_corpus.py",
    } <= sources
    assert (
        band.check_prefit(
            p["train"],
            p["dev"],
            p["split"],
            p["protocol"],
            p["config"],
            p["review"],
            config=cfg,
            correctness_smoke=True,
        )
        == review
    )
    fitted = band.Scalers.fit(train, split_path=p["split"])
    for channel in range(4):
        raw = train["x"][:, :, channel][train["observed"][:, :, channel]].astype(np.float64)
        assert fitted.channel_mean[channel] == np.float32(raw.mean())
        assert fitted.channel_std[channel] == np.float32(raw.std())
    before = fitted.to_dict()
    dev["x"] += 1000
    assert fitted.to_dict() == before
    with pytest.raises(ValueError, match="train"):
        band.Scalers.fit(dev, split_path=p["split"])
    for step in range(6):
        indices = band.batch_indices(np.arange(20), 4, 7, "pretrain", step)
        np.testing.assert_array_equal(
            indices, legacy.batch_indices(np.arange(20), 4, 7, "pretrain", step)
        )
        paired = band.paired_indices(indices)
        assert not np.any(paired == indices)
        np.testing.assert_array_equal(np.sort(indices), np.sort(paired))


def test_scientific_recipe_cf_and_unresolved_budget_fail_closed(monkeypatch):
    from dataclasses import replace

    cfg = band.Config()
    cfg.validate()
    for key, value in {
        "architecture": "legacy",
        "history": 24,
        "latent": 128,
        "sigreg_weight": 0.04,
        "lr": 0.001,
        "seed": 17,
        "method": "cf_jepa",
        "pretrain_updates": 5999,
    }.items():
        with pytest.raises(ValueError):
            replace(cfg, **{key: value}).validate()
    fs, p, _, review, *_ = gate_fixture(monkeypatch)
    fs.json(p["config"], cfg.to_dict())
    review.update(evidence_kind="REAL_TRAIN_DEVELOPMENT_FIT")
    fs.json(p["review"], review)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("Unresolved scientific budget reached data.")
    )
    with pytest.raises(ValueError, match="budget"):
        band.run(
            p["train"],
            p["dev"],
            p["split"],
            p["protocol"],
            p["config"],
            p["review"],
            output=fs.base / "never-fit",
            config=cfg,
            device="cuda:0",
        )


def test_loader_rejects_test_role_before_numeric_members_and_rng_codec_replays(monkeypatch):
    fs, p, _, _, _, _, _, train, _ = gate_fixture(monkeypatch)
    train = {**train, "corpus_role": np.array("final_test")}
    fs.npz(p["train"], train)
    with pytest.raises(ValueError, match="test-role"):
        band.load_corpus(
            p["train"], role="train", split_hash=band.sha256(p["split"]), correctness_smoke=True
        )
    state = band.rng_state()
    expected_torch = torch.rand(4)
    expected_numpy = np.random.random(4)
    band.restore_rng(torch.load(codec(state), map_location="cpu", weights_only=True))
    assert torch.equal(expected_torch, torch.rand(4))
    np.testing.assert_array_equal(expected_numpy, np.random.random(4))


@pytest.mark.skipif(
    not ROOT_RUNNER_CHECKS,
    reason="NOT_RUN builder: prior optimizer cache denial; ROOT-only synthetic CPU fixture",
)
@pytest.mark.parametrize("method", ["shared_ssl", "masked_ssl", "permuted_ssl"])
@pytest.mark.parametrize("seed", [13, 23])
def test_root_only_full_safe_resume_and_inference_replay(monkeypatch, method, seed):
    from marine_echo.inference.native_band_replication_acoustic import NativeBandAcousticPredictor

    fs, p, cfg, _, _, _, _, _, dev = gate_fixture(monkeypatch, method=method, seed=seed)
    # Transport only the NEW module's checkpoints with actual Torch codecs.
    # No legacy factory or fitter is patched; original optimizer/schedule execute.
    monkeypatch.setattr(band, "atomic_checkpoint", fs.checkpoint)

    def run(output, resume=None, stop_after=None):
        return band.run(
            p["train"],
            p["dev"],
            p["split"],
            p["protocol"],
            p["config"],
            p["review"],
            output=output,
            config=cfg,
            device="cpu",
            correctness_smoke=True,
            resume=resume,
            stop_after=stop_after,
        )

    first, continued = fs.base / "whole", fs.base / "continued"
    run(first)
    interrupted = run(continued, stop_after=1)
    assert interrupted["steps"] == 1
    run(continued, resume=continued / "latest.pt")
    a = torch.load(codec(torch.load(first / "latest.pt", weights_only=True)), weights_only=True)
    b = torch.load(continued / "latest.pt", weights_only=True)
    assert a["kind"] == b["kind"] == "native_band_replication_ssl_resume_v2"
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert a["state"]["sequence"] == b["state"]["sequence"]
    assert a["pre_scheduler"] == b["pre_scheduler"]
    assert a["read_scheduler"] == b["read_scheduler"]
    predictor = NativeBandAcousticPredictor(first / "inference.pt")
    values = predictor.forecast(dev["x"], dev["observed"], dev["metadata"], dev["query"])
    with np.load(first / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(values, saved["predictions"], atol=2e-5, rtol=1e-5)
    report = json.loads((first / "run.json").read_text())
    assert (
        report["optimizer_steps_total"]
        == report["pretrain_steps"] + report["readout_steps_all_probes"]
    )
    assert report["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
