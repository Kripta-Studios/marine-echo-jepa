"""Synthetic-only runner checks. Temporary artifacts remain in the allowed evidence tree."""

import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_echo.training.native_ssl import (
    Config,
    Scalers,
    batch_indices,
    check_prefit,
    daily_metrics,
    paired_indices,
    restore_rng,
    rng_state,
    run,
    sha256,
)


def corpus(n=24, role="train"):
    rng = np.random.default_rng(91)
    x = rng.normal(-80, 3, (n, 96, 4)).astype(np.float32)
    mask = rng.random(x.shape) > 0.1
    meta = np.zeros((n, 4, 10), np.float32)
    meta[:, :, 4] = 230 / 250
    query = np.tile(meta[:, :1], (1, 3, 1))
    query[:, :, -1] = [1, 3, 6]
    cutoff = np.arange(n) * 120 + 96
    return {
        "x": x,
        "observed": mask,
        "metadata": meta,
        "future": rng.normal(-80, 3, (n, 3, 4, 4)).astype(np.float32),
        "future_observed": np.ones((n, 3, 4, 4), bool),
        "y": rng.normal(-80, 3, (n, 3)).astype(np.float32),
        "y_observed": np.ones((n, 3), bool),
        "query": query,
        "row_id": np.array([f"synthetic-{role}-{i}" for i in range(n)]),
        "deployment": np.full(n, f"synthetic-{role}"),
        "archive_sha256": np.full(n, f"synthetic-archive-{role}"),
        "cutoff": cutoff,
        "target_dates": np.full((n, 3), "2001-01-01"),
        "corpus_role": np.asarray(role),
        "split_sha256": np.asarray("synthetic"),
        "ssl_eligible": np.ones(n, bool),
        "context_ids": cutoff[:, None] - np.arange(95, -1, -1),
        "future_ids": cutoff[:, None, None] + np.array([1, 3, 6])[None, :, None] + np.arange(4),
    }


def test_train_only_scalers_ignore_missing_fills(artifact_dir):
    config = Config()
    paths = approved_fixture(artifact_dir, config)
    split = paths[2]
    train = corpus()
    train["split_sha256"] = np.asarray(sha256(split))
    train["x"][~train["observed"]] = 1e8
    scaler = Scalers.fit(train, split_path=split)
    expected = np.array([train["x"][:, :, c][train["observed"][:, :, c]].mean() for c in range(4)])
    np.testing.assert_allclose(scaler.channel_mean, expected, rtol=1e-6)
    dev = corpus(role="development")
    with pytest.raises(ValueError, match="train"):
        Scalers.fit(dev, split_path=split)
    scaled = scaler.channels(train["x"], train["observed"])
    assert np.all(scaled[~train["observed"]] == 0)
    assert np.isfinite(scaled).all()


def test_batch_sequence_target_multiset_nonidentity():
    for step in range(15):
        aligned = batch_indices(np.arange(31), 8, 7, "pretrain", step)
        control = batch_indices(np.arange(31), 8, 7, "pretrain", step)
        np.testing.assert_array_equal(aligned, control)
        paired = paired_indices(control)
        assert np.all(control != paired)
        np.testing.assert_array_equal(np.sort(aligned), np.sort(paired))


def test_independent_daily_reconstruction_equal_dates_horizons_deployments():
    n = 60
    y = np.arange(n * 3).reshape(n, 3) / 20
    scales = np.concatenate([np.ones(18), np.full(22, 5.0), np.full(20, 10.0)])
    predictions = np.repeat(y[..., None], 5, axis=-1) + (np.arange(5) - 1) * scales[:, None, None]
    mask = np.ones((n, 3), bool)
    dates = np.full((n, 3), "2001-01-01")
    dates[18:40] = "2001-01-02"
    deployment = np.array(["a"] * 40 + ["b"] * 20)
    mask[0:2, 1] = False  # first day at h3 drops below eighteen
    result = daily_metrics(predictions, y, mask, dates, deployment)
    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    losses = np.maximum(
        q * (y[..., None] - predictions), (q - 1) * (y[..., None] - predictions)
    ).mean(-1)
    by_deployment = []
    for d in np.unique(deployment):
        by_horizon = []
        for h in range(3):
            scores = []
            for day in np.unique(dates[:, h]):
                selected = (deployment == d) & (dates[:, h] == day) & mask[:, h]
                if selected.sum() >= 18:
                    scores.append(losses[selected, h].mean())
            by_horizon.append(np.mean(scores))
        by_deployment.append(np.mean(by_horizon))
    assert result["primary_pinball"] == pytest.approx(np.mean(by_deployment), abs=1e-12)
    assert abs(result["primary_pinball"] - losses[mask].mean()) > 0.01
    assert result["coverage"]["scored_rows"][1] == 42


@pytest.fixture
def artifact_dir(monkeypatch):
    """Real codecs, in-memory transport: managed Python file writes are blocked.

    This verifies serialized state and runner behavior, not durable filesystem IO.
    Root can set MARINE_NATIVE_DURABLE_TESTS=1 in its authorized environment.
    """
    import tempfile
    import uuid

    base = Path("evidence/ssl-builder-v1/synthetic-tests")
    if os.environ.get("MARINE_NATIVE_DURABLE_TESTS") == "1":
        base.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="case-", dir=base))
    root = (base / f"memory-{uuid.uuid4()}").resolve()
    files = {}

    def scoped(path):
        return isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(root)

    originals = {
        name: getattr(Path, name) for name in ("mkdir", "write_text", "read_text", "exists")
    }

    def write_text(path, value, *args, **kwargs):
        if scoped(path):
            files[str(path.resolve())] = value.encode("utf-8")
            return len(value)
        return originals["write_text"](path, value, *args, **kwargs)

    def read_text(path, *args, **kwargs):
        if scoped(path):
            return files[str(path.resolve())].decode("utf-8")
        return originals["read_text"](path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda path, *a, **k: None if scoped(path) else originals["mkdir"](path, *a, **k),
    )
    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: str(path.resolve()) in files if scoped(path) else originals["exists"](path),
    )
    from marine_echo.training import native_ssl

    real_hash = native_ssl.sha256
    monkeypatch.setattr(
        native_ssl,
        "sha256",
        lambda path: (
            __import__("hashlib").sha256(files[str(Path(path).resolve())]).hexdigest()
            if scoped(path)
            else real_hash(path)
        ),
    )
    # Imported test helper must use the same codec transport.
    monkeypatch.setattr(__import__(__name__, fromlist=["sha256"]), "sha256", native_ssl.sha256)
    save, load, replace = torch.save, torch.load, os.replace

    def checkpoint_save(value, path, *args, **kwargs):
        if scoped(path):
            stream = io.BytesIO()
            save(value, stream, *args, **kwargs)
            files[str(Path(path).resolve())] = stream.getvalue()
            return
        return save(value, path, *args, **kwargs)

    def checkpoint_load(path, *args, **kwargs):
        if scoped(path):
            return load(io.BytesIO(files[str(Path(path).resolve())]), *args, **kwargs)
        return load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", checkpoint_save)
    monkeypatch.setattr(torch, "load", checkpoint_load)
    monkeypatch.setattr(
        os,
        "replace",
        lambda src, dst: (
            files.__setitem__(str(Path(dst).resolve()), files.pop(str(Path(src).resolve())))
            if scoped(src) and scoped(dst)
            else replace(src, dst)
        ),
    )
    np_load = np.load
    monkeypatch.setattr(
        np,
        "load",
        lambda path, *a, **k: (
            np_load(io.BytesIO(files[str(Path(path).resolve())]), *a, **k)
            if scoped(path)
            else np_load(path, *a, **k)
        ),
    )
    for name in ("savez", "savez_compressed"):
        original = getattr(np, name)

        def array_save(path, *args, _original=original, **kwargs):
            if scoped(path):
                stream = io.BytesIO()
                _original(stream, *args, **kwargs)
                files[str(Path(path).resolve())] = stream.getvalue()
                return
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(np, name, array_save)
    return root


def approved_fixture(base, config):
    train, dev = base / "train.npz", base / "dev.npz"
    split, protocol, config_path = base / "split.json", base / "protocol.md", base / "config.json"
    split.write_text(
        json.dumps(
            {
                "synthetic": True,
                "sources": [
                    {
                        "deployment": f"synthetic-{role}",
                        "archive_sha256": f"synthetic-archive-{role}",
                        "role": role,
                    }
                    for role in ("train", "development")
                ],
            }
        )
    )
    protocol.write_text("Synthetic correctness protocol; no real data.")
    config_path.write_text(json.dumps(config.to_dict(), sort_keys=True))
    for path, role in ((train, "train"), (dev, "development")):
        arrays = corpus(role=role)
        arrays["split_sha256"] = np.asarray(sha256(split))
        np.savez(path, **arrays)
    from marine_echo.training.native_ssl import required_sources

    paths = [train, dev, split, protocol, config_path, *required_sources(config)]
    review = {
        "status": "APPROVED_PREFIT",
        "reviewer_session_id": "synthetic-independent-fixture",
        "implementer_session_id": "synthetic-implementer-fixture",
        "allowed_methods": [config.method],
        "train_npz_sha256": sha256(train),
        "dev_npz_sha256": sha256(dev),
        "split_sha256": sha256(split),
        "protocol_sha256": sha256(protocol),
        "bindings": {str(p.resolve()): sha256(p) for p in paths},
    }
    review_path = base / "review.json"
    review_path.write_text(json.dumps(review))
    return train, dev, split, protocol, config_path, review_path


def test_prefit_fail_closed_and_test_corpus(artifact_dir):
    config = Config(
        pretrain_updates=1, readout_updates=1, batch_size=4, width=24, latent=8, blocks=1, heads=4
    )
    paths = approved_fixture(artifact_dir, config)
    _train, _dev, _split, protocol, _config_path, review_path = paths
    check_prefit(*paths, config=config)
    review = json.loads(review_path.read_text())
    for change in (
        {"bindings": {}},
        {"status": "REJECTED"},
        {"reviewer_session_id": "synthetic-implementer-fixture"},
        {"train_npz_sha256": "bad"},
    ):
        review_path.write_text(json.dumps(review | change))
        with pytest.raises(ValueError):
            check_prefit(*paths, config=config)
    review_path.write_text(json.dumps(review))
    protocol.write_text("modified protocol")
    with pytest.raises(ValueError, match="binding"):
        check_prefit(*paths, config=config)


def test_rng_cpu_continuation():
    torch.manual_seed(9)
    np.random.seed(9)
    state = rng_state()
    a, b = torch.randn(5), np.random.normal(size=5)
    restore_rng(state)
    assert torch.equal(a, torch.randn(5))
    np.testing.assert_array_equal(b, np.random.normal(size=5))


def test_source_membership_required_for_scaler_fit(artifact_dir):
    paths = approved_fixture(artifact_dir, Config())
    split = paths[2]
    train = corpus()
    train["split_sha256"] = np.asarray(sha256(split))
    for key, wrong in (
        ("archive_sha256", "heldout-archive"),
        ("deployment", "synthetic-development"),
    ):
        changed = {**train, key: np.full(24, wrong)}
        with pytest.raises(ValueError, match="membership"):
            Scalers.fit(changed, split_path=split)
    with pytest.raises(ValueError, match="provenance"):
        Scalers.fit(
            {key: value for key, value in train.items() if key != "archive_sha256"},
            split_path=split,
        )


def test_each_source_binding_required_before_any_fit(artifact_dir, monkeypatch):
    config = Config(
        pretrain_updates=1, readout_updates=1, width=24, latent=8, blocks=1, heads=4, batch_size=4
    )
    paths = approved_fixture(artifact_dir, config)
    original = json.loads(paths[-1].read_text())
    from marine_echo.training.native_ssl import required_sources

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Rejected review reached scaler fitting.")

    monkeypatch.setattr(Scalers, "fit", unexpected_fit)
    for path in [*paths[:5], *required_sources(config)]:
        review = {
            **original,
            "bindings": {k: v for k, v in original["bindings"].items() if k != str(path.resolve())},
        }
        paths[-1].write_text(json.dumps(review))
        with pytest.raises(ValueError, match="binding"):
            run(
                *paths,
                output=artifact_dir / "rejected",
                config=config,
                device="cpu",
                correctness_smoke=True,
            )


def test_test_role_is_rejected_before_numeric_read_or_scalers(artifact_dir, monkeypatch):
    config = Config(
        pretrain_updates=1, readout_updates=1, width=24, latent=8, blocks=1, heads=4, batch_size=4
    )
    paths = approved_fixture(artifact_dir, config)
    train, _, split, _, _, review_path = paths
    arrays = corpus(role="final_test")
    arrays["split_sha256"] = np.asarray(sha256(split))
    np.savez(train, **arrays)
    review = json.loads(review_path.read_text())
    review["train_npz_sha256"] = sha256(train)
    review["bindings"][str(train.resolve())] = sha256(train)
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(Scalers, "fit", lambda *a, **k: pytest.fail("Test corpus reached fit."))
    with pytest.raises(ValueError, match="test-role"):
        run(
            *paths,
            output=artifact_dir / "rejected-test",
            config=config,
            device="cpu",
            correctness_smoke=True,
        )


def test_direct_ceiling_and_matched_selection_opportunities():
    with pytest.raises(ValueError, match="direct"):
        Config(method="direct", pretrain_updates=6000).validate()
    direct = Config(method="direct", pretrain_updates=3000, pretrain_cadence=750)
    direct.validate()
    shared = Config()
    shared.validate()
    assert (
        direct.pretrain_updates // direct.pretrain_cadence
        == shared.pretrain_updates // shared.pretrain_cadence
        == 4
    )
    with pytest.raises(ValueError, match="Planned total"):
        Config(readout_updates=2000).validate()


def test_metadata_ids_and_shared_support_do_not_silently_drop_deployments():
    from marine_echo.training.native_ssl import validate_corpus

    data = corpus()
    invalid = {**data, "future_ids": data["future_ids"] - 3}
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_corpus(invalid, role="train", correctness_smoke=True)
    actual = np.repeat(data["metadata"][:, None], 96, axis=1)
    actual[0, 0, 0, 4] = 0.8
    data["observed"][0, 0, 0] = True
    with pytest.raises(ValueError, match="broadcast"):
        validate_corpus({**data, "context_metadata": actual}, role="train", correctness_smoke=True)
    predictions = np.zeros((24, 3, 5))
    support = np.ones((24, 3), bool)
    support[:, 2] = False
    result = daily_metrics(
        predictions,
        data["y"],
        data["y_observed"],
        data["target_dates"],
        data["deployment"],
        support=support,
    )
    assert result["primary_pinball"] is None
    assert result["coverage"]["scored_rows"] == [24, 24, 0]


@pytest.mark.parametrize(
    "method", ["shared_ssl", "masked_ssl", "direct", "random_frozen", "permuted_ssl", "cf_jepa"]
)
@pytest.mark.parametrize("interrupt_step", [1, 2])
def test_full_resume_and_inference_replay(artifact_dir, method, interrupt_step):
    torch.set_num_threads(1)
    config = Config(
        method=method,
        pretrain_updates=2,
        readout_updates=2,
        batch_size=4,
        width=24,
        latent=8,
        blocks=1,
        heads=4,
        pretrain_cadence=1,
        readout_cadence=1,
        cf_width=24,
        cf_latent=8,
        cf_blocks=1,
    )
    paths = approved_fixture(artifact_dir, config)
    output = artifact_dir / "run"
    run(
        *paths,
        output=output,
        config=config,
        device="cpu",
        correctness_smoke=True,
        stop_after=interrupt_step,
    )
    checkpoint = output / "latest.pt"
    resumed = artifact_dir / "resumed"
    run(
        *paths,
        output=resumed,
        config=config,
        device="cpu",
        correctness_smoke=True,
        resume=checkpoint,
    )
    continuous = artifact_dir / "continuous"
    run(*paths, output=continuous, config=config, device="cpu", correctness_smoke=True)
    first = torch.load(resumed / "inference.pt", weights_only=True)
    second = torch.load(continuous / "inference.pt", weights_only=True)
    for key, value in first["model"].items():
        assert torch.equal(value, second["model"][key]), key
    from marine_echo.training.native_ssl import predict_from_checkpoint

    data = corpus(role="development")
    replay = predict_from_checkpoint(resumed / "inference.pt", data)
    # A reusable issuance endpoint must not even require assessment arrays.
    inputs_only = {key: data[key] for key in ("x", "observed", "metadata", "query")}
    np.testing.assert_array_equal(
        predict_from_checkpoint(resumed / "inference.pt", inputs_only), replay
    )
    with np.load(resumed / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(replay, saved["predictions"])
        np.testing.assert_array_equal(saved["query"], data["query"])
    assert (
        json.loads((resumed / "run.json").read_text())["evidence_kind"]
        == "SYNTHETIC_CORRECTNESS_ONLY"
    )
    a = torch.load(resumed / "complete.pt", weights_only=True)
    b = torch.load(continuous / "complete.pt", weights_only=True)
    assert a["state"]["sequence"] == b["state"]["sequence"]
    assert a["state"]["records"] == b["state"]["records"]
    assert a["pre_scheduler"] == b["pre_scheduler"]
    assert a["read_scheduler"] == b["read_scheduler"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert not a["rng"]["cuda"] and not b["rng"]["cuda"]
