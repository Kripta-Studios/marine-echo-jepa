"""SYNTHETIC_CORRECTNESS_ONLY CPU checks using main's frozen core helpers.

Serialization uses real codecs and an in-memory filesystem transport. Durable
artifact checks are parent-owned; previously denied writes are never retried.
"""

import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

# Import main's regular package first; extend only its module search path to find
# the new builder module. No main file, junction, or installed package is changed.
BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
import marine_echo.training

marine_echo.training.__path__.append(str(BUILDER / "src/marine_echo/training"))
from marine_echo.training import native_downstream as ds


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(1)
    assert Path(ds.core.__file__).resolve() == MAIN / "src/marine_echo/training/native_ssl.py"
    assert not torch.cuda.is_initialized()


def arrays(role):
    n = 24
    rng = np.random.default_rng(19)
    x = rng.normal(-80, 4, (n, 96, 4)).astype(np.float32)
    metadata = np.zeros((n, 4, 10), np.float32)
    metadata[:, :, 4] = 0.92
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[:, :, -1] = [1, 3, 6]
    cutoff = np.arange(n) * 120 + 96
    return {
        "x": x,
        "observed": np.ones_like(x, bool),
        "metadata": metadata,
        "future": np.zeros((n, 3, 4, 4), np.float32),
        "future_observed": np.ones((n, 3, 4, 4), bool),
        "y": (x[:, -1, :3] + [1, 2, 3]).astype(np.float32),
        "y_observed": np.ones((n, 3), bool),
        "query": query,
        "row_id": np.array([f"synthetic-{role}-{i}" for i in range(n)]),
        "deployment": np.full(n, f"synthetic-{role}"),
        "archive_sha256": np.full(n, f"synthetic-archive-{role}"),
        "cutoff": cutoff,
        "corpus_role": np.asarray(role),
        "target_dates": np.full((n, 3), "2001-01-01"),
        "ssl_eligible": np.ones(n, bool),
        "context_ids": cutoff[:, None] - np.arange(95, -1, -1),
        "future_ids": cutoff[:, None, None] + np.array([1, 3, 6])[None, :, None] + np.arange(4),
    }


@pytest.fixture
def memory(monkeypatch):
    """No durable artifact IO; real torch weights-only and NumPy NPZ codecs."""
    if os.environ.get("MARINE_NATIVE_DOWNSTREAM_DURABLE_TESTS") == "1":
        if BUILDER != MAIN:
            raise RuntimeError(
                "Durable checks are coordinator-owned; never retry denied builder writes."
            )
        directory = MAIN / "evidence/ssl-research-v1/downstream-synthetic-durable"
        directory.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="case-", dir=directory))
        (root / "parent").mkdir()
        return root, {}
    root = (BUILDER / "evidence/ssl-downstream-builder-v1/virtual").resolve()
    files = {}
    scoped = lambda p: isinstance(p, (str, Path)) and Path(p).resolve().is_relative_to(root)
    key = lambda p: str(Path(p).resolve())
    original_methods = {
        n: getattr(Path, n)
        for n in ("exists", "mkdir", "read_text", "write_text", "read_bytes", "open", "iterdir")
    }
    monkeypatch.setattr(
        Path, "exists", lambda p: key(p) in files if scoped(p) else original_methods["exists"](p)
    )
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda p, *a, **k: None if scoped(p) else original_methods["mkdir"](p, *a, **k),
    )
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda p: (
            iter([Path(k) for k in files if Path(k).parent == p.resolve()])
            if scoped(p)
            else original_methods["iterdir"](p)
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **k: (
            files[key(p)].decode() if scoped(p) else original_methods["read_text"](p, *a, **k)
        ),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda p, s, *a, **k: (
            files.__setitem__(key(p), s.encode())
            if scoped(p)
            else original_methods["write_text"](p, s, *a, **k)
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: files[key(p)] if scoped(p) else original_methods["read_bytes"](p),
    )

    class Writer(io.BytesIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            files[key(self.path)] = self.getvalue()
            super().close()

    def open_file(path, mode="r", *args, **kwargs):
        if not scoped(path):
            return original_methods["open"](path, mode, *args, **kwargs)
        if "x" in mode and key(path) in files:
            raise FileExistsError(path)
        return Writer(path) if any(c in mode for c in "wx") else io.BytesIO(files[key(path)])

    monkeypatch.setattr(Path, "open", open_file)
    for module in (ds, ds.core):
        monkeypatch.setattr(
            module,
            "sha256",
            lambda p: hashlib.sha256(
                files[key(p)] if scoped(p) else Path(p).read_bytes()
            ).hexdigest(),
        )
    save, load, replace = torch.save, torch.load, os.replace

    def save_torch(value, path, *args, **kwargs):
        if not scoped(path):
            return save(value, path, *args, **kwargs)
        stream = io.BytesIO()
        save(value, stream, *args, **kwargs)
        files[key(path)] = stream.getvalue()

    monkeypatch.setattr(torch, "save", save_torch)
    monkeypatch.setattr(
        torch,
        "load",
        lambda p, *a, **k: (
            load(io.BytesIO(files[key(p)]), *a, **k) if scoped(p) else load(p, *a, **k)
        ),
    )
    monkeypatch.setattr(
        os,
        "replace",
        lambda a, b: files.__setitem__(key(b), files.pop(key(a))) if scoped(a) else replace(a, b),
    )
    np_load = np.load
    monkeypatch.setattr(
        np,
        "load",
        lambda p, *a, **k: (
            np_load(io.BytesIO(files[key(p)]), *a, **k) if scoped(p) else np_load(p, *a, **k)
        ),
    )
    for name in ("savez", "savez_compressed"):
        original = getattr(np, name)

        def save_arrays(path, *args, _original=original, **kwargs):
            if not scoped(path):
                return _original(path, *args, **kwargs)
            stream = io.BytesIO()
            _original(stream, *args, **kwargs)
            files[key(path)] = stream.getvalue()

        monkeypatch.setattr(np, name, save_arrays)
    return root, files


def fixture_run(root, *, mode="frozen_readout", method="shared_ssl"):
    if mode == "direct_end_to_end":
        method = "direct"
    config = ds.DownstreamConfig(mode=mode, method=method, updates=3, cadence=1, batch_size=4)
    core_config = ds.core.Config(
        method=method, width=24, latent=8, blocks=1, heads=4, cf_width=24, cf_latent=8, cf_blocks=1
    )
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "role": role,
                        "deployment": f"synthetic-{role}",
                        "archive_sha256": f"synthetic-archive-{role}",
                    }
                    for role in ("train", "development")
                ]
            }
        )
    )
    for role in ("train", "development"):
        data = arrays(role)
        data["split_sha256"] = np.asarray(ds.sha256(split))
        path = root / f"{role}.npz"
        np.savez(path, **data)
        (root / f"{role}.json").write_text(
            json.dumps(
                {
                    "role": role,
                    "npz_sha256": ds.sha256(path),
                    "split_sha256": ds.sha256(split),
                    "issued": 24,
                    "source_reports": [{"deployment": f"synthetic-{role}", "issued": 24}],
                }
            )
        )
    protocol, adr, cfg = root / "protocol.md", root / "adr0016.md", root / "config.json"
    protocol.write_text("SYNTHETIC_CORRECTNESS_ONLY downstream protocol")
    adr.write_text("SYNTHETIC_CORRECTNESS_ONLY ADR0016 fixture")
    cfg.write_text(json.dumps(config.to_dict()))
    data = arrays("train")
    data["split_sha256"] = np.asarray(ds.sha256(split))
    scalers = ds.core.Scalers.fit(data, split_path=split)
    model = ds.core.initialize_model(7, **ds.core.model_dimensions(core_config), method=method)
    core_cfg = root / "parent/config.json"
    core_cfg.write_text(json.dumps(core_config.to_dict()))
    core_bindings = {
        str(p.resolve()): ds.sha256(p)
        for p in [
            root / "train.npz",
            root / "development.npz",
            split,
            adr,
            core_cfg,
            *ds.core.required_sources(core_config),
        ]
    }
    ancestor = root / "parent/selected_encoder.pt"
    common = {
        "config": core_config.to_dict(),
        "scalers": scalers.to_dict(),
        "bindings": core_bindings,
        "selected_pretrain_step": 1500,
    }
    torch.save(
        {
            "kind": "native_ssl_selected_encoder_v1",
            "encoder": ds.core.cpu_state(model.encoder),
            **common,
        },
        ancestor,
    )
    inference = root / "parent/inference.pt"
    torch.save(
        {
            "kind": "native_ssl_weights_only_inference_v1",
            "model": ds.core.cpu_state(model),
            **common,
        },
        inference,
    )
    membership = root / "parent/membership.json"
    membership.write_text(
        json.dumps(
            {
                "train_row_ids": data["row_id"].tolist(),
                "train_deployments": data["deployment"].tolist(),
                "train_archive_sha256": data["archive_sha256"].tolist(),
                "ssl_eligible_indices": list(range(24)),
                "supervised_indices": list(range(24)),
                "sequence": [],
            }
        )
    )
    parent_review = root / "parent/review.json"
    parent_review.write_text(
        json.dumps(
            {
                "status": "APPROVED_PREFIT",
                "reviewer_session_id": "synthetic-parent-reviewer",
                "implementer_session_id": "synthetic-parent-implementer",
                "allowed_methods": [method],
                "bindings": core_bindings,
                "train_npz_sha256": ds.sha256(root / "train.npz"),
                "dev_npz_sha256": ds.sha256(root / "development.npz"),
                "split_sha256": ds.sha256(split),
                "protocol_sha256": ds.sha256(adr),
            }
        )
    )
    parent = root / "parent/run.json"
    parent.write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "config": core_config.to_dict(),
                "bindings": core_bindings,
                "selected_pretrain_step": 1500,
                "membership_sha256": ds.sha256(membership),
                "inference_sha256": ds.sha256(inference),
                "review_sha256": ds.sha256(parent_review),
            }
        )
    )
    inputs = ds.RunInputs(
        train=root / "train.npz",
        dev=root / "development.npz",
        train_cohort=root / "train.json",
        dev_cohort=root / "development.json",
        split=split,
        adr0016=adr,
        protocol=protocol,
        config=cfg,
        review=root / "review.json",
        encoder=None if mode == "direct_end_to_end" else ancestor,
        ancestor_review=None if mode == "direct_end_to_end" else parent_review,
    )
    bindings = {str(p.resolve()): ds.sha256(p) for p in ds.required_paths(inputs, core_config)}
    inputs.review.write_text(
        json.dumps(
            {
                "status": "APPROVED_DOWNSTREAM_PREFIT",
                "reviewer_session_id": "synthetic-downstream-reviewer",
                "implementer_session_id": ds.IMPLEMENTER_SESSION_ID,
                "allowed_modes": [mode],
                "allowed_methods": [method],
                "bindings": bindings,
                "correctness_core_config": core_config.to_dict(),
                "train_npz_sha256": ds.sha256(inputs.train),
                "dev_npz_sha256": ds.sha256(inputs.dev),
                "split_sha256": ds.sha256(split),
            }
        )
    )
    return config, core_config, inputs, model


def test_reserved_final_test_role_matches_actual_native_split(memory):
    root, _ = memory
    _, _, inputs, _ = fixture_run(root)
    split = json.loads(inputs.split.read_text())
    split["sources"].append(
        {
            "role": "final_test",
            "deployment": "reserved-final-site",
            "archive_sha256": "reserved-final-archive",
        }
    )
    inputs.split.write_text(json.dumps(split))
    roles = ds._split_roles(inputs)
    assert roles["final_test"] == {"reserved-final-site": "reserved-final-archive"}


def test_full_parent_review_can_contain_more_bindings_than_minimal_run_record(memory):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    review = json.loads(inputs.ancestor_review.read_text())
    additional = MAIN / "src/marine_echo/__init__.py"
    review["bindings"][str(additional.resolve())] = ds.sha256(additional)
    inputs.ancestor_review.write_text(json.dumps(review))
    report_path = root / "parent/run.json"
    parent = json.loads(report_path.read_text())
    parent["review_sha256"] = ds.sha256(inputs.ancestor_review)
    report_path.write_text(json.dumps(parent))
    downstream = json.loads(inputs.review.read_text())
    for path in (inputs.ancestor_review, report_path):
        downstream["bindings"][str(path.resolve())] = ds.sha256(path)
    inputs.review.write_text(json.dumps(downstream))
    ds.check_prefit(inputs, config, core_config, correctness_smoke=True)


@pytest.mark.parametrize("relative", ["data/native_ssl_corpus.py", "training/aeon_corpus.py"])
def test_changed_transitive_scaler_source_binding_is_rejected(memory, relative):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    source = MAIN / "src/marine_echo" / relative
    review = json.loads(inputs.review.read_text())
    review["bindings"][str(source.resolve())] = "0" * 64
    inputs.review.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="binding"):
        ds.check_prefit(inputs, config, core_config, correctness_smoke=True)


@pytest.mark.parametrize("mode", ["frozen_readout", "full_finetune", "direct_end_to_end"])
def test_updates_freezing_full_resume_and_target_free_replay(memory, mode):
    root, _ = memory
    config, core_config, inputs, initial = fixture_run(root, mode=mode)
    before = ds.sha256(root / "parent/selected_encoder.pt")
    interrupted = root / "interrupted"
    ds.run(
        inputs,
        config,
        output=interrupted,
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        stop_after=1,
    )
    resumed, continuous = root / "resumed", root / "continuous"
    ds.run(
        inputs,
        config,
        output=resumed,
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        resume=interrupted / "latest.pt",
    )
    report = ds.run(
        inputs,
        config,
        output=continuous,
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
    )
    a, b = [torch.load(p / "inference.pt", weights_only=True) for p in (resumed, continuous)]
    for k, v in a["model"].items():
        assert torch.equal(v, b["model"][k]), k
    changed = any(
        not torch.equal(v, b["model"]["encoder." + k])
        for k, v in initial.encoder.state_dict().items()
    )
    assert changed == (mode != "frozen_readout")
    assert ds.sha256(root / "parent/selected_encoder.pt") == before
    if mode == "frozen_readout":
        assert ds.sha256(continuous / "selected_encoder.pt") == before
    assert report["supervised_updates"] == 3 and report["mode"] == mode
    assert report["evidence_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    assert report["sample_presentations"] == 12
    assert report["label_observations_processed"] == 36
    assert report["resources_additional_downstream"]["elapsed_gpu_seconds"] == 0
    assert report["resources_additional_downstream"]["elapsed_seconds"] > 0
    assert report["parameters_optimized"] == report["parameters_head"] + (
        report["parameters_encoder"] if mode != "frozen_readout" else 0
    )
    x = arrays("development")
    past = {k: x[k] for k in ("x", "observed", "metadata", "query")}
    inference = ds.load_inference(continuous / "inference.pt")
    predicted = inference.forecast(**past)
    with np.load(continuous / "predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(predicted, saved["predictions"])
        np.testing.assert_array_equal(
            ds.core.predict_from_checkpoint(continuous / "inference.pt", past), predicted
        )
        np.testing.assert_array_equal(saved["query"], x["query"])
    assert inference.encode(past["x"], past["observed"], past["metadata"]).shape == (24, 8)
    checkpoint = torch.load(continuous / "latest.pt", weights_only=True)
    assert checkpoint["state"]["step"] == 3
    assert len(checkpoint["state"]["sequence"]) == 3
    for step, entry in enumerate(checkpoint["state"]["sequence"]):
        np.testing.assert_array_equal(
            entry["indices"], ds.core.batch_indices(np.arange(24), 4, 7, "readout", step)
        )
    latest = torch.load(resumed / "latest.pt", weights_only=True)
    assert latest["state"]["sequence"] == checkpoint["state"]["sequence"]
    assert latest["scheduler"] == checkpoint["scheduler"]
    assert latest["optimizer"]["param_groups"] == checkpoint["optimizer"]["param_groups"]
    for key, item in latest["optimizer"]["state"].items():
        for name, value in item.items():
            assert torch.equal(value, checkpoint["optimizer"]["state"][key][name])
    assert torch.equal(latest["rng"]["torch"], checkpoint["rng"]["torch"])
    assert latest["rng"]["python"] == checkpoint["rng"]["python"]
    assert torch.equal(latest["rng"]["numpy"][1], checkpoint["rng"]["numpy"][1])


def test_stale_bindings_protocol_ancestor_and_self_review_stop_before_numeric_loading(
    memory, monkeypatch
):
    root, files = memory
    config, core_config, inputs, _ = fixture_run(root)
    monkeypatch.setattr(
        ds.core,
        "load_corpus",
        lambda *a, **k: pytest.fail("Rejected lineage reached numeric loading."),
    )
    review = json.loads(inputs.review.read_text())
    for path in ds.required_paths(inputs, core_config):
        original = files.get(str(path))
        changed = {
            **review,
            "bindings": {k: v for k, v in review["bindings"].items() if k != str(path)},
        }
        inputs.review.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="binding"):
            ds.run(
                inputs,
                config,
                output=root / "rejected",
                device="cpu",
                correctness_smoke=True,
                smoke_core_config=core_config,
            )
        if original is not None:
            files[str(path)] = original
    inputs.review.write_text(
        json.dumps({**review, "reviewer_session_id": ds.IMPLEMENTER_SESSION_ID})
    )
    with pytest.raises(ValueError, match="distinct"):
        ds.run(
            inputs,
            config,
            output=root / "rejected",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
        )


def test_head_initialization_and_supervised_batches_are_independent_of_ancestor_rng(memory):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    a = ds.prepare_model(config, core_config, torch.load(inputs.encoder, weights_only=True))
    torch.manual_seed(999)
    np.random.seed(999)
    direct = ds.DownstreamConfig(
        mode="direct_end_to_end", method="direct", updates=3, cadence=1, batch_size=4
    )
    b = ds.prepare_model(direct, core_config, None)
    assert all(torch.equal(v, b.readout.state_dict()[k]) for k, v in a.readout.state_dict().items())
    assert np.array_equal(
        ds.supervised_indices(np.arange(24), config, 1),
        ds.supervised_indices(np.arange(24), direct, 1),
    )


def test_real_budget_and_modes_are_immutable():
    from dataclasses import FrozenInstanceError

    frozen = ds.DownstreamConfig()
    assert frozen.updates == 2000 and frozen.cadence == 500
    with pytest.raises(FrozenInstanceError):
        frozen.updates = 7
    ds.DownstreamConfig(mode="full_finetune", updates=3000, cadence=750).validate()
    with pytest.raises(ValueError):
        ds.DownstreamConfig(updates=5001).validate()


def refresh_review(inputs, core_config):
    """Fixture signing only; never an independent scientific approval."""
    review = json.loads(inputs.review.read_text())
    review["correctness_core_config"] = core_config.to_dict()
    review["bindings"] = {str(p): ds.sha256(p) for p in ds.required_paths(inputs, core_config)}
    inputs.review.write_text(json.dumps(review))


@pytest.mark.parametrize(
    "change",
    ["protocol", "config", "ancestor", "status", "mode", "method", "cohort_role", "runtime_config"],
)
def test_hash_role_and_runtime_gates_reject_before_numeric_loading(memory, monkeypatch, change):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    review = json.loads(inputs.review.read_text())
    if change in ("protocol", "config"):
        getattr(inputs, change).write_text("tampered")
    elif change == "ancestor":
        with inputs.encoder.open("wb") as stream:
            stream.write(b"tampered")
    elif change == "cohort_role":
        cohort = json.loads(inputs.train_cohort.read_text())
        inputs.train_cohort.write_text(json.dumps({**cohort, "role": "test"}))
        refresh_review(inputs, core_config)
    elif change == "runtime_config":
        config = replace(config, seed=13)
    else:
        key, value = {
            "status": ("status", "REJECTED"),
            "mode": ("allowed_modes", []),
            "method": ("allowed_methods", []),
        }[change]
        inputs.review.write_text(json.dumps({**review, key: value}))
    monkeypatch.setattr(
        ds.core, "load_corpus", lambda *a, **k: pytest.fail("Gate reached numerical payload.")
    )
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Gate reached weights payload."))
    with pytest.raises(ValueError):
        ds.run(
            inputs,
            config,
            output=root / "rejected",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
        )


@pytest.mark.parametrize(
    "change",
    [
        "deployment",
        "rows",
        "scalers",
        "features",
        "parent_identity",
        "parent_config",
        "parent_reviewer",
    ],
)
def test_ancestor_internal_lineage_rejected_even_with_fresh_downstream_bindings(
    memory, monkeypatch, change
):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    parent_path = inputs.encoder.parent / "run.json"
    parent = json.loads(parent_path.read_text())
    membership_path = inputs.encoder.parent / "membership.json"
    membership = json.loads(membership_path.read_text())
    if change in ("deployment", "rows"):
        key = "train_deployments" if change == "deployment" else "train_row_ids"
        membership[key][0] = (
            "synthetic-development" if change == "deployment" else "synthetic-development-0"
        )
        membership_path.write_text(json.dumps(membership))
        parent["membership_sha256"] = ds.sha256(membership_path)
    elif change in ("scalers", "features"):
        selected = torch.load(inputs.encoder, weights_only=True)
        inference_path = inputs.encoder.parent / "inference.pt"
        inference = torch.load(inference_path, weights_only=True)
        if change == "scalers":
            selected["scalers"]["channel_mean"][0] += 9
            inference["scalers"] = selected["scalers"]
            torch.save(inference, inference_path)
            parent["inference_sha256"] = ds.sha256(inference_path)
        else:
            selected["encoder"][next(iter(selected["encoder"]))] += 1
        torch.save(selected, inputs.encoder)
    elif change == "parent_config":
        parent["config"]["history"] = 24
    elif change == "parent_reviewer":
        review = json.loads(inputs.ancestor_review.read_text())
        review["reviewer_session_id"] = review["implementer_session_id"]
        inputs.ancestor_review.write_text(json.dumps(review))
        parent["review_sha256"] = ds.sha256(inputs.ancestor_review)
    else:
        parent["inference_sha256"] = "0" * 64
    parent_path.write_text(json.dumps(parent))
    refresh_review(inputs, core_config)
    monkeypatch.setattr(
        torch.optim, "AdamW", lambda *a, **k: pytest.fail("Invalid lineage reached an optimizer.")
    )
    with pytest.raises(ValueError):
        ds.run(
            inputs,
            config,
            output=root / "rejected",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
        )


@pytest.mark.parametrize("method", ["shared_ssl", "cf_jepa"])
@pytest.mark.parametrize("mode", ["frozen_readout", "full_finetune"])
def test_gradient_routes_and_cf_ema_buffer_semantics(memory, method, mode):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root, mode=mode, method=method)
    selected = torch.load(inputs.encoder, weights_only=True)
    model = ds.prepare_model(config, core_config, selected)
    if method == "cf_jepa":
        # Distinguish the selected EMA weights from the newly initialized online
        # branch, so accidentally forecasting online features is observable.
        with torch.no_grad():
            next(model.online.parameters()).add_(5)
        online = ds.core.cpu_state(model.online)
    before = ds.core.cpu_state(model.encoder)
    data = arrays("train")
    batch = ds.core.tensor_batch(
        data, np.arange(4), ds.core.Scalers.from_dict(selected["scalers"]), 96, "cpu"
    )
    predicted = ds._forecast_train(model, batch, mode)
    ds.core.pinball(predicted, batch["y"], batch["y_observed"]).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.readout.parameters())
    grads = [p.grad for p in model.encoder.parameters()]
    if mode == "frozen_readout":
        assert all(g is None for g in grads)
        assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())
        assert not model.encoder.training
    else:
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        assert model.encoder.training
    if method == "cf_jepa":
        assert any("running_mean" in k for k in before)
        assert all(torch.equal(v, model.online.state_dict()[k]) for k, v in online.items())
        assert all(p.grad is None for p in model.online.parameters())


def test_independent_daily_pinball_reconstruction_masks_and_equal_weighting(memory):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    data = arrays("development")
    data["split_sha256"] = np.asarray(ds.sha256(inputs.split))
    data["y_observed"][:4, 1] = False
    data["y"][:4, 1] = 0
    # Preserve20 valid anchors for horizon1; the masked values cannot score.
    np.savez(inputs.dev, **data)
    # A direct trajectory lets this isolated test change development without
    # constructing an approved pretrained ancestor for a different cohort.
    config = replace(config, mode="direct_end_to_end", method="direct")
    core_config = replace(core_config, method="direct")
    inputs = replace(inputs, encoder=None, ancestor_review=None)
    inputs.config.write_text(json.dumps(config.to_dict()))
    cohort = json.loads(inputs.dev_cohort.read_text())
    inputs.dev_cohort.write_text(json.dumps({**cohort, "npz_sha256": ds.sha256(inputs.dev)}))
    review = json.loads(inputs.review.read_text())
    review.update(
        allowed_modes=[config.mode],
        allowed_methods=[config.method],
        dev_npz_sha256=ds.sha256(inputs.dev),
    )
    inputs.review.write_text(json.dumps(review))
    refresh_review(inputs, core_config)
    ds.run(
        inputs,
        config,
        output=root / "scored",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
    )
    metrics = json.loads((root / "scored/metrics.json").read_text())
    with np.load(root / "scored/predictions.npz", allow_pickle=False) as saved:
        residual = saved["targets"].astype(float)[..., None] - saved["predictions"].astype(float)
        q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        losses = np.maximum(residual * q, residual * (q - 1)).mean(-1)
        manual = np.mean([losses[saved["target_observed"][:, h], h].mean() for h in range(3)])
        assert metrics["primary_pinball"] == pytest.approx(manual, abs=1e-12)
        assert metrics["coverage"]["scored_rows"] == [24, 20, 24]
        np.testing.assert_array_equal(
            saved["query_native_bounds_m"], data["query"][:, :, 3:5] * 250
        )
        assert np.all(np.diff(saved["predictions"], axis=-1) >= 0)
        # Changing only a masked assessment value cannot change reconstruction.
        perturbed = saved["targets"].copy()
        perturbed[~saved["target_observed"]] = 1e9
        again = ds.core.daily_metrics(
            saved["predictions"],
            perturbed,
            saved["target_observed"],
            saved["target_dates"],
            saved["deployment"],
        )
        assert again == metrics


def test_explicit_resume_identity_and_output_protection(memory, monkeypatch):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    interrupted = root / "interrupted"
    ds.run(
        inputs,
        config,
        output=interrupted,
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        stop_after=1,
    )
    checkpoint = torch.load(interrupted / "latest.pt", weights_only=True)
    checkpoint["identities"]["device_kind"] = "cuda"
    torch.save(checkpoint, root / "bad-resume.pt")
    monkeypatch.setattr(
        ds.core,
        "load_corpus",
        lambda *a, **k: pytest.fail("Invalid continuation reached payloads."),
    )
    with pytest.raises(ValueError, match="identities"):
        ds.run(
            inputs,
            config,
            output=root / "bad",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
            resume=root / "bad-resume.pt",
        )
    with pytest.raises(ValueError, match="output"):
        ds.run(
            inputs,
            config,
            output=interrupted,
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
        )


@pytest.mark.parametrize("change", ["sequence", "scheduler", "best_encoder"])
def test_resume_counters_and_frozen_selected_state_rejected_before_optimizer(
    memory, monkeypatch, change
):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    ds.run(
        inputs,
        config,
        output=root / "interrupted",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        stop_after=1,
    )
    payload = torch.load(root / "interrupted/latest.pt", weights_only=True)
    if change == "sequence":
        payload["state"]["sequence"][0]["indices"][0] = 999
    elif change == "scheduler":
        payload["scheduler"]["last_epoch"] += 1
    else:
        key = next(k for k in payload["state"]["best_model"] if k.startswith("encoder."))
        payload["state"]["best_model"][key] += 1
    torch.save(payload, root / "bad-resume.pt")
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *a, **k: pytest.fail("Invalid continuation reached an optimizer."),
    )
    with pytest.raises(ValueError, match="Resume|resume"):
        ds.run(
            inputs,
            config,
            output=root / "resumed",
            device="cpu",
            correctness_smoke=True,
            smoke_core_config=core_config,
            resume=root / "bad-resume.pt",
        )


def test_scheduled_selection_ties_keep_earliest_and_patience_counts_checks(memory, monkeypatch):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    config = replace(config, updates=10)
    inputs.config.write_text(json.dumps(config.to_dict()))
    refresh_review(inputs, core_config)
    metrics = ds.core.daily_metrics

    def tied(*args, **kwargs):
        result = metrics(*args, **kwargs)
        result["primary_pinball"] = 1.0
        return result

    monkeypatch.setattr(ds.core, "daily_metrics", tied)
    report = ds.run(
        inputs,
        config,
        output=root / "stopped",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
    )
    assert report["supervised_updates"] == 5
    assert report["selected_supervised_step"] == 1
    state = torch.load(root / "stopped/latest.pt", weights_only=True)["state"]
    assert state["stopped"] and state["bad_checks"] == 4
    assert [r["selected"] for r in state["records"]] == [True, False, False, False, False]


def test_daily_equal_weighting_over_dates_horizons_and_deployments():
    # Unequal row counts/deployment-day counts expose an accidental row mean.
    counts = [18, 24, 30]
    values = [1.0, 3.0, 9.0]
    targets = np.repeat(values, counts)[:, None] * np.array([[1.0, 2.0, 4.0]])
    prediction = np.zeros((*targets.shape, 5))
    mask = np.ones(targets.shape, bool)
    dates = np.repeat(["2001-01-01", "2001-01-02", "2001-01-01"], counts)[:, None].repeat(3, 1)
    deployments = np.repeat(["a", "a", "b"], counts)
    score = ds.core.daily_metrics(prediction, targets, mask, dates, deployments)
    # Symmetric quantile mean .5; two equally weighted dates for deployment a,
    # one date for b, then three horizons, then equally weighted deployments.
    expected = np.mean(
        [
            np.mean([0.5 * (1 + 3) / 2 * h for h in (1, 2, 4)]),
            np.mean([0.5 * 9 * h for h in (1, 2, 4)]),
        ]
    )
    assert score["primary_pinball"] == pytest.approx(expected, abs=1e-12)
    assert score["primary_pinball"] != pytest.approx(targets.mean() * 0.5)
    mask[deployments == "b", 0] = False
    assert (
        ds.core.daily_metrics(prediction, targets, mask, dates, deployments)["primary_pinball"]
        is None
    )


def test_test_role_npz_cannot_reach_numerical_members(memory, monkeypatch):
    root, _ = memory
    data = arrays("train")
    data["corpus_role"] = np.asarray("test")
    data["split_sha256"] = np.asarray("fixture")
    np.savez(root / "test-role.npz", **data)
    load = np.load
    reads = []

    class MetadataOnly:
        def __init__(self, archive):
            self.archive = archive

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            reads.append(key)
            assert key == "corpus_role", "A rejected test role reached numerical members."
            return self.archive[key]

    monkeypatch.setattr(np, "load", lambda *a, **k: MetadataOnly(load(*a, **k)))
    with pytest.raises(ValueError, match="test-role"):
        ds.core.load_corpus(
            root / "test-role.npz", role="train", split_hash="fixture", correctness_smoke=True
        )
    assert reads == ["corpus_role"]


@pytest.mark.parametrize("seed", [7, 13, 23])
def test_all_native_controls_match_initialization_and_sample_sequence(seed):
    core_config = ds.core.Config(width=24, latent=8, blocks=1, heads=4)
    reference = ds.prepare_model(
        ds.DownstreamConfig(mode="direct_end_to_end", method="direct", seed=seed), core_config, None
    )
    state = ds.core.cpu_state(reference)
    pool = np.arange(0, 48, 2)
    for method in ("shared_ssl", "masked_ssl", "random_frozen", "permuted_ssl", "direct"):
        config = ds.DownstreamConfig(method=method, seed=seed)
        model = ds.prepare_model(config, core_config, None)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
        for step in (0, 1, 17):
            np.testing.assert_array_equal(
                ds.supervised_indices(pool, config, step),
                ds.supervised_indices(
                    pool, replace(config, method="direct", mode="direct_end_to_end"), step
                ),
            )


def test_frozen_cf_full_runner_resume_and_context_only_replay(memory):
    root, _ = memory
    config, core_config, inputs, initial = fixture_run(root, method="cf_jepa")
    ds.run(
        inputs,
        config,
        output=root / "interrupted",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        stop_after=1,
    )
    ds.run(
        inputs,
        config,
        output=root / "resumed",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
        resume=root / "interrupted/latest.pt",
    )
    ds.run(
        inputs,
        config,
        output=root / "continuous",
        device="cpu",
        correctness_smoke=True,
        smoke_core_config=core_config,
    )
    resumed = torch.load(root / "resumed/inference.pt", weights_only=True)
    continuous = torch.load(root / "continuous/inference.pt", weights_only=True)
    assert all(torch.equal(v, continuous["model"][k]) for k, v in resumed["model"].items())
    assert all(
        torch.equal(v, continuous["model"]["encoder." + k])
        for k, v in initial.encoder.state_dict().items()
    )
    assert ds.sha256(inputs.encoder) == ds.sha256(root / "resumed/selected_encoder.pt")
    past = {k: arrays("development")[k] for k in ("x", "observed", "metadata", "query")}
    with np.load(root / "resumed/predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            ds.load_inference(root / "resumed/inference.pt").forecast(**past), saved["predictions"]
        )


def test_cli_accepts_explicit_cuda_index_without_executing_fit(tmp_path, monkeypatch):
    from marine_echo.training import native_downstream as module

    config = tmp_path / "cli-config.json"
    config.write_text(json.dumps(module.DownstreamConfig().to_dict()))
    seen = []

    def inspect_only(inputs, cfg, **kwargs):
        seen.append(kwargs["device"])
        return {
            "status": "SYNTHETIC_CORRECTNESS_ONLY",
            "mode": cfg.mode,
            "supervised_updates": 0,
            "inference_sha256": "NOT_RUN",
        }

    monkeypatch.setattr(module, "run", inspect_only)
    argv = []
    for flag in (
        "train",
        "dev",
        "train-cohort",
        "dev-cohort",
        "split",
        "adr0016",
        "protocol",
        "review",
        "output",
    ):
        argv += ["--" + flag, str(tmp_path / flag)]
    argv += ["--config", str(config), "--device", "cuda:0"]
    module.main(argv)
    assert seen == ["cuda:0"]


def test_reviewed_historical_shared_model_snapshot_keeps_original_ancestor_bindings(memory):
    root, _ = memory
    config, core_config, inputs, _ = fixture_run(root)
    model_path = Path(ds.core.native_temporal.__file__).resolve()
    snapshot = root / "historical-model.py"
    snapshot.write_text(
        model_path.read_text() + "\n# SYNTHETIC_CORRECTNESS_ONLY provenance fixture\n"
    )
    old_hash = ds.sha256(snapshot)
    parent_path = inputs.encoder.parent / "run.json"
    parent = json.loads(parent_path.read_text())
    parent_review = json.loads(inputs.ancestor_review.read_text())
    parent["bindings"][str(model_path)] = old_hash
    parent_review["bindings"][str(model_path)] = old_hash
    inputs.ancestor_review.write_text(json.dumps(parent_review))
    parent["review_sha256"] = ds.sha256(inputs.ancestor_review)
    parent_path.write_text(json.dumps(parent))
    reviewed = json.loads(inputs.review.read_text())
    reviewed["compatible_ancestor_model_source"] = {
        "path": str(snapshot.resolve()),
        "sha256": old_hash,
        "methods": ["shared_ssl"],
    }
    inputs.review.write_text(json.dumps(reviewed))
    refresh_review(inputs, core_config)
    ds.check_prefit(inputs, config, core_config, correctness_smoke=True)
    # Snapshot bytes are part of this distinct prefit review, never silently
    # substituted for historical corpora/scalers or allowed to change afterward.
    snapshot.write_text("tampered")
    with pytest.raises(ValueError, match="binding"):
        ds.check_prefit(inputs, config, core_config, correctness_smoke=True)
