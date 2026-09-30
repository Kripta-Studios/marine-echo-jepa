"""SYNTHETIC_CORRECTNESS_ONLY CPU fixtures; immutable MAIN helpers are imported.

Only package search paths are extended to import the NEW candidate modules.
No original factory, function or project source is patched by this harness.
"""

import builtins
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Bound synthetic CPU checks; runtime inference itself never changes threading.
torch.set_num_threads(1)

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN / "src"))
import marine_echo
import marine_echo.inference
import marine_echo.models
import marine_echo.training

for package, folder in (
    (marine_echo.models, "models"),
    (marine_echo.training, "training"),
    (marine_echo.inference, "inference"),
):
    candidate = str(BUILDER / "src/marine_echo" / folder)
    if candidate not in package.__path__:
        package.__path__.append(candidate)


def synthetic_inputs(n=3, history=96):
    rng = np.random.default_rng(71)
    x = rng.normal(-78, 3, (n, history, 4)).astype(np.float32)
    observed = np.ones_like(x, dtype=bool)
    observed[:, ::3, 1:] = False
    metadata = np.zeros((n, 4, 10), np.float32)
    metadata[..., 0] = np.array([38000, 125000, 200000, 455000]) / 455000
    metadata[..., 1:3] = 1
    metadata[..., 4] = 230 / 250
    query = np.repeat(metadata[:, :1], 3, axis=1)
    query[..., 9] = [1, 3, 6]
    return x, observed, metadata, query


def codec(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    return stream


def scalers():
    return {
        "channel_mean": [-78.0] * 4,
        "channel_std": [3.0] * 4,
        "target_mean": [-76.0] * 3,
        "target_std": [2.0] * 3,
    }


def provenance():
    # Frozen source ancestry is opaque provenance and must never be executed.
    return {"UNEXECUTABLE_SYNTHETIC_CORRECTNESS_ONLY_SOURCE": "a" * 64}


def artifact(method="shared_ssl", selected=False, batch_size=2, seed=13):
    from marine_echo.training import native_band_replication_ssl as band

    config = band.Config(
        method=method,
        seed=seed,
        width=16,
        latent=8,
        blocks=2,
        heads=4,
        pretrain_updates=2,
        readout_updates=2,
        pretrain_cadence=2,
        readout_cadence=1,
        batch_size=batch_size,
    )
    model = band.initialize_model(seed, **band.model_dimensions(config), method=method)
    return {
        "kind": "native_band_replication_ssl_selected_encoder_v2"
        if selected
        else "native_band_replication_ssl_weights_only_inference_v2",
        "architecture": band.ARCHITECTURE,
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "correctness_smoke": True,
        "config": config.to_dict(),
        "scalers": scalers(),
        "bindings": provenance(),
        "selected_pretrain_step": 2,
        "encoder" if selected else "model": band.cpu_state(model.encoder if selected else model),
    }, model


class MemoryFiles:
    """Real codecs with explicit virtual transport; no original factories patched."""

    def __init__(self, monkeypatch):
        self.base = EVIDENCE / "SYNTHETIC_CORRECTNESS_ONLY_VIRTUAL"
        self.files, self.directories = {}, {self.base}
        original_path_open, original_exists = Path.open, Path.exists
        original_iterdir, original_mkdir, original_builtin = Path.iterdir, Path.mkdir, builtins.open
        original_savez_compressed = np.savez_compressed

        def virtual(path):
            return Path(path).is_relative_to(self.base)

        def opened(path, mode="r", *args, **kwargs):
            path = Path(path)
            if not virtual(path):
                return original_path_open(path, mode, *args, **kwargs)
            if "x" in mode and path in self.files:
                raise FileExistsError(path)
            if "r" in mode:
                if path not in self.files:
                    raise FileNotFoundError(path)
                return (
                    io.BytesIO(self.files[path])
                    if "b" in mode
                    else io.StringIO(self.files[path].decode("utf-8"))
                )
            binary = "b" in mode
            base = io.BytesIO if binary else io.StringIO

            class Output(base):
                def close(stream):
                    if not stream.closed:
                        payload = stream.getvalue()
                        self.files[path] = payload if binary else payload.encode("utf-8")
                        self.directories.add(path.parent)
                    super().close()

            return Output()

        def builtin(path, mode="r", *args, **kwargs):
            if isinstance(path, (str, Path)) and virtual(path):
                return opened(path, mode, *args, **kwargs)
            return original_builtin(path, mode, *args, **kwargs)

        def iterdir(path):
            if not virtual(path):
                return original_iterdir(path)
            if path not in self.directories:
                raise FileNotFoundError(path)
            return iter(
                sorted({p for p in self.files.keys() | self.directories if p.parent == path})
            )

        def mkdir(path, *args, **kwargs):
            if not virtual(path):
                return original_mkdir(path, *args, **kwargs)
            self.directories.add(path)

        def savez_compressed(path, *args, **kwargs):
            # NumPy's ZipFile calls io.open, which is outside this explicit
            # virtual transport. Use its real codec on a virtual file handle.
            # This is synthetic test I/O only, never a scientific fitter or
            # a retry of an operating-system permission denial.
            if isinstance(path, (str, Path)) and virtual(path):
                with opened(path, "wb") as stream:
                    return original_savez_compressed(stream, *args, **kwargs)
            return original_savez_compressed(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", opened)
        monkeypatch.setattr(
            Path,
            "exists",
            lambda p: (
                p in self.files or p in self.directories if virtual(p) else original_exists(p)
            ),
        )
        monkeypatch.setattr(Path, "iterdir", iterdir)
        monkeypatch.setattr(Path, "mkdir", mkdir)
        monkeypatch.setattr(builtins, "open", builtin)
        monkeypatch.setattr(np, "savez_compressed", savez_compressed)

    def json(self, path, value):
        self.files[Path(path)] = json.dumps(value, sort_keys=True).encode()
        self.directories.add(Path(path).parent)

    def npz(self, path, data):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        self.files[Path(path)] = buffer.getvalue()
        self.directories.add(Path(path).parent)

    def checkpoint(self, path, data):
        self.files[Path(path)] = codec(data).getvalue()
        self.directories.add(Path(path).parent)


def synthetic_corpus(role, split_hash, n=20):
    x, mask, meta, query = synthetic_inputs(n=n)
    cutoff = np.arange(1000, 1000 + n)
    rng = np.random.default_rng(81 if role == "train" else 91)
    return {
        "corpus_role": np.array(role),
        "split_sha256": np.array(split_hash),
        "x": x,
        "observed": mask,
        "metadata": meta,
        "query": query,
        "future": rng.normal(-77, 3, (n, 3, 4, 4)).astype(np.float32),
        "future_observed": np.ones((n, 3, 4, 4), bool),
        "y": rng.normal(-76, 2, (n, 3)).astype(np.float32),
        "y_observed": np.ones((n, 3), bool),
        "ssl_eligible": np.ones(n, bool),
        "row_id": np.array([f"synthetic-{role}-{i}" for i in range(n)]),
        "deployment": np.repeat(f"synthetic-{role}", n),
        "archive_sha256": np.repeat("a" * 64 if role == "train" else "b" * 64, n),
        "cutoff": cutoff,
        "context_ids": cutoff[:, None] - np.arange(95, -1, -1),
        "future_ids": cutoff[:, None, None] + np.array([1, 3, 6])[None, :, None] + np.arange(4),
        "target_dates": np.tile(["2024-01-02", "2024-01-03", "2024-01-04"], (n, 1)),
    }


def gate_fixture(monkeypatch, *, method="shared_ssl", mode="frozen_readout", seed=13):
    from marine_echo.training import native_band_replication_downstream as downstream
    from marine_echo.training import native_band_replication_ssl as core

    fs = MemoryFiles(monkeypatch)
    p = fs.base
    paths = {
        key: p / f"{key}.json"
        for key in (
            "split",
            "protocol",
            "config",
            "review",
            "train_cohort",
            "dev_cohort",
            "downstream_config",
            "downstream_review",
        )
    }
    paths.update(train=p / "train.npz", dev=p / "dev.npz")
    fs.json(
        paths["split"],
        {
            "sources": [
                {"deployment": f"synthetic-{role}", "role": role, "archive_sha256": value * 64}
                for role, value in (("train", "a"), ("development", "b"), ("final_test", "c"))
            ]
        },
    )
    split_hash = core.sha256(paths["split"])
    train, dev = (synthetic_corpus(role, split_hash) for role in ("train", "development"))
    fs.npz(paths["train"], train)
    fs.npz(paths["dev"], dev)
    fs.json(
        paths["protocol"],
        {"evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY", "architecture": core.ARCHITECTURE},
    )
    cfg = core.Config(
        method=method,
        seed=seed,
        width=16,
        latent=8,
        blocks=2,
        heads=4,
        batch_size=4,
        pretrain_updates=2,
        readout_updates=2,
        pretrain_cadence=2,
        readout_cadence=1,
    )
    fs.json(paths["config"], cfg.to_dict())
    core_sources = core.required_sources(cfg)
    bound = [paths[k] for k in ("train", "dev", "split", "protocol", "config")] + core_sources
    review = {
        "status": "APPROVED_PREFIT",
        "architecture": core.ARCHITECTURE,
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
        "allowed_roles": ["train", "development"],
        "allowed_methods": [method],
        "allowed_seeds": [seed],
        "implementer_session_id": core.IMPLEMENTER_SESSION_ID,
        "root_coordinator_session_id": "synthetic-coordinator",
        "reviewer_session_id": "synthetic-distinct-reviewer",
        "bindings": {str(q.resolve()): core.sha256(q) for q in bound},
        "train_npz_sha256": core.sha256(paths["train"]),
        "dev_npz_sha256": core.sha256(paths["dev"]),
        "split_sha256": split_hash,
        "protocol_sha256": core.sha256(paths["protocol"]),
    }
    fs.json(paths["review"], review)
    for role, key, payload in (
        ("train", "train_cohort", "train"),
        ("development", "dev_cohort", "dev"),
    ):
        fs.json(
            paths[key],
            {
                "role": role,
                "npz_sha256": core.sha256(paths[payload]),
                "split_sha256": split_hash,
                "issued": 20,
                "source_reports": [{"deployment": f"synthetic-{role}", "issued": 20}],
            },
        )
    fitted = core.Scalers.fit(train, split_path=paths["split"])
    parent = p / "ancestor"
    encoder_path = None if mode == "direct_end_to_end" else parent / "selected_encoder.pt"
    if encoder_path:
        model = core.initialize_model(seed, **core.model_dimensions(cfg), method=method)
        bindings = review["bindings"]
        common = {
            "architecture": core.ARCHITECTURE,
            "config": cfg.to_dict(),
            "scalers": fitted.to_dict(),
            "bindings": bindings,
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
            "correctness_smoke": True,
        }
        fs.checkpoint(
            encoder_path,
            {
                **common,
                "kind": "native_band_replication_ssl_selected_encoder_v2",
                "encoder": core.cpu_state(model.encoder),
            },
        )
        fs.checkpoint(
            parent / "inference.pt",
            {
                **common,
                "kind": "native_band_replication_ssl_weights_only_inference_v2",
                "model": core.cpu_state(model),
                "selected_pretrain_step": 2,
            },
        )
        membership = {
            "train_row_ids": train["row_id"].tolist(),
            "train_deployments": train["deployment"].tolist(),
            "train_archive_sha256": train["archive_sha256"].tolist(),
            "ssl_eligible_indices": list(range(20)),
            "supervised_indices": list(range(20)),
            "sequence": [],
        }
        fs.json(parent / "membership.json", membership)
        # Core config binding remains the exact original reviewed virtual path.
        fs.json(
            parent / "run.json",
            {
                "architecture": core.ARCHITECTURE,
                "status": "COMPLETED",
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                "config": cfg.to_dict(),
                "bindings": bindings,
                "selected_pretrain_step": 2,
                "inference_sha256": core.sha256(parent / "inference.pt"),
                "membership_sha256": core.sha256(parent / "membership.json"),
                "review_sha256": core.sha256(paths["review"]),
            },
        )
    dcfg = downstream.DownstreamConfig(
        seed=seed, mode=mode, method=method, updates=4, cadence=2, batch_size=4
    )
    fs.json(paths["downstream_config"], dcfg.to_dict())
    inputs = downstream.RunInputs(
        train=paths["train"],
        dev=paths["dev"],
        train_cohort=paths["train_cohort"],
        dev_cohort=paths["dev_cohort"],
        split=paths["split"],
        adr0016=paths["protocol"],
        protocol=paths["protocol"],
        config=paths["downstream_config"],
        review=paths["downstream_review"],
        encoder=encoder_path,
        ancestor_review=paths["review"] if encoder_path else None,
        ancestor_config=paths["config"] if encoder_path else None,
    )
    dreview = {
        **review,
        "status": "APPROVED_DOWNSTREAM_PREFIT",
        "allowed_modes": [mode],
        "correctness_core_config": cfg.to_dict(),
    }
    fs.json(paths["downstream_review"], dreview)
    dreview["bindings"] = {str(q): core.sha256(q) for q in downstream.required_paths(inputs, cfg)}
    fs.json(paths["downstream_review"], dreview)
    return fs, paths, cfg, review, inputs, dcfg, dreview, train, dev
