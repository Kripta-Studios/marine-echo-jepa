"""Hash-gated, resumable native acoustic SSL and matched frozen-readout runner.

Real fitting is owner-executed after distinct APPROVED_PREFIT review. The explicit
correctness-smoke path admits only visibly synthetic rows on CPU, with synthetic
approval fixtures; it is never scientific evidence. Final test is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

from marine_echo.inference import native_encoder as legacy_encoder
from marine_echo.models import native_band_temporal as native_temporal
from marine_echo.models import native_temporal as legacy_temporal
from marine_echo.models import sigreg
from marine_echo.models.native_band_temporal import (
    ARCHITECTURE,
    QUANTILES,
    QueryHead,
)
from marine_echo.models.native_band_temporal import (
    NativeBandTemporalModel as NativeTemporalModel,
)
from marine_echo.training import native_desktop_resources_v2 as native_resources
from marine_echo.training import native_ssl as legacy_core

IMPLEMENTER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
MAIN = Path(__file__).resolve().parents[3].parent / "marine-echo-jepa"
METHODS = ("shared_ssl", "masked_ssl", "direct", "random_frozen", "permuted_ssl")
HELPER_ROOT = Path(legacy_core.__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Config:
    architecture: str = ARCHITECTURE
    method: str = "shared_ssl"
    seed: int = 7
    history: int = 96
    pretrain_updates: int = 6000
    readout_updates: int = 500
    batch_size: int = 64
    width: int = 192
    latent: int = 64
    blocks: int = 4
    heads: int = 4
    lr: float = 0.0003
    weight_decay: float = 0.0001
    sigreg_weight: float = 0.03
    pretrain_cadence: int = 1500
    readout_cadence: int = 250
    patience: int = 4
    min_daily_anchors: int = 18
    cf_source: str = str(MAIN / "external/cf-jepa-vnext")
    cf_width: int = 256
    cf_latent: int = 128
    cf_blocks: int = 5
    cf_lr: float = 0.00034
    cf_weight_decay: float = 0.05
    selection_policy: str = "train_only_frozen_probe_per_candidate_daily_dev_patience4"
    inference_replay_atol_db: float = 0.00002
    inference_replay_rtol: float = 0.00001

    def to_dict(self):
        return asdict(self)

    def validate(self, *, correctness_smoke=False):
        if self.seed not in (7, 13, 23) or (self.lr, self.weight_decay, self.sigreg_weight) != (
            0.0003,
            0.0001,
            0.03,
        ):
            raise ValueError("Band seed/optimizer/encoded SIGReg coefficient are frozen.")
        if self.selection_policy != Config.__dataclass_fields__["selection_policy"].default:
            raise ValueError("Band checkpoint selection policy is frozen.")
        integers = (
            "seed",
            "history",
            "pretrain_updates",
            "readout_updates",
            "batch_size",
            "width",
            "latent",
            "blocks",
            "heads",
            "pretrain_cadence",
            "readout_cadence",
            "patience",
            "min_daily_anchors",
        )
        if any(type(getattr(self, name)) is not int for name in integers):
            raise ValueError("Band integer dimensions/budgets must be exact integers.")
        if any(
            not isinstance(getattr(self, name), (int, float))
            or isinstance(getattr(self, name), bool)
            or not math.isfinite(getattr(self, name))
            for name in (
                "lr",
                "weight_decay",
                "sigreg_weight",
                "inference_replay_atol_db",
                "inference_replay_rtol",
            )
        ):
            raise ValueError("Band numeric configuration must be finite.")
        if self.architecture != ARCHITECTURE or self.history != 96:
            raise ValueError("Band family requires its exact architecture and H96.")
        if not correctness_smoke:
            fixed = (
                self.seed,
                self.width,
                self.latent,
                self.blocks,
                self.heads,
                self.lr,
                self.weight_decay,
                self.sigreg_weight,
                self.batch_size,
                self.readout_updates,
                self.pretrain_cadence,
                self.readout_cadence,
            )
            if fixed != (self.seed, 192, 64, 4, 4, 0.0003, 0.0001, 0.03, 64, 500, 1500, 250):
                raise ValueError(
                    "Band scientific recipe is frozen; no dimension/objective/schedule tuning."
                )
            expected_updates = (
                0 if self.method == "random_frozen" else 3000 if self.method == "direct" else 6000
            )
            if self.pretrain_updates != expected_updates:
                raise ValueError("Band method requires its prospective fixed update recipe.")
        if self.method not in METHODS or self.history not in (24, 96):
            raise ValueError("Unknown method or undeclared history.")
        if not 0 <= self.pretrain_updates <= 50000 or not 1 <= self.readout_updates <= 5000:
            raise ValueError("Update ceilings exceeded or no readout updates.")
        if self.method == "direct" and self.pretrain_updates > 5000:
            raise ValueError("Supervised/direct backbone updates must be <=5000.")
        if self.batch_size < 2 or min(self.pretrain_cadence, self.readout_cadence) < 1:
            raise ValueError("Batch/cadence must permit nonidentity paired SSL.")
        if self.patience != 4 or self.min_daily_anchors != 18:
            raise ValueError("Frozen protocol requires patience4 and >=18 anchors.")
        if not correctness_smoke and (
            self.width != 192 or self.latent not in (64, 128) or self.blocks != 4 or self.heads != 4
        ):
            raise ValueError(
                "Real fit must use declared architecture; tiny models are correctness only."
            )
        if self.lr <= 0 or self.weight_decay < 0 or self.sigreg_weight < 0:
            raise ValueError("Invalid optimizer/objective configuration.")
        if not correctness_smoke:
            candidates = (
                1
                if self.method == "random_frozen"
                else max(1, math.ceil(self.pretrain_updates / self.pretrain_cadence))
            )
            planned_supervised = candidates * self.readout_updates + (
                self.pretrain_updates if self.method == "direct" else 0
            )
            if planned_supervised > 5000:
                raise ValueError(
                    "Planned total supervised/readout updates exceed5000 for this joint trajectory."
                )
            if self.method == "cf_jepa" and (
                self.cf_width,
                self.cf_latent,
                self.cf_blocks,
                self.cf_lr,
                self.cf_weight_decay,
            ) != (256, 128, 5, 0.00034, 0.05):
                raise ValueError(
                    "CF baseline requires pinned author128 dimensions/LR/weight decay."
                )


def required_sources(config: Config) -> list[Path]:
    """Bind all band factories plus immutable inherited/scaler/loader sources."""
    if config.method not in METHODS or config.architecture != ARCHITECTURE:
        raise ValueError("Band source closure rejects legacy architecture and CF.")
    paths = [
        Path(__file__),
        Path(native_temporal.__file__),
        Path(legacy_temporal.__file__),
        Path(legacy_core.__file__),
        Path(legacy_encoder.__file__),
        Path(sigreg.__file__),
        Path(native_resources.__file__),
        Path(native_resources.original.__file__),
        native_resources.OWNER_PATH,
        MAIN / 'docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md',
        Path(__file__).resolve().parents[1] / "inference/native_band_replication_acoustic.py",
        Path(__file__).with_name("native_band_operational_downstream.py"),
    ]
    paths += [
        Path(__file__).resolve().parents[1] / "inference/native_band_replication_encoder.py",
        Path(__file__).resolve().parents[1] / "inference/native_band_replication_latent.py",
        HELPER_ROOT / "training/native_band_ssl.py",
        HELPER_ROOT / "training/native_band_downstream.py",
        HELPER_ROOT / "inference/native_band_acoustic.py",
        HELPER_ROOT / "inference/native_latent.py",
    ]
    paths += [
        HELPER_ROOT / name
        for name in (
            "__init__.py",
            "models/__init__.py",
            "training/__init__.py",
            "data/__init__.py",
            "data/native_ssl_corpus.py",
            "training/aeon_corpus.py",
        )
    ]
    return list(dict.fromkeys(p.resolve() for p in paths))


def _check_budget(review):
    """Admission only; no owner receipt is a scientific prefit approval."""
    path = Path(review.get("budget_resolution_path", "")).resolve()
    expected = (
        Path(__file__).resolve().parents[3]
        / "orchestration/native_band_budget_owner_resolution_v1.json"
    )
    if path != expected or review.get("band_budget_status") != "ROOT_RESOLVED":
        raise ValueError("Exact integrated root budget receipt is required.")
    if review.get("bindings", {}).get(str(path)) != sha256(path):
        raise ValueError("Budget receipt source binding differs.")
    receipt = json.loads(path.read_bytes())
    if (
        receipt.get("kind") != "native_band_owner_budget_resolution_v1"
        or receipt.get("status") != "ROOT_RESOLVED"
    ):
        raise ValueError("Owner resolution is missing or pending.")
    for key, value in {
        "seed7_recipe_limit_including_controls": 11,
        "revision_gpu_hours_limit_full_owned": 12,
        "aggregate_gpu_hours_limit_full_owned": 96,
    }.items():
        if type(receipt.get(key)) is not int or receipt[key] != value:
            raise ValueError("Replication budgets remain exactly11/12/96.")


def check_prefit(
    train, dev, split, protocol, config_path, review_path, *, config, correctness_smoke=False
):
    """Verify exact artifacts BEFORE np.load, scaler fitting, or optimizer setup."""
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if review.get("status") != "APPROVED_PREFIT":
        raise ValueError("Independent APPROVED_PREFIT review required before fitting.")
    if review.get("architecture") != ARCHITECTURE:
        raise ValueError("Prefit review must explicitly admit the new band architecture.")
    expected_kind = (
        "SYNTHETIC_CORRECTNESS_ONLY" if correctness_smoke else "REAL_TRAIN_DEVELOPMENT_FIT"
    )
    if review.get("evidence_kind") != expected_kind or set(review.get("allowed_roles", [])) != {
        "train",
        "development",
    }:
        raise ValueError("Prefit evidence kind and explicit TRAIN/development roles must match.")
    if not correctness_smoke:
        budget = review.get("budget_resolution_path")
        if review.get("band_budget_status") != "ROOT_RESOLVED" or not budget:
            raise ValueError("Root must resolve the screen budget before a distinct new prefit.")
        budget_path = Path(budget).resolve()
        if review.get("bindings", {}).get(str(budget_path)) != sha256(budget_path):
            raise ValueError("Exact root budget resolution binding is required.")
        _check_budget(review)
    reviewer = review.get("reviewer_session_id")
    implementer = review.get("implementer_session_id", IMPLEMENTER_SESSION_ID)
    if implementer != IMPLEMENTER_SESSION_ID or not review.get("root_coordinator_session_id"):
        raise ValueError("Exact implementer and distinct root identities are required.")
    excluded = (implementer, IMPLEMENTER_SESSION_ID, review.get("root_coordinator_session_id"))
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or reviewer != reviewer.strip()
        or reviewer.casefold() in {str(v).casefold() for v in excluded}
    ):
        raise ValueError("Reviewer must be a distinct session from the implementer.")
    if config.seed not in review.get("allowed_seeds", []):
        raise ValueError("Seed is outside the exact independent replication review.")
    if config.method not in review.get("allowed_methods", []):
        raise ValueError("Method is not approved.")
    paths = [
        Path(p).resolve() for p in (train, dev, split, protocol, config_path)
    ] + required_sources(config)
    bindings = review.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Required source/corpus/config/protocol bindings are missing.")
    for path in paths:
        if bindings.get(str(path)) != sha256(path):
            raise ValueError(f"Missing or stale prefit binding: {path}")
    for name, path in (
        ("train_npz_sha256", train),
        ("dev_npz_sha256", dev),
        ("split_sha256", split),
        ("protocol_sha256", protocol),
    ):
        if review.get(name) != sha256(Path(path)):
            raise ValueError(f"Prefit identity mismatch: {name}")
    approved_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if approved_config != config.to_dict():
        raise ValueError("Runtime configuration differs from reviewed configuration.")
    return review


def load_corpus(path, *, role, split_hash, correctness_smoke=False):
    with np.load(path, allow_pickle=False) as archive:
        # Inspect roles before reading numerical members.
        if str(archive["corpus_role"].item()) != role:
            raise ValueError(f"Expected {role} corpus; test-role payloads are prohibited.")
        if str(archive["split_sha256"].item()) != split_hash:
            raise ValueError("Corpus split identity mismatch.")
        data = {}
        for key in archive.files:
            data[key] = archive[key]
            if psutil.Process().memory_info().rss >= 22 * 2**30:
                raise RuntimeError("RSS resource limit exceeded during corpus loading.")
    validate_corpus(data, role=role, correctness_smoke=correctness_smoke)
    return data


def validate_corpus(data, *, role, correctness_smoke=False):
    if str(data["corpus_role"].item()) != role or role not in ("train", "development"):
        raise ValueError("Only declared train/development corpora are admitted.")
    n = len(data["x"])
    if n < 2 or data["x"].shape != (n, 96, 4):
        raise ValueError("Require identical issued support with full 96-interval context.")
    shapes = {
        "observed": (n, 96, 4),
        "metadata": (n, 4, 10),
        "future": (n, 3, 4, 4),
        "future_observed": (n, 3, 4, 4),
        "y": (n, 3),
        "y_observed": (n, 3),
        "query": (n, 3, 10),
        "target_dates": (n, 3),
        "row_id": (n,),
        "deployment": (n,),
        "cutoff": (n,),
        "ssl_eligible": (n,),
        "context_ids": (n, 96),
        "future_ids": (n, 3, 4),
    }
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"Invalid corpus shape: {key}")
    for key in ("observed", "future_observed", "y_observed", "ssl_eligible"):
        if data[key].dtype != bool:
            raise ValueError(f"Expected boolean mask: {key}")
    for key in ("x", "metadata", "future", "y", "query"):
        if data[key].dtype != np.float32:
            raise ValueError(f"Expected float32: {key}")
    for key in ("row_id", "deployment", "target_dates"):
        if data[key].dtype.kind != "U":
            raise ValueError(f"Expected unicode identity/date: {key}")
    for key in ("cutoff", "context_ids", "future_ids"):
        if data[key].dtype.kind not in "iu":
            raise ValueError(f"Expected integer source interval identity: {key}")
    for value, mask in (("x", "observed"), ("future", "future_observed"), ("y", "y_observed")):
        if not np.isfinite(data[value][data[mask]]).all():
            raise ValueError(f"Nonfinite observed numeric payload: {value}")
    if not np.isfinite(data["metadata"]).all() or not np.isfinite(data["query"]).all():
        raise ValueError("Nonfinite measurement metadata.")
    for key, mask_key, shape in (
        ("context_metadata", "observed", (n, 96, 4, 10)),
        ("future_metadata", "future_observed", (n, 3, 4, 4, 10)),
    ):
        if key in data:
            actual = data[key]
            static = (
                data["metadata"][:, None, :, :9]
                if key == "context_metadata"
                else data["metadata"][:, None, None, :, :9]
            )
            if actual.shape != shape or not np.isfinite(actual).all():
                raise ValueError(f"Invalid per-observation metadata: {key}")
            mismatch = np.any(actual[..., :9] != static, axis=-1) & data[mask_key]
            if mismatch.any():
                raise ValueError(
                    "Observed metadata must preserve the reviewed static broadcast invariant."
                )
    context = data["cutoff"][:, None] - np.arange(95, -1, -1)
    future = data["cutoff"][:, None, None] + np.array([1, 3, 6])[None, :, None] + np.arange(4)
    if not np.array_equal(context, data["context_ids"]) or not np.array_equal(
        future, data["future_ids"]
    ):
        raise ValueError(
            "Context/future IDs must be contiguous, native horizons, and nonoverlapping."
        )
    if not np.array_equal(data["ssl_eligible"], data["future_observed"][:, :, :, 0].all((1, 2))):
        raise ValueError("SSL eligibility must equal primary future-block observation support.")
    if len(np.unique(data["row_id"])) != n:
        raise ValueError("Duplicate issued row identity.")
    if correctness_smoke and not all(str(r).startswith("synthetic-") for r in data["row_id"]):
        raise ValueError("CPU correctness smoke admits explicitly synthetic rows only.")
    if not correctness_smoke:
        if (data["query"][:, :, 4] <= data["query"][:, :, 3]).any():
            raise ValueError("Query must retain valid native depth bounds.")
        if not np.allclose(data["query"][:, :, 0], 38000 / 455000):
            raise ValueError("Native query must request the published 38-kHz product.")
    if not np.array_equal(data["query"][:, :, -1], np.tile([1, 3, 6], (n, 1))):
        raise ValueError("Query offsets differ from declared horizons.")


@dataclass
class Scalers:
    channel_mean: np.ndarray
    channel_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, data, *, split_path):
        if str(data["corpus_role"].item()) != "train":
            raise ValueError("Scaler fitting requires train corpus only.")
        validate_membership(data, split_path, role="train")
        means, stds = [], []
        for c in range(4):
            values = data["x"][:, :, c][data["observed"][:, :, c]].astype(np.float64)
            means.append(float(values.mean()) if len(values) else 0)
            stds.append(max(float(values.std()), 0.01) if len(values) else 1)
        target_mean, target_std = [], []
        for h in range(3):
            values = data["y"][:, h][data["y_observed"][:, h]].astype(np.float64)
            if not len(values):
                raise ValueError("Every horizon requires observed train targets.")
            target_mean.append(values.mean())
            target_std.append(max(values.std(), 0.01))
        return cls(
            *[np.asarray(v, dtype=np.float32) for v in (means, stds, target_mean, target_std)]
        )

    def channels(self, values, mask):
        return np.where(
            mask, (np.where(mask, values, 0) - self.channel_mean) / self.channel_std, 0
        ).astype(np.float32)

    def targets(self, values, mask):
        return np.where(
            mask, (np.where(mask, values, 0) - self.target_mean) / self.target_std, 0
        ).astype(np.float32)

    def to_dict(self):
        return {key: value.tolist() for key, value in vars(self).items()}

    @classmethod
    def from_dict(cls, data):
        checked = legacy_encoder._scalers(data)
        return cls(**{key: value.copy() for key, value in vars(checked).items()})


def validate_membership(data, split_path, *, role):
    """Validate per-row source/deployment membership in the hash-reviewed split."""
    if str(data["split_sha256"].item()) != sha256(Path(split_path)):
        raise ValueError("Scaler/reader requires verified split identity.")
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    allowed = {s["deployment"]: s["archive_sha256"] for s in split["sources"] if s["role"] == role}
    archives = data.get("archive_sha256")
    if archives is None or archives.shape != (len(data["x"]),):
        raise ValueError("Per-row source provenance required before fitting.")
    if any(
        allowed.get(str(d)) != str(a) for d, a in zip(data["deployment"], archives, strict=True)
    ):
        raise ValueError(f"Corpus lacks reviewed {role} source/deployment membership.")


def initialize_model(seed, *, width=192, latent=64, blocks=4, heads=4, method="shared_ssl"):
    if method not in METHODS:
        raise ValueError("Band factory rejects CF and unknown methods.")
    # fork_rng makes every common tensor independent of prior constructor consumption.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NativeTemporalModel(width, latent, blocks, heads)
        # Head initialization is independent of backbone/predictor RNG consumption.
        torch.manual_seed(seed + 100000)
        model.readout = QueryHead(latent, 5, latent)
    return model


def batch_indices(pool, size, seed, phase, step):
    phase_code = {"pretrain": 1, "readout": 2}[phase]
    rng = np.random.default_rng(np.random.SeedSequence([seed, phase_code, step]))
    return rng.choice(np.asarray(pool), size=min(size, len(pool)), replace=False)


def model_dimensions(config):
    if config.method not in METHODS or config.architecture != ARCHITECTURE:
        raise ValueError("Band model dimensions reject CF/legacy architecture.")
    return {
        "width": config.width,
        "latent": config.latent,
        "blocks": config.blocks,
        "heads": config.heads,
    }


def paired_indices(indices):
    if len(indices) < 2 or len(np.unique(indices)) != len(indices):
        raise ValueError("Permuted control requires at least two distinct rows.")
    return np.roll(indices, 1)


def pinball(predictions, targets, observed):
    q = predictions.new_tensor(QUANTILES)
    residual = targets.unsqueeze(-1) - predictions
    values = torch.maximum(q * residual, (q - 1) * residual).mean(-1)
    # Equal horizon weighting on training support; no future-mask issuance filtering.
    scores = [values[:, h][observed[:, h]].mean() for h in range(3) if observed[:, h].any()]
    if not scores:
        raise ValueError("Batch has no observed supervised targets.")
    return torch.stack(scores).mean()


def daily_metrics(
    predictions, targets, observed, dates, deployment, *, support=None, min_anchors=18
):
    """Mean over source dates, then horizons, then deployments; native dB scale."""
    if (
        predictions.shape != (*targets.shape, 5)
        or targets.shape != observed.shape
        or targets.shape != dates.shape
    ):
        raise ValueError("Invalid row artifact dimensions.")
    if not np.isfinite(predictions).all() or (np.diff(predictions, axis=-1) < 0).any():
        raise ValueError("Predictions must be finite and monotonic for every issuance.")
    common = np.ones_like(observed, bool) if support is None else np.asarray(support, dtype=bool)
    if common.shape != observed.shape:
        raise ValueError("Shared assessment support has wrong shape.")
    mask = observed & common & (dates != "") & np.isfinite(targets)
    safe_targets = np.where(mask, targets, 0).astype(np.float64)
    residual = safe_targets[..., None] - predictions.astype(np.float64)
    q = np.asarray(QUANTILES)
    loss = np.maximum(q * residual, (q - 1) * residual).mean(-1)
    deployments, daily, scored = {}, [], np.zeros(3, int)
    eligible_dates = np.zeros(3, int)
    for d in sorted(np.unique(deployment).tolist()):
        horizon_scores = []
        for h in range(3):
            scores = []
            for date in sorted(np.unique(dates[deployment == d, h]).tolist()):
                selected = (deployment == d) & (dates[:, h] == date) & mask[:, h]
                count = int(selected.sum())
                if count >= min_anchors:
                    score = float(loss[selected, h].mean())
                    scores.append(score)
                    daily.append(
                        {
                            "deployment": d,
                            "horizon": (1, 3, 6)[h],
                            "source_date": date,
                            "anchors": count,
                            "pinball": score,
                        }
                    )
                    scored[h] += count
                    eligible_dates[h] += 1
            horizon_scores.append(float(np.mean(scores)) if scores else None)
        complete = all(s is not None for s in horizon_scores)
        deployments[d] = {
            "horizon_pinball": horizon_scores,
            "primary_pinball": float(np.mean(horizon_scores)) if complete else None,
        }
    complete = bool(deployments) and all(
        v["primary_pinball"] is not None for v in deployments.values()
    )
    primary = (
        float(np.mean([v["primary_pinball"] for v in deployments.values()])) if complete else None
    )
    return {
        "primary_pinball": primary,
        "deployments": deployments,
        "daily": daily,
        "coverage": {
            "issued_rows": len(targets),
            "target_observed_rows": observed.sum(0).tolist(),
            "common_support_rows": common.sum(0).tolist(),
            "scored_rows": scored.tolist(),
            "eligible_deployment_dates": eligible_dates.tolist(),
        },
        "quantiles": list(QUANTILES),
        "horizons": [1, 3, 6],
        "minimum_daily_anchors": min_anchors,
        "aggregation": "source_date_then_horizon_then_deployment_equal_weight",
        "unit": "source_reported_conditioned_native_product_dB",
    }


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].astype(np.int64)),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    raw = state["numpy"]
    np.random.set_state((raw[0], raw[1].cpu().numpy().astype(np.uint32), raw[2], raw[3], raw[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state requires CUDA for training continuation.")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def schedule(optimizer, total):
    warmup = max(1, math.ceil(total * 0.1))

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step - warmup) / max(total - warmup, 1))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def atomic_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".pending")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def tensor_batch(data, indices, scalers, history, device, *, future=False, include_targets=True):
    x, mask = data["x"][indices, -history:], data["observed"][indices, -history:]
    arrays = {
        "x": scalers.channels(x, mask),
        "observed": mask,
        "metadata": data["metadata"][indices],
        "query": data["query"][indices],
    }
    if include_targets:
        arrays.update(
            y=scalers.targets(data["y"][indices], data["y_observed"][indices]),
            y_observed=data["y_observed"][indices],
        )
    if future:
        arrays.update(
            future=scalers.channels(data["future"][indices], data["future_observed"][indices]),
            future_observed=data["future_observed"][indices],
        )
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


@torch.no_grad()
def predict(model, data, scalers, config, device):
    model.eval()
    result = []
    for start in range(0, len(data["x"]), config.batch_size):
        ids = np.arange(start, min(start + config.batch_size, len(data["x"])))
        batch = tensor_batch(data, ids, scalers, config.history, device, include_targets=False)
        values = (
            model.forecast(batch["x"], batch["observed"], batch["metadata"], batch["query"])
            .cpu()
            .numpy()
        )
        result.append(
            values * scalers.target_std[None, :, None] + scalers.target_mean[None, :, None]
        )
    return np.concatenate(result)


def predict_from_checkpoint(path, data, *, device="cpu"):
    from marine_echo.inference.native_band_replication_acoustic import NativeBandAcousticPredictor

    return NativeBandAcousticPredictor(path, device=device).forecast(
        data["x"], data["observed"], data["metadata"], data["query"]
    )


class Resources:
    def __init__(self, device, output):
        self.device, self.output = device, Path(output)
        self.peak_rss = self.peak_allocated = self.peak_reserved = 0
        self.gpu_elapsed = 0.0
        self.elapsed = 0.0
        self.started = time.monotonic()
        self.lock = None

    def __enter__(self):
        if str(self.device).startswith("cuda"):
            if not torch.cuda.is_available() or torch.version.cuda is None:
                raise RuntimeError("Verified CUDA PyTorch required; no CPU fallback or install.")
            # Shared local native-run lock, never delete stale/unowned locks automatically.
            folder = Path(__file__).resolve().parents[3] / "evidence/ssl-builder-v1"
            folder.mkdir(parents=True, exist_ok=True)
            self.lock = folder / "gpu-owner.lock"
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"pid": os.getpid(), "output": str(self.output.resolve())}, stream)
            try:
                native_resources.inspect_gpu_ownership(self.output)
                torch.cuda.set_per_process_memory_fraction(
                    min(
                        1.0,
                        0.98
                        * 10
                        * 2**30
                        / torch.cuda.get_device_properties(self.device).total_memory,
                    ),
                    self.device,
                )
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            except BaseException:
                self.lock.unlink()
                self.lock = None
                raise
        self.started = time.monotonic()
        return self

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.peak_rss >= 22 * 2**30:
            raise RuntimeError("RSS resource limit exceeded.")
        if str(self.device).startswith("cuda"):
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved = max(
                self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
            )
            if max(self.peak_allocated, self.peak_reserved) >= 10 * 2**30:
                raise RuntimeError("CUDA resource limit exceeded.")

    def snapshot(self):
        self.check()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - self.started + self.elapsed
        return {
            "peak_rss_bytes": self.peak_rss,
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
            "elapsed_seconds": elapsed,
            "elapsed_gpu_seconds": elapsed if str(self.device).startswith("cuda") else 0,
            "gpu_time_definition": "synchronized elapsed device-owned runtime, including probes and evaluation",
        }

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.unlink()


def sequence_hash(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def run(
    train_path,
    dev_path,
    split_path,
    protocol_path,
    config_path,
    review_path,
    *,
    output,
    config,
    device="cuda",
    resume=None,
    correctness_smoke=False,
    stop_after=None,
):
    config.validate(correctness_smoke=correctness_smoke)
    if not correctness_smoke and str(device) != "cuda:0":
        raise ValueError("Scientific replication requires explicitly indexed cuda:0.")
    if not correctness_smoke and config.method == "direct":
        raise ValueError(
            "Scientific direct comparisons require the separate fixed native_band_replication_downstream endpoint."
        )
    if str(device).startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if correctness_smoke and device != "cpu":
        raise ValueError("Correctness optimizer smokes are synthetic CPU only.")
    review = check_prefit(
        train_path,
        dev_path,
        split_path,
        protocol_path,
        config_path,
        review_path,
        config=config,
        correctness_smoke=correctness_smoke,
    )
    train = load_corpus(
        train_path,
        role="train",
        split_hash=review["split_sha256"],
        correctness_smoke=correctness_smoke,
    )
    dev = load_corpus(
        dev_path,
        role="development",
        split_hash=review["split_sha256"],
        correctness_smoke=correctness_smoke,
    )
    validate_membership(train, split_path, role="train")
    validate_membership(dev, split_path, role="development")
    if set(train["deployment"]) & set(dev["deployment"]) or set(train["row_id"]) & set(
        dev["row_id"]
    ):
        raise ValueError("Reserved native train/development deployments must be disjoint.")
    output = Path(output)
    if output.exists() and resume is None and any(output.iterdir()):
        raise ValueError("Band runs require a new empty output or explicit resume.")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and resume is None:
        raise ValueError("Existing run requires explicit resume; no silent overwrite.")
    bindings = {
        str(Path(p).resolve()): sha256(Path(p))
        for p in [
            train_path,
            dev_path,
            split_path,
            protocol_path,
            config_path,
            *required_sources(config),
            *(
                [Path(review["budget_resolution_path"])]
                if "budget_resolution_path" in review
                else []
            ),
        ]
    }
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # Deterministic CPU continuation is required; cross-device replay is numerical.
    torch.use_deterministic_algorithms(True)
    model = initialize_model(
        config.seed,
        **model_dimensions(config),
        method=config.method,
    )
    initial_readout = cpu_state(model.readout)
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            saved.get("kind") != "native_band_replication_ssl_resume_v2"
            or saved.get("architecture") != ARCHITECTURE
            or saved.get("device") != str(device)
            or saved["state"].get("review_sha256") != sha256(Path(review_path))
            or saved["state"]["bindings"] != bindings
            or saved["state"]["config"] != config.to_dict()
        ):
            raise ValueError("Resume checkpoint code/data/config identities differ.")
    scalers = (
        Scalers.from_dict(saved["state"]["scalers"])
        if saved is not None
        else Scalers.fit(train, split_path=split_path)
    )
    eligible = np.flatnonzero(train["ssl_eligible"])
    supervised = np.flatnonzero(train["y_observed"].any(1))
    if config.method not in ("random_frozen", "direct") and len(eligible) < 2:
        raise ValueError("At least two eligible TRAIN SSL rows required.")
    total_pretrain = 0 if config.method == "random_frozen" else config.pretrain_updates
    # Direct learns features under supervised gradients, then receives identical
    # frozen probes. Its additional labels/compute are separately counted.
    direct_head = copy.deepcopy(model.readout) if config.method == "direct" else None
    parameters = (
        (list(model.online.parameters()) + list(model.predictors.parameters()))
        if config.method == "cf_jepa"
        else list(model.encoder.parameters())
    )
    if config.method in ("shared_ssl", "permuted_ssl"):
        parameters += list(model.predictor.parameters())
    elif config.method == "masked_ssl":
        parameters += list(model.reconstruction.parameters())
    elif direct_head is not None:
        parameters += list(direct_head.parameters())
    parameters = [p for p in parameters if p.requires_grad]
    pre_optimizer = torch.optim.AdamW(
        parameters,
        lr=config.cf_lr if config.method == "cf_jepa" else config.lr,
        weight_decay=config.cf_weight_decay if config.method == "cf_jepa" else config.weight_decay,
    )
    pre_scheduler = schedule(pre_optimizer, max(total_pretrain, 1))
    read_optimizer = torch.optim.AdamW(
        model.readout.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    read_scheduler = schedule(read_optimizer, config.readout_updates)
    state = {
        "phase": "pretrain" if total_pretrain else "probe",
        "pretrain_step": 0,
        "readout_step": 0,
        "probe_step": 0,
        "optimizer_steps": 0,
        "probe_bad": 0,
        "encoder_bad": 0,
        "probe_best_score": None,
        "best_score": None,
        "probe_best_head": None,
        "best_model": None,
        "selected_pretrain_step": None,
        "records": [],
        "sequence": [],
        "bindings": bindings,
        "config": config.to_dict(),
        "review_sha256": sha256(Path(review_path)),
        "initial_readout": initial_readout,
        "scalers": scalers.to_dict(),
    }
    if resume is not None:
        state = saved["state"]
        from marine_echo.inference.native_band_replication_acoustic import _check_state

        _check_state(saved["model"], model.state_dict())
        model.load_state_dict(saved["model"], strict=True)
        if direct_head is not None:
            direct_head.load_state_dict(saved["direct_head"])
        pre_optimizer.load_state_dict(saved["pre_optimizer"])
        pre_scheduler.load_state_dict(saved["pre_scheduler"])
        read_optimizer.load_state_dict(saved["read_optimizer"])
        read_scheduler.load_state_dict(saved["read_scheduler"])
        scalers = Scalers.from_dict(state["scalers"])
        if saved["device"] != str(device):
            raise ValueError(
                "Training resume requires the original device kind; inference is portable."
            )
    with Resources(device, output) as resources:
        resources.check()
        model.to(device)
        if direct_head is not None:
            direct_head.to(device)
        if resume is not None:
            restore_rng(saved["rng"])
        for optimizer in (pre_optimizer, read_optimizer):
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(device)
        if resume is not None:
            resources.elapsed = saved["resources"]["elapsed_seconds"]
            resources.peak_rss = saved["resources"]["peak_rss_bytes"]
            resources.peak_allocated = saved["resources"]["peak_allocated_bytes"]
            resources.peak_reserved = saved["resources"]["peak_reserved_bytes"]

        def save(name="latest.pt"):
            payload = {
                "kind": "native_band_replication_ssl_resume_v2",
                "architecture": ARCHITECTURE,
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"
                if correctness_smoke
                else "REAL_TRAIN_DEVELOPMENT_FIT",
                "correctness_smoke": correctness_smoke,
                "state": state,
                "model": cpu_state(model),
                "direct_head": cpu_state(direct_head) if direct_head is not None else None,
                "pre_optimizer": pre_optimizer.state_dict(),
                "pre_scheduler": pre_scheduler.state_dict(),
                "read_optimizer": read_optimizer.state_dict(),
                "read_scheduler": read_scheduler.state_dict(),
                "rng": rng_state(),
                "resources": resources.snapshot(),
                "device": str(device),
            }
            atomic_checkpoint(output / name, payload)
            if name != "latest.pt":
                atomic_checkpoint(output / "latest.pt", payload)

        def begin_probe():
            nonlocal read_optimizer, read_scheduler
            model.readout.load_state_dict(state["initial_readout"])
            model.freeze_encoder()
            model.readout.requires_grad_(True).train()
            read_optimizer = torch.optim.AdamW(
                model.readout.parameters(), lr=config.lr, weight_decay=config.weight_decay
            )
            read_scheduler = schedule(read_optimizer, config.readout_updates)
            state.update(
                phase="probe",
                readout_step=0,
                probe_bad=0,
                probe_best_score=None,
                probe_best_head=None,
            )

        if state["phase"] == "probe" and state["readout_step"] == 0:
            begin_probe()
        start_steps = state["optimizer_steps"]
        while state["phase"] != "done":
            if state["phase"] == "pretrain":
                model.train()
                if config.method == "cf_jepa":
                    model.encoder.requires_grad_(False).eval()
                else:
                    model.encoder.requires_grad_(True)
                step = state["pretrain_step"]
                pool = supervised if config.method == "direct" else eligible
                ids = batch_indices(pool, config.batch_size, config.seed, "pretrain", step)
                batch = tensor_batch(
                    train,
                    ids,
                    scalers,
                    config.history,
                    device,
                    future=config.method not in ("direct", "masked_ssl"),
                    include_targets=config.method == "direct",
                )
                pre_optimizer.zero_grad(set_to_none=True)
                if config.method in ("shared_ssl", "permuted_ssl"):
                    details = model.shared_loss(
                        batch["x"],
                        batch["observed"],
                        batch["metadata"],
                        batch["future"],
                        batch["future_observed"],
                        batch["query"],
                        weight=config.sigreg_weight,
                        permuted=config.method == "permuted_ssl",
                    )
                    loss = details["loss"]
                elif config.method == "masked_ssl":
                    generator = torch.Generator(device=device).manual_seed(config.seed + step * 17)
                    loss = model.masked_loss(
                        batch["x"], batch["observed"], batch["metadata"], generator=generator
                    )
                elif config.method == "direct":
                    latent = model.encoder.encode(batch["x"], batch["observed"], batch["metadata"])
                    loss = pinball(
                        direct_head(latent, batch["query"]).sort(-1).values,
                        batch["y"],
                        batch["y_observed"],
                    )
                else:
                    loss = model.cf_loss(
                        batch["x"],
                        batch["observed"],
                        batch["metadata"],
                        batch["future"],
                        batch["future_observed"],
                        batch["query"],
                        step=step,
                        total=total_pretrain,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; stopping without masking failure.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                pre_optimizer.step()
                pre_scheduler.step()
                if config.method == "cf_jepa":
                    model.update_ema(step, total_pretrain)
                state["pretrain_step"] += 1
                state["optimizer_steps"] += 1
                targets = paired_indices(ids) if config.method == "permuted_ssl" else ids
                state["sequence"].append(
                    {
                        "phase": "pretrain",
                        "step": step,
                        "context_indices": ids.tolist(),
                        "target_indices": targets.tolist(),
                        "context_sha256": sequence_hash(ids),
                        "target_multiset_sha256": sequence_hash(np.sort(targets)),
                        "loss": float(loss.detach()),
                    }
                )
                cadence = config.pretrain_cadence
                if (
                    state["pretrain_step"] % cadence == 0
                    or state["pretrain_step"] == total_pretrain
                ):
                    begin_probe()
                    save(f"pretrain-{state['pretrain_step']:06d}.pt")
            else:
                model.eval()
                model.freeze_encoder()
                model.readout.train()
                step = state["readout_step"]
                ids = batch_indices(supervised, config.batch_size, config.seed, "readout", step)
                batch = tensor_batch(train, ids, scalers, config.history, device)
                read_optimizer.zero_grad(set_to_none=True)
                predicted = model.forecast(
                    batch["x"], batch["observed"], batch["metadata"], batch["query"]
                )
                loss = pinball(predicted, batch["y"], batch["y_observed"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite readout loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.readout.parameters(), 1, error_if_nonfinite=True
                )
                read_optimizer.step()
                read_scheduler.step()
                state["readout_step"] += 1
                state["optimizer_steps"] += 1
                state["probe_step"] += 1
                state["sequence"].append(
                    {
                        "phase": "readout",
                        "candidate": state["pretrain_step"],
                        "step": step,
                        "context_indices": ids.tolist(),
                        "context_sha256": sequence_hash(ids),
                        "loss": float(loss.detach()),
                    }
                )
                evaluate = (
                    state["readout_step"] % config.readout_cadence == 0
                    or state["readout_step"] == config.readout_updates
                )
                if evaluate:
                    predictions = predict(model, dev, scalers, config, device)
                    metrics = daily_metrics(
                        predictions,
                        dev["y"],
                        dev["y_observed"],
                        dev["target_dates"],
                        dev["deployment"],
                    )
                    score = metrics["primary_pinball"]
                    if score is None:
                        raise ValueError(
                            "Development lacks >=18 anchors/day at every deployment/horizon."
                        )
                    state["records"].append(
                        {
                            "pretrain_step": state["pretrain_step"],
                            "readout_step": state["readout_step"],
                            "daily_dev_pinball": score,
                        }
                    )
                    if state["probe_best_score"] is None or score < state["probe_best_score"]:
                        state["probe_best_score"] = score
                        state["probe_best_head"] = cpu_state(model.readout)
                        state["probe_bad"] = 0
                    else:
                        state["probe_bad"] += 1
                    if (
                        state["readout_step"] == config.readout_updates
                        or state["probe_bad"] >= config.patience
                    ):
                        model.readout.load_state_dict(state["probe_best_head"])
                        if (
                            state["best_score"] is None
                            or state["probe_best_score"] < state["best_score"]
                        ):
                            state["best_score"] = state["probe_best_score"]
                            state["best_model"] = cpu_state(model)
                            state["selected_pretrain_step"] = state["pretrain_step"]
                            state["encoder_bad"] = 0
                        else:
                            state["encoder_bad"] += 1
                        state["phase"] = (
                            "done"
                            if state["pretrain_step"] >= total_pretrain
                            or state["encoder_bad"] >= config.patience
                            else "pretrain"
                        )
                    save(f"readout-{state['pretrain_step']:06d}-{state['readout_step']:06d}.pt")
            resources.check()
            if stop_after is not None and state["optimizer_steps"] - start_steps >= stop_after:
                save()
                return {
                    "status": "INTERRUPTED_CORRECTNESS_CHECK",
                    "steps": state["optimizer_steps"],
                }
        save("complete.pt")
        model.load_state_dict(state["best_model"])
        model.freeze_encoder()
        inference = {
            "kind": "native_band_replication_ssl_weights_only_inference_v2",
            "architecture": ARCHITECTURE,
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"
            if correctness_smoke
            else "REAL_TRAIN_DEVELOPMENT_FIT",
            "correctness_smoke": correctness_smoke,
            "model": cpu_state(model),
            "config": config.to_dict(),
            "scalers": scalers.to_dict(),
            "selected_pretrain_step": state["selected_pretrain_step"],
            "bindings": bindings,
        }
        atomic_checkpoint(output / "inference.pt", inference)
        atomic_checkpoint(
            output / "selected_encoder.pt",
            {
                "kind": "native_band_replication_ssl_selected_encoder_v2",
                "architecture": ARCHITECTURE,
                "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"
                if correctness_smoke
                else "REAL_TRAIN_DEVELOPMENT_FIT",
                "correctness_smoke": correctness_smoke,
                "encoder": cpu_state(model.encoder),
                "config": config.to_dict(),
                "scalers": scalers.to_dict(),
                "bindings": bindings,
            },
        )
        predictions = predict(model, dev, scalers, config, device)
        np.savez_compressed(
            output / "predictions.npz",
            predictions=predictions,
            targets=dev["y"],
            target_observed=dev["y_observed"],
            target_dates=dev["target_dates"],
            row_id=dev["row_id"],
            deployment=dev["deployment"],
            cutoff=dev["cutoff"],
            query=dev["query"],
            metadata=dev["metadata"],
            context_observed=dev["observed"],
            assessment_support=np.ones_like(dev["y_observed"]),
            query_native_bounds_m=dev["query"][:, :, 3:5] * 250,
            query_frequency_hz=dev["query"][:, :, 0] * 455000,
            query_interval_seconds=dev["query"][:, :, 1] * 3600,
            quantiles=np.asarray(QUANTILES),
            horizons=np.asarray([1, 3, 6]),
        )
        metrics = daily_metrics(
            predictions, dev["y"], dev["y_observed"], dev["target_dates"], dev["deployment"]
        )
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (output / "scalers.json").write_text(
            json.dumps(scalers.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        membership = {
            "train_row_ids": train["row_id"].tolist(),
            "train_deployments": train["deployment"].tolist(),
            "train_archive_sha256": train["archive_sha256"].tolist(),
            "ssl_eligible_indices": eligible.tolist(),
            "supervised_indices": supervised.tolist(),
            "sequence": state["sequence"],
        }
        (output / "membership.json").write_text(
            json.dumps(membership, indent=2) + "\n", encoding="utf-8"
        )
        report = {
            "architecture": ARCHITECTURE,
            "correctness_smoke": correctness_smoke,
            "status": "COMPLETED",
            "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"
            if correctness_smoke
            else "REAL_TRAIN_DEVELOPMENT_FIT",
            "config": config.to_dict(),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_pretrain": sum(p.numel() for p in parameters),
            "parameters_readout": sum(p.numel() for p in model.readout.parameters()),
            "pretrain_steps": state["pretrain_step"],
            "readout_steps_all_probes": state["probe_step"],
            "optimizer_steps_total": state["optimizer_steps"],
            "selected_pretrain_step": state["selected_pretrain_step"],
            "selected_daily_dev_pinball": state["best_score"],
            "selection_records": state["records"],
            "resources": resources.snapshot(),
            "bindings": bindings,
            "review_sha256": sha256(Path(review_path)),
            "membership_sha256": sha256(output / "membership.json"),
            "inference_sha256": sha256(output / "inference.pt"),
            "forecast_mode": "frozen_train_fitted_readout",
            "direct_extra_supervised_feature_updates": state["pretrain_step"]
            if config.method == "direct"
            else 0,
            "test_access": "NOT_RUN",
            "scientific_claim": "NOT_ESTABLISHED",
        }
        (output / "run.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "dev", "output", "review"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=MAIN / "configs/native_ssl_split_v1.json")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=MAIN / "docs/adr/0016-native-ssl-selection-and-source-baseline.md",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--method", choices=METHODS)
    for name, default in (
        ("seed", 7),
        ("history", 96),
        ("pretrain-updates", 6000),
        ("readout-updates", 500),
        ("batch-size", 64),
        ("latent", 64),
        ("pretrain-cadence", 1500),
        ("readout-cadence", 250),
    ):
        parser.add_argument(f"--{name}", type=int, help=f"Default without a config: {default}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--prepare-config-only",
        action="store_true",
        help="Write configuration for review without loading a corpus or fitting.",
    )
    args = parser.parse_args(argv)
    config_path = args.config or args.output / "config.json"
    values = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if args.config and config_path.exists()
        else Config().to_dict()
    )
    for field in (
        "method",
        "seed",
        "history",
        "pretrain_updates",
        "readout_updates",
        "batch_size",
        "latent",
        "pretrain_cadence",
        "readout_cadence",
    ):
        requested = getattr(args, field)
        if requested is not None:
            values[field] = requested
    config = Config(**values)
    if args.prepare_config_only:
        config.validate()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": "CONFIG_READY_FOR_REVIEW_NO_FITTING",
                    "path": str(config_path.resolve()),
                    "sha256": sha256(config_path),
                }
            )
        )
        return
    result = run(
        args.train,
        args.dev,
        args.split,
        args.protocol,
        config_path,
        args.review,
        output=args.output,
        config=config,
        device=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "pretrain_steps", "optimizer_steps_total", "inference_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
