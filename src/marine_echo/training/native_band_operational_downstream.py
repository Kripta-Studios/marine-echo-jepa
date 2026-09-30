"""Reviewed, separate downstream endpoints using the frozen native SSL helpers.

No screen probe is a downstream result. Only TRAIN supplies optimization and
scaling; development selects scheduled checkpoints on native daily pinball.
The portable inference interface never accepts future crops or target values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from marine_echo.inference import native_band_acoustic as original_inference
from marine_echo.inference.native_band_replication_acoustic import (
    NativeBandAcousticPredictor,
    _check_state,
    _configuration,
)
from marine_echo.training import native_band_operational_ssl as core
from marine_echo.training import native_band_ssl as original_core

ARCHITECTURE = core.ARCHITECTURE

IMPLEMENTER_SESSION_ID = "01a0ef20-7397-7ba1-a98c-f59bc38ddcc1"
MODES = ("frozen_readout", "full_finetune", "direct_end_to_end")
METHODS = core.METHODS
sha256 = core.sha256


@dataclass(frozen=True)
class DownstreamConfig:
    architecture: str = ARCHITECTURE
    mode: str = "frozen_readout"
    method: str = "shared_ssl"
    seed: int = 7
    history: int = 96
    updates: int | None = None
    cadence: int | None = None
    patience: int = 4
    min_daily_anchors: int = 18
    batch_size: int = 64
    lr: float = 0.0003
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    selection_policy: str = "scheduled_native_daily_pinball_earliest_strict_improvement_v1"

    def __post_init__(self):
        if self.updates is None:
            object.__setattr__(self, "updates", 2000 if self.mode == "frozen_readout" else 3000)
        if self.cadence is None:
            object.__setattr__(self, "cadence", 500 if self.mode == "frozen_readout" else 750)

    def to_dict(self):
        return asdict(self)

    def validate(self, *, correctness_smoke=False):
        if (self.lr, self.weight_decay, self.gradient_clip) != (0.0003, 0.0001, 1.0):
            raise ValueError("Band downstream optimizer/clipping are frozen.")
        if any(
            type(getattr(self, name)) is not int
            for name in (
                "seed",
                "history",
                "updates",
                "cadence",
                "patience",
                "min_daily_anchors",
                "batch_size",
            )
        ):
            raise ValueError("Downstream dimensions/budgets must be exact integers.")
        if any(
            not isinstance(getattr(self, name), (int, float))
            or isinstance(getattr(self, name), bool)
            or not np.isfinite(getattr(self, name))
            for name in ("lr", "weight_decay", "gradient_clip")
        ):
            raise ValueError("Downstream numeric configuration must be finite.")
        if self.architecture != ARCHITECTURE:
            raise ValueError("Exact band downstream architecture required.")
        if self.mode not in MODES or self.method not in METHODS:
            raise ValueError("Unsupported downstream mode/method.")
        if self.mode == "direct_end_to_end" and self.method != "direct":
            raise ValueError("direct_end_to_end requires the direct method and no ancestor.")
        if self.seed not in (7, 13, 23) or self.history != 96:
            raise ValueError("Band downstream requires admitted seed7/13/23 and H96.")
        if not isinstance(self.updates, int) or not 1 <= self.updates <= 5000:
            raise ValueError("Hard supervised trajectory limit is5000 updates.")
        if not isinstance(self.cadence, int) or not 1 <= self.cadence <= self.updates:
            raise ValueError("Invalid checkpoint cadence.")
        if self.batch_size < 1 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration.")
        if self.patience != 4 or self.min_daily_anchors != 18 or self.gradient_clip != 1:
            raise ValueError("Patience4, daily floor18, and gradient clipping1 are fixed.")
        if (
            self.selection_policy
            != DownstreamConfig.__dataclass_fields__["selection_policy"].default
        ):
            raise ValueError("Unsupported checkpoint selection policy.")
        if not correctness_smoke:
            budget = (2000, 500) if self.mode == "frozen_readout" else (3000, 750)
            if (self.updates, self.cadence) != budget or (
                self.batch_size,
                self.lr,
                self.weight_decay,
            ) != (64, 0.0003, 0.0001):
                raise ValueError(
                    "Real downstream runs require the frozen endpoint budget/optimizer."
                )


@dataclass(frozen=True)
class RunInputs:
    train: Path
    dev: Path
    train_cohort: Path
    dev_cohort: Path
    split: Path
    adr0016: Path
    protocol: Path
    config: Path
    review: Path
    encoder: Path | None = None
    ancestor_review: Path | None = None
    ancestor_config: Path | None = None


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _ancestor_paths(inputs):
    if inputs.encoder is None:
        return []
    parent = Path(inputs.encoder).parent
    if inputs.ancestor_review is None:
        raise ValueError("Pretrained transfer requires the original independent ancestor review.")
    return [
        Path(inputs.encoder),
        parent / "run.json",
        parent / "inference.pt",
        parent / "membership.json",
        inputs.ancestor_config or parent / "config.json",
        Path(inputs.ancestor_review),
    ]


def required_paths(inputs, core_config):
    """All project helper/source files and lineage artifacts required in review."""
    paths = [
        Path(__file__),
        inputs.train,
        inputs.dev,
        inputs.train_cohort,
        inputs.dev_cohort,
        inputs.split,
        inputs.adr0016,
        inputs.protocol,
        inputs.config,
        *core.required_sources(core_config),
        *_ancestor_paths(inputs),
    ]
    source = core.HELPER_ROOT
    paths += [
        source / "__init__.py",
        source / "training/__init__.py",
        source / "models/__init__.py",
        source / "data/native_ssl_corpus.py",
        source / "training/aeon_corpus.py",
    ]
    if inputs.review.exists():
        budget = _json(inputs.review).get("budget_resolution_path")
        if budget:
            paths.append(Path(budget))
    return list(dict.fromkeys(Path(p).resolve() for p in paths))


def _distinct_review(review, *, status, implementer):
    if review.get("status") != status:
        raise ValueError(f"Independent {status} review required before fitting.")
    reviewer = review.get("reviewer_session_id")
    author = review.get("implementer_session_id")
    excluded = (author, implementer, review.get("root_coordinator_session_id"))
    if (
        not author
        or not isinstance(reviewer, str)
        or not reviewer
        or reviewer != reviewer.strip()
        or reviewer.casefold() in {str(v).casefold() for v in excluded}
    ):
        raise ValueError("Reviewer must be a distinct session from the implementer.")


def _bindings(record, paths):
    bindings = record.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Required bindings are missing.")
    for path in paths:
        if bindings.get(str(Path(path).resolve())) != sha256(Path(path)):
            raise ValueError(f"Missing or stale binding: {path}")


def _cohort(inputs, role):
    path, payload = (
        (inputs.train_cohort, inputs.train) if role == "train" else (inputs.dev_cohort, inputs.dev)
    )
    record = _json(path)
    if record.get("role") != role:
        raise ValueError("Only train/development cohort roles are admitted; test remains closed.")
    if record.get("npz_sha256") != sha256(payload) or record.get("split_sha256") != sha256(
        inputs.split
    ):
        raise ValueError("Cohort data/split ancestry mismatch.")
    return record


def _split_roles(inputs):
    sources = _json(inputs.split).get("sources", [])
    if not isinstance(sources, list) or any(
        s.get("role") not in ("train", "development", "final_test")
        or not s.get("deployment")
        or not s.get("archive_sha256")
        for s in sources
    ):
        raise ValueError("Invalid declared split source roles.")
    if len({s["deployment"] for s in sources}) != len(sources):
        raise ValueError("Split repeats a source/deployment identity.")
    by_role = {
        r: {s["deployment"]: s["archive_sha256"] for s in sources if s["role"] == r}
        for r in ("train", "development", "final_test")
    }
    if not by_role["train"] or not by_role["development"]:
        raise ValueError("TRAIN/development split membership is required.")
    for a, b in (("train", "development"), ("train", "final_test"), ("development", "final_test")):
        if set(by_role[a]) & set(by_role[b]) or set(by_role[a].values()) & set(by_role[b].values()):
            raise ValueError("Split source/deployment roles overlap.")
    return by_role


def check_prefit(inputs, config, core_config, *, correctness_smoke=False):
    """No NumPy payload or torch checkpoint loading occurs until this succeeds."""
    review = _json(inputs.review)
    expected_kind = (
        "SYNTHETIC_CORRECTNESS_ONLY" if correctness_smoke else "REAL_TRAIN_DEVELOPMENT_FIT"
    )
    if (
        review.get("architecture") != ARCHITECTURE
        or review.get("evidence_kind") != expected_kind
        or set(review.get("allowed_roles", [])) != {"train", "development"}
    ):
        raise ValueError(
            "Downstream review must explicitly admit band architecture/evidence/roles."
        )
    if not correctness_smoke:
        budget_path = review.get("budget_resolution_path")
        if review.get("band_budget_status") != "ROOT_RESOLVED" or not budget_path:
            raise ValueError("Root must resolve the screen/endpoint budget before new prefit.")
        _bindings(review, [Path(budget_path)])
        core._check_budget(review)
    _distinct_review(
        review, status="APPROVED_DOWNSTREAM_PREFIT", implementer=IMPLEMENTER_SESSION_ID
    )
    if review.get("implementer_session_id") != IMPLEMENTER_SESSION_ID:
        raise ValueError("Downstream review does not identify this implementation session.")
    if config.seed not in review.get("allowed_seeds", []):
        raise ValueError("Seed is outside the exact independent replication review.")
    if config.mode not in review.get("allowed_modes", []) or config.method not in review.get(
        "allowed_methods", []
    ):
        raise ValueError("Mode/method is not approved for downstream fitting.")
    _bindings(review, required_paths(inputs, core_config))
    for key, path in (
        ("train_npz_sha256", inputs.train),
        ("dev_npz_sha256", inputs.dev),
        ("split_sha256", inputs.split),
    ):
        if review.get(key) != sha256(path):
            raise ValueError(f"Review identity mismatch: {key}")
    if _json(inputs.config) != config.to_dict():
        raise ValueError("Runtime downstream config differs from reviewed config.")
    if correctness_smoke and review.get("correctness_core_config") != core_config.to_dict():
        raise ValueError("CPU correctness architecture is not explicitly bound in review.")
    _split_roles(inputs)
    _cohort(inputs, "train")
    _cohort(inputs, "development")
    if inputs.encoder is not None:
        paths = _ancestor_paths(inputs)
        parent = _json(paths[1])
        if _parent_core(inputs) is original_core and core_config.seed != 7:
            raise ValueError(
                "Original version1 parent cannot supply seed13/23 replication features."
            )
        parent_review = _json(paths[5])
        _distinct_review(
            parent_review, status="APPROVED_PREFIT", implementer=core.IMPLEMENTER_SESSION_ID
        )
        if config.method not in parent_review.get("allowed_methods", []):
            raise ValueError("Ancestor method was not independently approved.")
        required = [
            inputs.train,
            inputs.dev,
            inputs.split,
            inputs.adr0016,
            paths[4],
            *_parent_core(inputs).required_sources(core_config),
        ]
        for path in required:
            path = Path(path).resolve()
            expected = sha256(path)
            historical = parent.get("bindings", {}).get(str(path))
            if (
                historical != expected
                or parent_review.get("bindings", {}).get(str(path)) != expected
            ):
                raise ValueError(f"Ancestor original source/data binding differs: {path}")
        if (
            parent.get("config") != core_config.to_dict()
            or _json(paths[4]) != core_config.to_dict()
        ):
            raise ValueError("Ancestor config identity mismatch.")
        # Reviews also bind cohort reports, tests and operational evidence; the
        # fit's minimal ancestry map must be contained in that complete review.
        if any(
            parent_review.get("bindings", {}).get(path) != expected
            for path, expected in parent.get("bindings", {}).items()
        ):
            raise ValueError("Ancestor review/run source bindings differ.")
        for key, path in (
            ("train_npz_sha256", inputs.train),
            ("dev_npz_sha256", inputs.dev),
            ("split_sha256", inputs.split),
            ("protocol_sha256", inputs.adr0016),
        ):
            if parent_review.get(key) != sha256(path):
                raise ValueError(f"Ancestor data/protocol identity mismatch: {key}")
        for key, path in (
            ("inference_sha256", paths[2]),
            ("membership_sha256", paths[3]),
            ("review_sha256", paths[5]),
        ):
            if parent.get(key) != sha256(path):
                raise ValueError(f"Ancestor artifact identity mismatch: {key}")
        expected_kind = (
            "SYNTHETIC_CORRECTNESS_ONLY" if correctness_smoke else "REAL_TRAIN_DEVELOPMENT_FIT"
        )
        if (
            parent.get("architecture") != ARCHITECTURE
            or parent_review.get("architecture") != ARCHITECTURE
        ):
            raise ValueError("Ancestor architecture/review differs from band family.")
        if parent.get("status") != "COMPLETED" or parent.get("evidence_kind") != expected_kind:
            raise ValueError("Incomplete or wrong-kind ancestor cannot supply transfer features.")
        membership = _json(paths[3])
        allowed = _split_roles(inputs)["train"]
        deployment, archives = (
            membership.get("train_deployments", []),
            membership.get("train_archive_sha256", []),
        )
        rows = membership.get("train_row_ids", [])
        if (
            not rows
            or len(rows) != len(deployment)
            or len(rows) != len(archives)
            or any(allowed.get(d) != a for d, a in zip(deployment, archives, strict=True))
        ):
            raise ValueError("Ancestor TRAIN source/deployment ancestry mismatch.")
        for name in ("ssl_eligible_indices", "supervised_indices"):
            indices = membership.get(name)
            if (
                not isinstance(indices, list)
                or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError("Ancestor membership indices are invalid.")
        for entry in membership.get("sequence", []):
            for name in ("indices", "context_indices", "target_indices"):
                if name in entry and any(
                    type(i) is not int or not 0 <= i < len(rows) for i in entry[name]
                ):
                    raise ValueError("Ancestor sampled membership is outside TRAIN.")
    return review


def _core_config(inputs, config, *, correctness_smoke, smoke_core_config):
    if config.mode == "direct_end_to_end":
        if (
            inputs.encoder is not None
            or inputs.ancestor_review is not None
            or inputs.ancestor_config is not None
        ):
            raise ValueError("Direct initialization must have no selected ancestor.")
        value = (
            smoke_core_config
            if correctness_smoke and smoke_core_config
            else core.Config(method="direct", seed=config.seed, pretrain_updates=3000)
        )
    else:
        if inputs.encoder is None or Path(inputs.encoder).name != "selected_encoder.pt":
            raise ValueError("Transfer requires an exact selected_encoder.pt artifact.")
        raw = _json(Path(inputs.encoder).parent / "run.json")["config"]
        if raw.get("architecture") != ARCHITECTURE:
            raise ValueError("Selected parent config is not the band family.")
        value = core.Config(**raw)
        if smoke_core_config is not None and value.to_dict() != smoke_core_config.to_dict():
            raise ValueError("Smoke architecture differs from selected ancestor.")
        if value.method != config.method or value.history != config.history:
            raise ValueError("Selected ancestor method/history differs from downstream.")
    if value.seed != config.seed:
        raise ValueError("Selected ancestor and downstream seed must match.")
    if not correctness_smoke:
        if smoke_core_config is not None:
            raise ValueError("Small architectures require explicit correctness_smoke.")
        dimensions = core.model_dimensions(value)
        expected = {"width": 192, "latent": 64, "blocks": 4, "heads": 4}
        if dimensions != expected:
            raise ValueError("Real endpoint backbone dimensions are frozen.")
    value.validate(correctness_smoke=correctness_smoke)
    return value


def _parent_core(inputs):
    """Typed source identity is checked before decoding selected parent tensors."""
    parent = _json(Path(inputs.encoder).parent / "run.json")
    bound = parent.get("bindings", {})
    if str(Path(core.__file__).resolve()) in bound:
        return core
    if str(Path(original_core.__file__).resolve()) in bound:
        return original_core
    raise ValueError("Parent must bind its original or replication SSL runner.")


def _load_ancestor(inputs, core_config):
    if inputs.encoder is None:
        return None
    selected = torch.load(inputs.encoder, map_location="cpu", weights_only=True)
    inference = torch.load(
        Path(inputs.encoder).parent / "inference.pt", map_location="cpu", weights_only=True
    )
    parent = _json(Path(inputs.encoder).parent / "run.json")
    original = _parent_core(inputs) is original_core
    selected_kind = (
        "native_band_ssl_selected_encoder_v1"
        if original
        else "native_band_replication_ssl_selected_encoder_v2"
    )
    inference_kind = (
        "native_band_ssl_weights_only_inference_v1"
        if original
        else "native_band_replication_ssl_weights_only_inference_v2"
    )
    if selected.get("kind") != selected_kind or inference.get("kind") != inference_kind:
        raise ValueError("Unsupported selected ancestor artifact kind.")
    for record in (selected, inference):
        if (
            record.get("architecture") != ARCHITECTURE
            or record.get("config") != core_config.to_dict()
            or record.get("bindings") != parent["bindings"]
        ):
            raise ValueError("Selected artifact source/config lineage mismatch.")
    if selected.get("scalers") != inference.get("scalers") or inference.get(
        "selected_pretrain_step"
    ) != parent.get("selected_pretrain_step"):
        raise ValueError("Selected artifact scaler/checkpoint lineage mismatch.")
    # Safe portable loading also validates every full-model tensor and scaler.
    predictor_class = (
        original_inference.NativeBandAcousticPredictor if original else NativeBandAcousticPredictor
    )
    validate_configuration = original_inference._configuration if original else _configuration
    checked = predictor_class.__new__(predictor_class)
    checked._initialize(inference, "cpu")
    validate_configuration(selected, torch.device("cpu"))
    core.Scalers.from_dict(selected.get("scalers"))
    _check_state(selected.get("encoder"), checked.model.encoder.state_dict())
    encoder = {
        k.removeprefix("encoder."): v
        for k, v in inference["model"].items()
        if k.startswith("encoder.")
    }
    if set(encoder) != set(selected["encoder"]) or any(
        not torch.equal(v, encoder[k]) for k, v in selected["encoder"].items()
    ):
        raise ValueError("Selected encoder differs from parent inference features.")
    return selected


def prepare_model(config, core_config, selected):
    """Fresh heads always use seed+100000, independently of pretraining RNG."""
    model = core.initialize_model(
        config.seed, **core.model_dimensions(core_config), method=config.method
    )
    if selected is not None:
        model.encoder.load_state_dict(selected["encoder"], strict=True)
    model.requires_grad_(False)
    model.readout.requires_grad_(True)
    if config.mode != "frozen_readout":
        model.encoder.requires_grad_(True)
    model.encoder.eval()
    return model


def supervised_indices(pool, config, step):
    return core.batch_indices(pool, config.batch_size, config.seed, "readout", step)


def _forecast_train(model, batch, mode):
    # Only the selected shared band features receive separate supervised gradients.
    # Frozen mode uses eval/no_grad; no EMA update or SSL loss occurs here.
    model.readout.train()
    if mode == "frozen_readout":
        model.encoder.eval()
        with torch.no_grad():
            features = model.encoder.encode(batch["x"], batch["observed"], batch["metadata"])
    else:
        model.encoder.train()
        features = model.encoder.encode(batch["x"], batch["observed"], batch["metadata"])
    return model.readout(features, batch["query"]).sort(dim=-1).values


def _verify_data(inputs, train, dev, selected):
    for role, data in (("train", train), ("development", dev)):
        core.validate_membership(data, inputs.split, role=role)
        cohort = _cohort(inputs, role)
        reports = cohort.get("source_reports", [])
        counts = {
            str(d): int((data["deployment"] == d).sum()) for d in np.unique(data["deployment"])
        }
        if (
            cohort.get("issued") != len(data["x"])
            or {r["deployment"]: r["issued"] for r in reports if r["issued"]} != counts
        ):
            raise ValueError("Cohort issued/source counts mismatch.")
    if set(train["row_id"].tolist()) & set(dev["row_id"].tolist()):
        raise ValueError("TRAIN/development row membership overlaps.")
    fitted = core.Scalers.fit(train, split_path=inputs.split)
    if selected is None:
        return fitted
    membership = _json(Path(inputs.encoder).parent / "membership.json")
    for name, key in (
        ("train_row_ids", "row_id"),
        ("train_deployments", "deployment"),
        ("train_archive_sha256", "archive_sha256"),
    ):
        if membership[name] != train[key].tolist():
            raise ValueError("Ancestor exact TRAIN row/source membership mismatch.")
    for name, indices in (
        ("ssl_eligible_indices", np.flatnonzero(train["ssl_eligible"])),
        ("supervised_indices", np.flatnonzero(train["y_observed"].any(1))),
    ):
        if membership[name] != indices.tolist():
            raise ValueError("Ancestor TRAIN eligibility membership mismatch.")
    # Recompute TRAIN statistics only to verify lineage, never replace ancestor
    # scalers and never fit development. Comparison is exact float32/list state.
    if selected["scalers"] != fitted.to_dict():
        raise ValueError("Ancestor scalers are not fitted exclusively to this TRAIN.")
    return core.Scalers.from_dict(selected["scalers"])


def _safe_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _safe_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_safe_cpu(v) for v in value)
    return copy.deepcopy(value)


def _verify_continuation(saved, config, pool, train, scalers):
    state = saved["state"]
    if state["scalers"] != scalers.to_dict():
        raise ValueError("Resume scaler identity differs from TRAIN lineage.")
    step = state.get("step")
    if type(step) is not int or not 0 <= step <= config.updates or len(state["sequence"]) != step:
        raise ValueError("Resume sampler counters differ from supervised trajectory.")
    if saved["scheduler"].get("last_epoch") != step:
        raise ValueError("Resume schedule differs from supervised counters.")
    for index, entry in enumerate(state["sequence"]):
        ids = supervised_indices(pool, config, index)
        if entry != {
            "step": index,
            "indices": ids.tolist(),
            "row_ids": train["row_id"][ids].tolist(),
            "sha256": core.sequence_hash(ids),
        }:
            raise ValueError(
                "Resume sampler sequence differs from independent seed-controlled TRAIN batches."
            )
    records = state["records"]
    expected = [s for s in range(1, step + 1) if s % config.cadence == 0 or s == config.updates]
    if [r["step"] for r in records] != expected:
        raise ValueError("Resume checkpoint selection cadence differs.")
    best_score, best_step, bad_checks = None, None, 0
    for record in records:
        score = record["daily_dev_pinball"]
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError("Resume selection score is invalid.")
        improved = best_score is None or score < best_score
        if record["selected"] != improved:
            raise ValueError("Resume checkpoint selection record differs.")
        if improved:
            best_score, best_step, bad_checks = score, record["step"], 0
        else:
            bad_checks += 1
    if (state["best_score"], state["best_step"], state["bad_checks"], state["stopped"]) != (
        best_score,
        best_step,
        bad_checks,
        bad_checks >= config.patience,
    ) or (state["best_model"] is None) != (best_step is None):
        raise ValueError("Resume best checkpoint/stopping counters differ.")


def run(
    inputs,
    config,
    *,
    output,
    device="cpu",
    resume=None,
    correctness_smoke=False,
    smoke_core_config=None,
    stop_after=None,
):
    config.validate(correctness_smoke=correctness_smoke)
    device = torch.device(device)
    if not correctness_smoke and str(device) != "cuda:0":
        raise ValueError("Scientific replication requires explicitly indexed cuda:0.")
    if device.type not in ("cpu", "cuda") or (correctness_smoke and device.type != "cpu"):
        raise ValueError("Correctness smoke is CPU only; endpoints support local CPU/CUDA only.")
    if stop_after is not None and (not correctness_smoke or stop_after < 1):
        raise ValueError("Artificial interruption is reserved for CPU correctness checks.")
    ancestor_config = _core_config(
        inputs, config, correctness_smoke=correctness_smoke, smoke_core_config=smoke_core_config
    )
    review = check_prefit(inputs, config, ancestor_config, correctness_smoke=correctness_smoke)
    identities = {
        "bindings": {str(p): sha256(p) for p in required_paths(inputs, ancestor_config)},
        "review_sha256": sha256(inputs.review),
        "config": config.to_dict(),
        "core_config": ancestor_config.to_dict(),
        "device_kind": device.type,
        "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY"
        if correctness_smoke
        else "REAL_TRAIN_DEVELOPMENT_FIT",
    }
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            saved.get("kind") != "native_band_replication_downstream_resume_v2"
            or saved.get("architecture") != ARCHITECTURE
            or saved.get("identities") != identities
        ):
            raise ValueError("Resume source/config/review/ancestor/device identities differ.")
    output = Path(output).resolve()
    if output.exists():
        if (
            resume is None
            or Path(resume).resolve().parent != output
            or Path(resume).name != "latest.pt"
        ):
            raise ValueError("Existing outputs require explicit same-run latest.pt resume.")
        if (output / "run.json").exists():
            raise ValueError("Completed outputs cannot be overwritten or silently extended.")
    # Refuse a nonempty directory even with filesystem transports that do not
    # represent directories separately; normal output directories are created below.
    try:
        nonempty = bool(list(output.iterdir()))
    except FileNotFoundError:
        nonempty = False
    if nonempty and (resume is None or Path(resume).resolve().parent != output):
        raise ValueError("New output path must be empty; no silent overwrite.")
    selected = _load_ancestor(inputs, ancestor_config)
    train = core.load_corpus(
        inputs.train,
        role="train",
        split_hash=sha256(inputs.split),
        correctness_smoke=correctness_smoke,
    )
    dev = core.load_corpus(
        inputs.dev,
        role="development",
        split_hash=sha256(inputs.split),
        correctness_smoke=correctness_smoke,
    )
    scalers = _verify_data(inputs, train, dev, selected)
    pool = np.flatnonzero(train["y_observed"].any(1))
    if not len(pool):
        raise ValueError("No observed TRAIN supervised rows.")
    if saved is not None:
        _verify_continuation(saved, config, pool, train, scalers)
    inference_config = replace(
        ancestor_config,
        method=config.method,
        seed=config.seed,
        history=config.history,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = prepare_model(config, ancestor_config, selected)
    frozen_state = core.cpu_state(model.encoder) if config.mode == "frozen_readout" else None
    if saved is not None:
        _check_state(saved["model"], model.state_dict())
        model.load_state_dict(saved["model"], strict=True)
        if frozen_state is not None:
            for candidate in (saved["model"], saved["state"]["best_model"]):
                if candidate is not None and any(
                    not torch.equal(v, candidate["encoder." + k]) for k, v in frozen_state.items()
                ):
                    raise ValueError("Frozen resume encoder differs from selected ancestor.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    state = {
        "step": 0,
        "best_step": None,
        "best_score": None,
        "best_model": None,
        "bad_checks": 0,
        "sequence": [],
        "records": [],
        "scalers": scalers.to_dict(),
        "stopped": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with core.Resources(device, output) as resources:
        model.to(device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = core.schedule(optimizer, config.updates)
        if saved is not None:
            state = saved["state"]
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            for item in optimizer.state.values():
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        item[key] = value.to(device)
            core.restore_rng(saved["rng"])
            resources.elapsed = saved["resources"]["elapsed_seconds"]
            resources.peak_rss = saved["resources"]["peak_rss_bytes"]
            resources.peak_allocated = saved["resources"]["peak_allocated_bytes"]
            resources.peak_reserved = saved["resources"]["peak_reserved_bytes"]

        def save(name="latest.pt"):
            payload = {
                "kind": "native_band_replication_downstream_resume_v2",
                "architecture": ARCHITECTURE,
                "correctness_smoke": correctness_smoke,
                "identities": identities,
                "state": _safe_cpu(state),
                "model": core.cpu_state(model),
                "optimizer": _safe_cpu(optimizer.state_dict()),
                "scheduler": scheduler.state_dict(),
                "rng": core.rng_state(),
                "resources": resources.snapshot(),
            }
            core.atomic_checkpoint(output / name, payload)
            if name != "latest.pt":
                core.atomic_checkpoint(output / "latest.pt", payload)

        while state["step"] < config.updates and not state["stopped"]:
            ids = supervised_indices(pool, config, state["step"])
            batch = core.tensor_batch(train, ids, scalers, config.history, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = _forecast_train(model, batch, config.mode)
            loss = core.pinball(predictions, batch["y"], batch["y_observed"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite downstream loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            state["sequence"].append(
                {
                    "step": state["step"],
                    "indices": ids.tolist(),
                    "row_ids": train["row_id"][ids].tolist(),
                    "sha256": core.sequence_hash(ids),
                }
            )
            state["step"] += 1
            resources.check()
            if frozen_state is not None and any(
                not torch.equal(v, model.encoder.state_dict()[k].detach().cpu())
                for k, v in frozen_state.items()
            ):
                raise RuntimeError("Frozen encoder parameters/buffers changed.")
            if state["step"] % config.cadence == 0 or state["step"] == config.updates:
                values = core.predict(model, dev, scalers, inference_config, device)
                metrics = core.daily_metrics(
                    values,
                    dev["y"],
                    dev["y_observed"],
                    dev["target_dates"],
                    dev["deployment"],
                    min_anchors=config.min_daily_anchors,
                )
                score = metrics["primary_pinball"]
                if score is None:
                    raise ValueError(
                        "Development has insufficient shared native daily support for selection."
                    )
                improved = state["best_score"] is None or score < state["best_score"]
                if improved:
                    state.update(
                        best_score=score,
                        best_step=state["step"],
                        best_model=core.cpu_state(model),
                        bad_checks=0,
                    )
                else:
                    state["bad_checks"] += 1
                state["records"].append(
                    {"step": state["step"], "daily_dev_pinball": score, "selected": improved}
                )
                state["stopped"] = state["bad_checks"] >= config.patience
                save(f"supervised-{state['step']:06d}.pt")
            if stop_after is not None and state["step"] >= stop_after:
                save()
                return {
                    "status": "INTERRUPTED_CORRECTNESS_CHECK",
                    "supervised_updates": state["step"],
                }
        if state["best_model"] is None:
            raise ValueError("No selected downstream checkpoint.")
        save("complete.pt")
        model.load_state_dict(state["best_model"], strict=True)
        model.eval()
        model.requires_grad_(False)
        bindings = identities["bindings"]
        ancestry = {
            "mode": config.mode,
            "ssl_only": False,
            "supervised_updates": state["step"],
            "selected_supervised_step": state["best_step"],
            "ancestor_encoder_sha256": sha256(inputs.encoder) if inputs.encoder else None,
            "ancestor_run_sha256": sha256(Path(inputs.encoder).parent / "run.json")
            if inputs.encoder
            else None,
        }
        inference = {
            "kind": "native_band_replication_ssl_weights_only_inference_v2",
            "architecture": ARCHITECTURE,
            "correctness_smoke": correctness_smoke,
            "model": core.cpu_state(model),
            "config": inference_config.to_dict(),
            "scalers": scalers.to_dict(),
            "bindings": bindings,
            "downstream_config": config.to_dict(),
            "supervised_ancestry": ancestry,
            "review_sha256": identities["review_sha256"],
            "evidence_kind": identities["evidence_kind"],
        }
        core.atomic_checkpoint(output / "inference.pt", inference)
        if config.mode == "frozen_readout":
            # Byte identity matters: reserializing the same tensors is insufficient.
            with (output / "selected_encoder.pt").open("xb") as stream:
                stream.write(Path(inputs.encoder).read_bytes())
        else:
            core.atomic_checkpoint(
                output / "selected_encoder.pt",
                {
                    "kind": "native_band_replication_downstream_supervised_encoder_v2",
                    "architecture": ARCHITECTURE,
                    "correctness_smoke": correctness_smoke,
                    "encoder": core.cpu_state(model.encoder),
                    "config": inference_config.to_dict(),
                    "scalers": scalers.to_dict(),
                    "bindings": bindings,
                    "supervised_ancestry": ancestry,
                    "evidence_kind": identities["evidence_kind"],
                },
            )
        values = core.predict(model, dev, scalers, inference_config, device)
        np.savez_compressed(
            output / "predictions.npz",
            predictions=values,
            targets=dev["y"],
            target_observed=dev["y_observed"],
            target_dates=dev["target_dates"],
            row_id=dev["row_id"],
            deployment=dev["deployment"],
            archive_sha256=dev["archive_sha256"],
            cutoff=dev["cutoff"],
            query=dev["query"],
            metadata=dev["metadata"],
            context_observed=dev["observed"],
            assessment_support=np.ones_like(dev["y_observed"]),
            query_native_bounds_m=dev["query"][:, :, 3:5] * 250,
            query_frequency_hz=dev["query"][:, :, 0] * 455000,
            query_interval_seconds=dev["query"][:, :, 1] * 3600,
            quantiles=np.asarray(core.QUANTILES),
            horizons=np.asarray([1, 3, 6]),
        )
        metrics = core.daily_metrics(
            values,
            dev["y"],
            dev["y_observed"],
            dev["target_dates"],
            dev["deployment"],
            min_anchors=config.min_daily_anchors,
        )
        _write_json(output / "metrics.json", metrics)
        sequence = state["sequence"]
        membership = {
            "train_row_ids": train["row_id"].tolist(),
            "train_deployments": train["deployment"].tolist(),
            "train_archive_sha256": train["archive_sha256"].tolist(),
            "supervised_indices": pool.tolist(),
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(
                json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        _write_json(output / "membership.json", membership)
        report = {
            "architecture": ARCHITECTURE,
            "correctness_smoke": correctness_smoke,
            "status": "COMPLETED",
            "evidence_kind": identities["evidence_kind"],
            "mode": config.mode,
            "config": config.to_dict(),
            "core_config": inference_config.to_dict(),
            "supervised_updates": state["step"],
            "optimizer_steps_total": state["step"],
            "selected_supervised_step": state["best_step"],
            "label_observations_processed": int(
                sum(train["y_observed"][entry["indices"]].sum() for entry in sequence)
            ),
            "sample_presentations": sum(len(entry["indices"]) for entry in sequence),
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_optimized": sum(p.numel() for p in parameters),
            "parameters_encoder": sum(p.numel() for p in model.encoder.parameters()),
            "parameters_head": sum(p.numel() for p in model.readout.parameters()),
            "resources_additional_downstream": resources.snapshot(),
            "selection_records": state["records"],
            "selected_daily_dev_pinball": state["best_score"],
            "bindings": bindings,
            "review_sha256": identities["review_sha256"],
            "reviewer_session_id": review["reviewer_session_id"],
            "membership_sha256": sha256(output / "membership.json"),
            "inference_sha256": sha256(output / "inference.pt"),
            "selected_encoder_sha256": sha256(output / "selected_encoder.pt"),
            "supervised_ancestry": ancestry,
            "initialization": "fresh_seed_controlled_head_and_selected_features"
            if selected
            else "matched_seed_controlled_random_model_and_head",
            "selection_policy": config.selection_policy,
            "stopping_patience": config.patience,
            "test_access": "NOT_RUN",
            "scientific_claim": "NOT_ESTABLISHED",
        }
        _write_json(output / "run.json", report)
        return report


class Inference(NativeBandAcousticPredictor):
    """Safe, frozen target-free endpoint loader without fit initialization."""

    def __init__(self, artifact, device):
        self._initialize(artifact, device)


def load_inference(path, *, device="cpu"):
    return Inference(torch.load(path, map_location="cpu", weights_only=True), device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "train",
        "dev",
        "train-cohort",
        "dev-cohort",
        "split",
        "adr0016",
        "protocol",
        "config",
        "review",
        "output",
    ):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    for flag in ("encoder", "ancestor-review", "ancestor-config", "resume"):
        parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args(argv)
    config = DownstreamConfig(**_json(args.config))
    inputs = RunInputs(**{name: getattr(args, name) for name in RunInputs.__dataclass_fields__})
    report = run(inputs, config, output=args.output, device=args.device, resume=args.resume)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "mode", "supervised_updates", "inference_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
