"""Load-only access to native SSL embeddings and trained forward latent heads.

Shared predictions represent four-interval future blocks at source offsets1/3/6.
CF predicts three ordinal future zones from ONLINE sequence features, while
encode returns SELECTED EMA pooled features. Full H96 CF inference is outside
its sampled crop-view support. Neither output is acoustic dB or a depth profile.
Ancestral paths are copied metadata, never opened, imported or executed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import fields

import numpy as np
import torch
from torch import nn

from marine_echo.inference import native_encoder as legacy
from marine_echo.inference.native_band_replication_acoustic import _check_state, _query
from marine_echo.models.native_band_temporal import ARCHITECTURE, NativeBandTemporalModel
from marine_echo.models.native_temporal import (
    CFTemporalEncoder,
    NativeTemporalModel,
    QueryHead,
)
from marine_echo.training.native_band_replication_ssl import Config as BandConfig
from marine_echo.training.native_ssl import Config

LEGACY_KIND = "native_ssl_weights_only_inference_v1"
BAND_KIND = "native_band_replication_ssl_weights_only_inference_v2"
LATENT_METHODS = {"shared_ssl", "permuted_ssl", "cf_jepa"}


class _LoadOnlyCF(nn.Module):
    """Exact CF state layout without stochastic predictor re-initialization.

    No CF training initializer, loss, EMA update or optimizer is constructed.
    All modules are empty meta templates until strict tensor assignment.
    """

    def __init__(self, config):
        super().__init__()
        dims = (config.cf_width, config.cf_latent, config.cf_blocks)
        self.online = CFTemporalEncoder(*dims)
        self.encoder = CFTemporalEncoder(*dims)
        self.readout = QueryHead(config.cf_latent, 5, config.cf_latent)
        self.predictors = nn.ModuleList(
            [nn.Linear(config.cf_latent, config.cf_latent) for _ in range(3)]
        )


def _configuration(artifact, device, band):
    raw = artifact.get("config")
    cls = BandConfig if band else Config
    expected = {f.name: f.default for f in fields(cls)}
    if not isinstance(raw, dict) or set(raw) != set(expected):
        raise ValueError("Complete exact saved Config fields required; no inferred defaults.")
    for key, default in expected.items():
        value = raw[key]
        if type(default) is int:
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid saved integer config: {key}")
        elif type(default) is float:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Nonfinite/incompatible saved numeric config: {key}")
            if value < 0:
                raise ValueError(f"Negative saved numeric config: {key}")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"Invalid saved string config: {key}")
    if raw["method"] not in legacy.TRAINING_KINDS or (band and raw["method"] == "cf_jepa"):
        raise ValueError("Unknown/incompatible method for the exact artifact family.")
    if band:
        if raw["architecture"] != ARCHITECTURE or artifact.get("architecture") != ARCHITECTURE:
            raise ValueError("Exact separate band architecture required.")
    elif "architecture" in artifact:
        raise ValueError("Legacy artifacts do not declare or reinterpret band architecture.")
    if raw["history"] != 96 or not 1 <= raw["batch_size"] <= 256:
        raise ValueError("Latent inference requires saved H96 and bounded batch size.")
    for key in ("width", "latent", "cf_width", "cf_latent"):
        if not 1 <= raw[key] <= 512:
            raise ValueError("Saved width/latent dimensions are missing or unbounded.")
    for key in ("blocks", "cf_blocks", "heads"):
        if not 1 <= raw[key] <= 16:
            raise ValueError("Saved block/head dimensions are missing or unbounded.")
    if raw["width"] % raw["heads"] or raw["lr"] <= 0 or raw["cf_lr"] <= 0:
        raise ValueError("Incompatible saved dimensions/numeric configuration.")
    evidence = artifact.get("evidence_kind")
    smoke = evidence == "SYNTHETIC_CORRECTNESS_ONLY"
    if smoke:
        if artifact.get("correctness_smoke") is not True or device.type != "cpu":
            raise ValueError("Tiny fixtures require explicitly synthetic CPU correctness identity.")
    elif evidence not in (None, "REAL_TRAIN_DEVELOPMENT_FIT") or artifact.get(
        "correctness_smoke"
    ) not in (None, False):
        raise ValueError("Unknown/inconsistent artifact evidence kind.")
    if band and evidence is None:
        raise ValueError("Band artifact evidence identity is required.")
    if not smoke:
        if raw["method"] == "cf_jepa":
            if (raw["cf_width"], raw["cf_latent"], raw["cf_blocks"]) != (256, 128, 5):
                raise ValueError("Real CF encoder dimensions must match the native contract.")
        elif (raw["width"], raw["latent"], raw["blocks"], raw["heads"]) not in (
            (192, 64, 4, 4),
            (192, 128, 4, 4),
        ):
            raise ValueError("Tiny dimensions require SYNTHETIC_CORRECTNESS_ONLY.")
        if band and (raw["seed"] not in (7, 13, 23) or raw["latent"] != 64):
            raise ValueError("Band scientific architecture is separate and frozen.")
    # Structural inference validation only, never Config.validate/fit policy.
    return cls(**raw) if band else legacy._config(raw)


def _artifact(artifact, device):
    if not isinstance(artifact, Mapping) or artifact.get("kind") != BAND_KIND:
        raise ValueError("Only original/band SSL weights-only inference artifacts are supported.")
    allowed = {
        "kind",
        "model",
        "config",
        "scalers",
        "bindings",
        "selected_pretrain_step",
        "evidence_kind",
        "correctness_smoke",
    }
    band = artifact["kind"] == BAND_KIND
    if band:
        allowed.add("architecture")
    if not set(artifact).issubset(allowed):
        raise ValueError("Unsafe/unknown fields or downstream supervised ancestry are unsupported.")
    config = _configuration(artifact, device, band)
    if config.seed not in (7, 13, 23):
        raise ValueError("Band replication seed must be exactly7/13/23.")
    step = artifact.get("selected_pretrain_step")
    if type(step) is not int or step < 0 or step > config.pretrain_updates:
        raise ValueError("Exact selected pretrain step is missing or incompatible.")
    if config.method in LATENT_METHODS and step == 0:
        raise ValueError("A trained forward objective requires a positive selected pretrain step.")
    return config, band, step


class NativeLatentPredictor:
    """Frozen context-only SSL inference; no forecast, targets or rollout API.

    Inputs cannot verify source-clock continuity: issuance dates/interval IDs
    remain the caller's responsibility. Device must be exactly cpu or cuda:0.
    Existing absolute provenance paths may be inaccessible after relocation.
    """

    def __init__(self, weights, device="cpu"):
        if str(device) not in ("cpu", "cuda:0"):
            raise ValueError("Device must be explicitly cpu or cuda:0.")
        self.device = torch.device(device)
        artifact = torch.load(weights, weights_only=True, map_location="cpu")
        self._config, band, self.selected_pretrain_step = _artifact(artifact, self.device)
        self._scalers = legacy._scalers(artifact.get("scalers"))
        self._bindings = legacy._bindings(artifact.get("bindings"))
        self._config_metadata = copy.deepcopy(artifact["config"])
        self._scaler_metadata = copy.deepcopy(artifact["scalers"])
        self.artifact_kind = artifact["kind"]
        self.architecture = ARCHITECTURE if band else None
        self.method = self._config.method
        self.training_kind = legacy.TRAINING_KINDS[self.method]
        self.feature_branch = (
            "selected_ema"
            if self.method == "cf_jepa"
            else ("shared_band_encoder" if band else "shared_encoder")
        )
        self.embedding_dimension = (
            self._config.cf_latent if self.method == "cf_jepa" else self._config.latent
        )
        with torch.device("meta"):
            if self.method == "cf_jepa":
                model = _LoadOnlyCF(self._config).float()
            else:
                model = (NativeBandTemporalModel if band else NativeTemporalModel)(
                    self._config.width, self._config.latent, self._config.blocks, self._config.heads
                ).float()
        state = artifact.get("model")
        _check_state(state, model.state_dict())
        for key, value in state.items():
            if (
                key.endswith(("running_var", "num_batches_tracked"))
                or key == "regularizer.global_step"
            ) and (value < 0).any():
                raise ValueError(f"Invalid nonnegative safe buffer: {key}")
        model.load_state_dict(state, strict=True, assign=True)
        self._model = model.eval().requires_grad_(False).to(self.device)

    @property
    def config_metadata(self):
        return copy.deepcopy(self._config_metadata)

    @property
    def scaler_metadata(self):
        return copy.deepcopy(self._scaler_metadata)

    @property
    def source_bindings(self):
        return copy.deepcopy(self._bindings)

    @property
    def objective_metadata(self):
        common = {
            "method": self.method,
            "training_kind": self.training_kind,
            "feature_branch": self.feature_branch,
            "architecture": self.architecture,
            "units": "latent coordinates; not acoustic dB, profiles or biological quantities",
        }
        if self.method == "cf_jepa":
            common.update(
                objective="cf_ordinal_future_zone",
                predictor_branch="online_sequence",
                target_branch="ema_teacher_sequence",
                axes=["sample", "context_token", "ordinal_zone", "latent"],
                ordinal_zones=[1, 2, 3],
                input_view="Full H96 context is outside sampled forward training crop-view support.",
                training_comparison="Normalized directions under source-defined loss; API returns raw linear predictions.",
            )
        elif self.method in LATENT_METHODS:
            common.update(
                objective="shared_future_block"
                if self.method == "shared_ssl"
                else "permuted_pairing_control",
                predictor_branch="shared_encoder_then_trained_query_head",
                target_branch="shared_encoder_not_detached",
                horizons=[1, 3, 6],
                horizon_units="native source interval offsets; not verified UTC",
                target_block_length=4,
                target_blocks_overlap=True,
                axes=["sample", "source_offset", "latent"],
            )
        else:
            common.update(objective="untrained_forward_head", latent_prediction_supported=False)
        return copy.deepcopy(common)

    def _batches(self, x, observed, metadata, query=None):
        x, observed, metadata = legacy._inputs(x, observed, metadata, self._config.history)
        if query is not None:
            query = _query(query, metadata)
        self._model.eval()
        size = min(self._config.batch_size, 64)
        for start in range(0, len(x), size):
            end = min(start + size, len(x))
            with np.errstate(over="ignore", invalid="ignore"):
                values = self._scalers.channels(x[start:end], observed[start:end])
            if not np.isfinite(values).all():
                raise ValueError("Observed normalized dB must be finite at float32 precision.")
            tensors = [
                torch.as_tensor(values, device=self.device),
                torch.as_tensor(observed[start:end].copy(order="C"), device=self.device),
                torch.as_tensor(metadata[start:end].copy(order="C"), device=self.device),
            ]
            if query is not None:
                tensors.append(
                    torch.as_tensor(query[start:end].copy(order="C"), device=self.device)
                )
            yield end - start, tensors

    def _output(self, tensor, shape):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Latent output must have its exact finite float32 shape.")
        return tensor.cpu().numpy()

    @torch.inference_mode()
    def encode(self, x, observed, metadata):
        """Selected shared/EMA pooled embedding[N,D], with saved TRAIN scalers."""
        results = []
        for count, batch in self._batches(x, observed, metadata):
            result = self._model.encoder.encode(*batch)
            results.append(self._output(result, (count, self.embedding_dimension)))
        return np.concatenate(results)

    @torch.inference_mode()
    def predict_latents(self, x, observed, metadata, query):
        """Shared native-offset target embeddings[N,3,D]; never acoustic forecasts."""
        if self.method == "cf_jepa":
            raise ValueError(
                "CF learns ordinal future zones, not queried horizons1/3/6; use predict_cf_zones."
            )
        if self.method not in LATENT_METHODS:
            raise ValueError(f"{self.method} has an untrained forward latent head; encode only.")
        # None must not silently remove the issued query validation path.
        if query is None:
            raise ValueError("An issued native query is required for shared latent prediction.")
        results = []
        for count, batch in self._batches(x, observed, metadata, query):
            context = self._model.encoder.encode(*batch[:3])
            result = self._model.predictor(context, batch[3])
            results.append(self._output(result, (count, 3, self.embedding_dimension)))
        return np.concatenate(results)

    @torch.inference_mode()
    def predict_cf_zones(self, x, observed, metadata):
        """Raw ONLINE ordinal-zone predictions[N,96,3,D], no issued query.

        Full observed H96 is outside the randomly cropped CF training view
        distribution. Zones have no fixed1/3/6 offset or acoustic forecast claim.
        """
        if self.method != "cf_jepa":
            raise ValueError("Only CF has ordinal future-zone sequence predictors.")
        results = []
        for count, batch in self._batches(x, observed, metadata):
            sequence = self._model.online.sequence(*batch)
            result = torch.stack([p(sequence) for p in self._model.predictors], dim=2)
            results.append(self._output(result, (count, 96, 3, self.embedding_dimension)))
        return np.concatenate(results)
