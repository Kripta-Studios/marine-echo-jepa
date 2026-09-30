"""Safe target-free inference for the separate nonlinear band family.

Frozen source ancestry is exposed, never opened or executed. Meta constructors
consume no CPU RNG and initialize no optimizer, determinism policy or disk cache.
Native inputs retain their actual sensing geometry; embeddings are not profiles.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from marine_echo.inference import native_encoder as legacy
from marine_echo.models.native_band_temporal import (
    ARCHITECTURE,
    NativeBandEncoder,
    NativeBandTemporalModel,
)
from marine_echo.training.native_band_ssl import METHODS, Config

TRAINING_KINDS = {key: value for key, value in legacy.TRAINING_KINDS.items() if key in METHODS}
SELECTED_KIND = "native_band_ssl_selected_encoder_v1"
INFERENCE_KIND = "native_band_ssl_weights_only_inference_v1"


def _configuration(artifact, device):
    raw = artifact.get("config")
    if (
        artifact.get("architecture") != ARCHITECTURE
        or not isinstance(raw, dict)
        or raw.get("architecture") != ARCHITECTURE
        or raw.get("method") not in METHODS
    ):
        raise ValueError("Exact band architecture/config required; legacy and CF are unsupported.")
    for key in ("seed", "history", "batch_size", "width", "latent", "blocks", "heads"):
        if type(raw.get(key)) is not int or raw[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Explicit compatible saved dimension/config required: {key}")
    try:
        config = Config(**raw)
    except TypeError as error:
        raise ValueError("Incompatible band Config fields.") from error
    if config.history != 96 or config.width % config.heads:
        raise ValueError("Band inference requires H96 and valid attention dimensions.")
    smoke = artifact.get("evidence_kind") == "SYNTHETIC_CORRECTNESS_ONLY"
    if smoke and (device.type != "cpu" or artifact.get("correctness_smoke") is not True):
        raise ValueError("Tiny correctness artifacts require explicit synthetic CPU identity.")
    if not smoke and artifact.get("evidence_kind") != "REAL_TRAIN_DEVELOPMENT_FIT":
        raise ValueError("Saved band evidence kind is missing or incompatible.")
    if not smoke and (config.seed, config.width, config.latent, config.blocks, config.heads) != (
        7,
        192,
        64,
        4,
        4,
    ):
        raise ValueError("Scientific band dimensions/seed are frozen.")
    config.validate(correctness_smoke=smoke)
    return config


def _check_state(state, expected):
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Safe state must contain exactly the required tensors/buffers.")
    for key, template in expected.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.shape != template.shape
            or value.dtype != template.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Incompatible or nonfinite safe tensor: {key}")


def _base(instance, artifact, device):
    if not isinstance(artifact, Mapping):
        raise TypeError("Expected a safe band artifact mapping.")
    instance.device = torch.device(device)
    if instance.device.type not in ("cpu", "cuda"):
        raise ValueError("Only local CPU/CUDA devices are supported.")
    instance.config = _configuration(artifact, instance.device)
    instance.scalers = legacy._scalers(artifact.get("scalers"))
    instance._bindings = legacy._bindings(artifact.get("bindings"))
    instance._config_metadata = copy.deepcopy(artifact["config"])
    instance._scaler_metadata = copy.deepcopy(artifact["scalers"])
    instance.method = instance.config.method
    instance.architecture = ARCHITECTURE
    instance.training_kind = TRAINING_KINDS[instance.method]
    instance.embedding_dimension = instance.config.latent


class NativeBandAcousticEncoder(legacy.NativeAcousticEncoder):
    """Selected shared-band encoder only; no query, readout, targets or future.

    Separately supervised selected encoders are explicitly unsupported here.
    Issuance/source-clock association remains the caller's responsibility.
    """

    def __init__(self, weights, device="cpu"):
        artifact = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(artifact, Mapping) or artifact.get("kind") != SELECTED_KIND:
            raise ValueError("Expected native_band_ssl_selected_encoder_v1 only.")
        _base(self, artifact, device)
        self.feature_branch = "shared_band_encoder"
        self.selected_pretrain_step = artifact.get("selected_pretrain_step")
        if self.selected_pretrain_step is not None and (
            type(self.selected_pretrain_step) is not int or self.selected_pretrain_step < 0
        ):
            raise ValueError("Optional selected step must be a nonnegative integer.")
        with torch.device("meta"):
            encoder = NativeBandEncoder(
                self.config.width, self.config.latent, self.config.blocks, self.config.heads
            ).float()
        _check_state(artifact.get("encoder"), encoder.state_dict())
        encoder.load_state_dict(artifact["encoder"], strict=True, assign=True)
        self.encoder = encoder.eval().requires_grad_(False).to(self.device)


def _query(query, metadata):
    query = np.asarray(query)
    if (
        query.shape != (len(metadata), 3, 10)
        or query.dtype.kind != "f"
        or not np.isfinite(query).all()
        or not np.allclose(query[..., :9], metadata[:, :1, :9], rtol=0, atol=1e-6)
        or not np.allclose(query[..., 9], [1, 3, 6], rtol=0, atol=1e-6)
    ):
        raise ValueError("Issued query must retain primary native fields and horizons1/3/6.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = query.astype(np.float32, copy=False)
    if not np.isfinite(result).all() or (result[..., 4] <= result[..., 3]).any():
        raise ValueError("Issued native query must remain finite/valid at model precision.")
    return result


def _ancestry(artifact, config):
    ancestry = artifact.get("supervised_ancestry")
    if ancestry is None:
        if "downstream_config" in artifact:
            raise ValueError("Downstream forecast requires explicit supervised ancestry.")
        return None
    from marine_echo.training.native_band_downstream import DownstreamConfig

    if not isinstance(ancestry, dict) or not isinstance(artifact.get("downstream_config"), dict):
        raise TypeError("Complete supervised ancestry/config is required.")
    try:
        downstream = DownstreamConfig(**artifact["downstream_config"])
    except TypeError as error:
        raise ValueError("Incompatible downstream config.") from error
    downstream.validate(correctness_smoke=artifact.get("correctness_smoke") is True)
    for key in ("method", "seed", "history", "batch_size", "lr", "weight_decay", "architecture"):
        if getattr(downstream, key) != getattr(config, key):
            raise ValueError("Downstream inference config differs from its supervised ancestry.")
    steps, selected = ancestry.get("supervised_updates"), ancestry.get("selected_supervised_step")
    if (
        ancestry.get("mode") != downstream.mode
        or ancestry.get("ssl_only") is not False
        or type(steps) is not int
        or not 1 <= steps <= downstream.updates
        or type(selected) is not int
        or not 1 <= selected <= steps
    ):
        raise ValueError("Invalid supervised update/selection ancestry.")
    for key in ("ancestor_encoder_sha256", "ancestor_run_sha256"):
        value = ancestry.get(key)
        if downstream.mode == "direct_end_to_end":
            if value is not None:
                raise ValueError("Direct endpoint must have no selected ancestor.")
        elif (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Transferred inference requires exact ancestor identities.")
    return copy.deepcopy(ancestry)


class NativeBandAcousticPredictor:
    """Safe band forecast plus reusable encoding, including downstream forecasts."""

    def __init__(self, weights, *, device="cpu"):
        self._initialize(torch.load(weights, map_location="cpu", weights_only=True), device)

    def _initialize(self, artifact, device):
        if not isinstance(artifact, Mapping) or artifact.get("kind") != INFERENCE_KIND:
            raise ValueError("Expected native_band_ssl_weights_only_inference_v1 only.")
        _base(self, artifact, device)
        self.feature_training_kind = self.training_kind
        self.supervised_ancestry = _ancestry(artifact, self.config)
        if self.supervised_ancestry is not None:
            mode = self.supervised_ancestry["mode"]
            self.training_kind = "supervised_downstream_" + mode
            if mode != "frozen_readout":
                self.feature_training_kind = "supervised_feature_encoder"
        else:
            self.training_kind = "train_fitted_readout_on_" + self.feature_training_kind
        with torch.device("meta"):
            model = NativeBandTemporalModel(
                self.config.width, self.config.latent, self.config.blocks, self.config.heads
            ).float()
        _check_state(artifact.get("model"), model.state_dict())
        model.load_state_dict(artifact["model"], strict=True, assign=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.encoder = self.model.encoder

    @property
    def config_metadata(self):
        return copy.deepcopy(self._config_metadata)

    @property
    def scaler_metadata(self):
        return copy.deepcopy(self._scaler_metadata)

    @property
    def source_bindings(self):
        return copy.deepcopy(self._bindings)

    @torch.inference_mode()
    def encode(self, x, observed, metadata):
        # Inherited encoder-only implementation validates raw native inputs and
        # batches with saved scalers. It never requires the readout/query path.
        return legacy.NativeAcousticEncoder.encode(self, x, observed, metadata)

    @torch.inference_mode()
    def forecast(self, x, observed, metadata, query):
        x, observed, metadata = legacy._inputs(x, observed, metadata, self.config.history)
        query = _query(query, metadata)
        self.model.eval()
        result = []
        for start in range(0, len(x), self.config.batch_size):
            end = min(start + self.config.batch_size, len(x))
            with np.errstate(over="ignore", invalid="ignore"):
                values = self.scalers.channels(x[start:end], observed[start:end])
            if not np.isfinite(values).all():
                raise ValueError("Observed scaled dB must be finite at model precision.")
            prediction = (
                self.model.forecast(
                    torch.as_tensor(values, device=self.device),
                    torch.as_tensor(observed[start:end].copy(order="C"), device=self.device),
                    torch.as_tensor(metadata[start:end].copy(order="C"), device=self.device),
                    torch.as_tensor(query[start:end].copy(order="C"), device=self.device),
                )
                .cpu()
                .numpy()
            )
            prediction = (
                prediction * self.scalers.target_std[None, :, None]
                + self.scalers.target_mean[None, :, None]
            )
            if prediction.shape != (end - start, 3, 5) or not np.isfinite(prediction).all():
                raise ValueError("Forecast output must be finite native dB [N,3,5].")
            result.append(prediction)
        return np.concatenate(result)


def load_inference(weights, *, device="cpu"):
    return NativeBandAcousticPredictor(weights, device=device)
