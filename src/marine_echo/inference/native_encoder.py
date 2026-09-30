"""Frozen encoder-only inference for native integrated acoustic products.

Inputs are raw-dB observed prefixes and encoded native measurement metadata.
The result is an embedding, with no forecast/head/query or depth-profile claim.
Selected-checkpoint source bindings are exposed as provenance, never executed.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from marine_echo.models.native_temporal import CFTemporalEncoder, SharedTemporalEncoder
from marine_echo.training.native_ssl import Config, Scalers

TRAINING_KINDS = {
    "shared_ssl": "ssl_pretrained",
    "masked_ssl": "ssl_pretrained",
    "cf_jepa": "ssl_pretrained",
    "permuted_ssl": "permuted_pairing_ssl_control",
    "direct": "supervised_feature_encoder",
    "random_frozen": "untrained_control",
}
CHANNEL_FREQUENCIES_HZ = (38000, 125000, 200000, 455000)


def _config(raw):
    if not isinstance(raw, dict) or raw.get("method") not in TRAINING_KINDS:
        raise ValueError("Selected encoder requires its original supported method/config.")
    fields = ["history", "batch_size", "heads"]
    fields += (
        ["cf_width", "cf_latent", "cf_blocks"]
        if raw["method"] == "cf_jepa"
        else ["width", "latent", "blocks"]
    )
    if any(type(raw.get(key)) is not int or raw[key] < 1 for key in fields):
        raise ValueError(
            "Saved history, batch size and backbone dimensions must be explicit positive integers."
        )
    if raw["history"] not in (24, 96):
        raise ValueError("Only saved histories 24/96 are supported.")
    if raw["method"] != "cf_jepa" and raw["width"] % raw["heads"]:
        raise ValueError("Saved shared width must be divisible by attention heads.")
    try:
        return Config(**raw)
    except TypeError as error:
        raise ValueError("Incompatible saved Config fields.") from error


def _scalers(raw):
    expected = {"channel_mean": (4,), "channel_std": (4,), "target_mean": (3,), "target_std": (3,)}
    if not isinstance(raw, dict) or set(raw) != set(expected):
        raise ValueError("Complete original TRAIN-fitted scaler state is required.")
    for key, shape in expected.items():
        values = np.asarray(raw[key])
        if values.shape != shape or values.dtype.kind not in "fi" or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved scaler: {key}")
        if key.endswith("std") and (values <= 0).any():
            raise ValueError(f"Saved scaler standard deviation must be positive: {key}")
    with np.errstate(over="ignore", invalid="ignore"):
        scalers = Scalers.from_dict(raw)
    if any(
        not np.isfinite(v).all() or (k.endswith("std") and (v <= 0).any())
        for k, v in vars(scalers).items()
    ):
        raise ValueError("Saved scalers must remain finite/positive at model float32 precision.")
    return scalers


def _bindings(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Original selected-checkpoint source bindings are required.")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in value)
        for key, value in raw.items()
    ):
        raise ValueError("Source bindings must contain path strings and SHA256 digests.")
    # No opening, importing, hashing or evaluating the ancestral paths.
    return copy.deepcopy(raw)


def _inputs(x, observed, metadata, history):
    x, observed, metadata = np.asarray(x), np.asarray(observed), np.asarray(metadata)
    if x.ndim != 3 or x.shape[1:] != (history, 4) or not len(x):
        raise ValueError(
            "Expected exactly the saved observed history [N,H,4], with no future suffix."
        )
    if x.dtype.kind != "f" or x.dtype.itemsize not in (4, 8):
        raise ValueError("Raw-dB context must be float32 or float64.")
    if observed.shape != x.shape or observed.dtype != np.bool_:
        raise ValueError("Observed masks must be Boolean and match [N,H,4].")
    if not observed[:, :, 0].all() or not np.isfinite(x[observed]).all():
        raise ValueError(
            "Issuance requires a complete primary 38-kHz prefix and finite observed dB."
        )
    if (
        metadata.shape != (len(x), 4, 10)
        or metadata.dtype.kind != "f"
        or not np.isfinite(metadata).all()
    ):
        raise ValueError("Expected complete finite native metadata [N,4,10].")
    frequency = np.asarray(CHANNEL_FREQUENCIES_HZ) / 455000
    if not np.allclose(metadata[:, :, 0], frequency[None], atol=1e-6, rtol=0):
        raise ValueError("Native channel order/frequency is 38/125/200/455 kHz.")
    if not np.allclose(metadata[:, :, 1], 1, atol=1e-6, rtol=0):
        raise ValueError("Native interval contract is nominal 60 min, encoded seconds/3600.")
    if not np.allclose(metadata[:, :, 2], 1, atol=1e-6, rtol=0):
        raise ValueError("Only native integrated products are supported.")
    if (metadata[:, :, 3] < 0).any() or (metadata[:, :, 4] <= metadata[:, :, 3]).any():
        raise ValueError("Native upper/lower bounds must be nonnegative and increasing.")
    if not np.allclose(metadata[:, :, 5], 0, atol=1e-6, rtol=0):
        raise ValueError("The encoded orientation contract uses 0=unknown.")
    if not np.isin(metadata[:, :, 6:9], [0, 1]).all():
        raise ValueError("Native processing/instrument/clock known flags must be 0/1.")
    if not np.allclose(metadata[:, :, 9], 0, atol=1e-6, rtol=0):
        raise ValueError("Context metadata must have issuance-relative offset 0.")
    with np.errstate(over="ignore", invalid="ignore"):
        encoded_metadata = metadata.astype(np.float32, copy=False)
    if (
        not np.isfinite(encoded_metadata).all()
        or (encoded_metadata[:, :, 4] <= encoded_metadata[:, :, 3]).any()
    ):
        raise ValueError(
            "Native metadata must remain finite with increasing bounds at model precision."
        )
    return x, observed, encoded_metadata


class NativeAcousticEncoder:
    """Load only the selected shared/EMA backbone and reuse saved TRAIN scalers.

    The interface accepts only context/masks/metadata. Source-date contiguity and
    association with the issuance cutoff remain the caller's responsibility:
    these three arrays contain no timestamps or source interval identifiers.
    Downstream supervised encoder artifacts are explicitly unsupported here.
    """

    def __init__(self, weights, device="cpu"):
        artifact = torch.load(weights, weights_only=True, map_location="cpu")
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("kind") != "native_ssl_selected_encoder_v1"
        ):
            raise ValueError(
                "Expected native_ssl_selected_encoder_v1; other artifacts are unsupported."
            )
        self.config = _config(artifact.get("config"))
        self.scalers = _scalers(artifact.get("scalers"))
        self._bindings = _bindings(artifact.get("bindings"))
        self._config_metadata = copy.deepcopy(artifact["config"])
        self._scaler_metadata = copy.deepcopy(artifact["scalers"])
        self.method = self.config.method
        self.training_kind = TRAINING_KINDS[self.method]
        self.feature_branch = "selected_ema" if self.method == "cf_jepa" else "shared_encoder"
        self.selected_pretrain_step = artifact.get("selected_pretrain_step")
        if self.selected_pretrain_step is not None and (
            type(self.selected_pretrain_step) is not int or self.selected_pretrain_step < 0
        ):
            raise ValueError("Optional selected pretrain step must be a nonnegative integer.")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only local CPU/CUDA inference devices are supported.")
        state = artifact.get("encoder")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("Selected encoder tensors are missing.")
        # Meta construction allocates no random weights and consumes no RNG.
        # Direct backbone constructors create no head, optimizer or fit policy.
        with torch.device("meta"):
            if self.method == "cf_jepa":
                encoder = CFTemporalEncoder(
                    self.config.cf_width, self.config.cf_latent, self.config.cf_blocks
                )
            else:
                encoder = SharedTemporalEncoder(
                    self.config.width, self.config.latent, self.config.blocks, self.config.heads
                )
            # Saved native weights/scalers are float32 regardless of the caller's
            # default dtype. Cast the empty meta template without global changes.
            encoder = encoder.float()
        expected = encoder.state_dict()
        if set(state) != set(expected):
            raise ValueError("Selected state must contain exactly the backbone parameters/buffers.")
        for key, template in expected.items():
            value = state[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.device.type != "cpu"
                or value.shape != template.shape
                or value.dtype != template.dtype
            ):
                raise ValueError(f"Incompatible selected encoder tensor: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite selected encoder tensor: {key}")
        try:
            encoder.load_state_dict(state, strict=True, assign=True)
        except (RuntimeError, TypeError) as error:
            raise ValueError("Incompatible selected backbone state.") from error
        self.encoder = encoder.eval().requires_grad_(False).to(self.device)
        self.embedding_dimension = (
            self.config.cf_latent if self.method == "cf_jepa" else self.config.latent
        )

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
    def encode(self, x: np.ndarray, observed: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Return float32 embeddings[N,D]; never consume query/target/future data."""
        x, observed, metadata = _inputs(x, observed, metadata, self.config.history)
        self.encoder.eval()
        encoded = []
        for start in range(0, len(x), self.config.batch_size):
            end = min(start + self.config.batch_size, len(x))
            with np.errstate(over="ignore", invalid="ignore"):
                values = self.scalers.channels(x[start:end], observed[start:end])
            if not np.isfinite(values).all():
                raise ValueError("Observed scaled dB must remain finite at model precision.")
            # Copy bounded mask/metadata batches: NumPy can call a single-row
            # view contiguous while retaining a negative leading stride.
            latent = self.encoder.encode(
                torch.as_tensor(values, device=self.device),
                torch.as_tensor(observed[start:end].copy(order="C"), device=self.device),
                torch.as_tensor(metadata[start:end].copy(order="C"), device=self.device),
            )
            if (
                latent.shape != (end - start, self.embedding_dimension)
                or not torch.isfinite(latent).all()
            ):
                raise ValueError("Encoder returned incompatible or nonfinite embeddings.")
            encoded.append(latent.cpu().numpy())
        return np.concatenate(encoded, axis=0)
