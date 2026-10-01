"""Load-only zero-SSL CF controls; forecasts are supervised quantiles, not SSL.

No provenance files are opened and no optimizer/model factory or RNG setup runs.
The corresponding fresh CF encoder branch is not an EMA-trained teacher.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from marine_echo.inference import native_encoder as legacy
from marine_echo.inference.native_band_acoustic import _check_state, _query
from marine_echo.models.native_temporal import CFTemporalEncoder, QueryHead
from marine_echo.training.native_cf_controls import (
    ENCODER_KINDS,
    INFERENCE_KIND,
    REAL,
    SYNTHETIC,
    TRAINING_KINDS,
    CFControlModel,
    Config,
    core_config,
)


def _base(instance, artifact, device):
    if not isinstance(artifact, Mapping) or str(device) not in ("cpu", "cuda:0"):
        raise ValueError("Safe control mapping and explicit cpu/cuda:0 required")
    allowed = {
        "kind",
        "encoder",
        "model",
        "config",
        "core_config",
        "scalers",
        "bindings",
        "supervised_ancestry",
        "evidence_kind",
        "review_sha256",
    }
    if set(artifact) - allowed:
        raise ValueError("Unknown or unsafe control artifact fields")
    evidence = artifact.get("evidence_kind")
    if evidence not in (REAL, SYNTHETIC) or (evidence == SYNTHETIC and str(device) != "cpu"):
        raise ValueError("Explicit control evidence kind; synthetic CPU only")
    raw = artifact.get("config")
    if not isinstance(raw, dict) or set(raw) != set(Config.__dataclass_fields__):
        raise ValueError("Complete exact control configuration required")
    instance.config = Config(**raw)
    instance.config.validate(correctness_smoke=evidence == SYNTHETIC)
    source = artifact.get("core_config")
    if (
        not isinstance(source, dict)
        or source.get("method") != "cf_jepa"
        or (
            source.get("seed"),
            source.get("history"),
            source.get("cf_width"),
            source.get("cf_latent"),
            source.get("cf_blocks"),
            source.get("heads"),
        )
        != (
            instance.config.seed,
            96,
            instance.config.width,
            instance.config.latent,
            instance.config.blocks,
            4,
        )
        or source.get("pretrain_updates") != 0
    ):
        raise ValueError("Explicit zero-pretrain source-factory configuration required")
    # Dataclass configuration only: no factory, RNG setup or provenance reads.
    expected = core_config(instance.config).to_dict()
    if set(source) != set(expected) or any(
        type(v) is not type(expected[k])
        or (type(v) is float and not np.isfinite(v))
        or (k != "cf_source" and v != expected[k])
        for k, v in source.items()
    ):
        raise ValueError("Missing/unsafe source Config fields")
    ancestry = artifact.get("supervised_ancestry")
    if not isinstance(ancestry, dict):
        raise ValueError("Complete control/readout/scaler ancestry required")  # noqa: TRY004
    fixed = {
        "mode": instance.config.mode,
        "ssl_only": False,
        "ssl_updates": 0,
        "ancestor_encoder_sha256": None,
        "ancestor_run_sha256": None,
        "initialization_seed": instance.config.seed,
        "readout_initialization_seed": instance.config.seed + 100000,
    }
    if set(ancestry) != set(fixed) | {
        "supervised_updates",
        "selected_supervised_step",
        "encoder_supervised_updates",
    } or any(
        ancestry.get(k, "MISSING") != v or type(ancestry.get(k)) is not type(v)
        for k, v in fixed.items()
    ):
        raise ValueError("A control must have explicit zero SSL and no fitted weight parent")
    steps, selected, encoder_steps = [
        ancestry.get(k)
        for k in ("supervised_updates", "selected_supervised_step", "encoder_supervised_updates")
    ]
    if (
        type(steps) is not int
        or type(selected) is not int
        or type(encoder_steps) is not int
        or not 1 <= selected <= steps <= instance.config.updates
        or encoder_steps != (0 if instance.config.method == "cf_random_frozen" else steps)
    ):
        raise ValueError("Accurate fitted control update/selection counts required")
    instance.device = torch.device(device)
    instance.scalers = legacy._scalers(artifact.get("scalers"))
    instance._bindings = legacy._bindings(artifact.get("bindings"))
    instance._config_metadata = copy.deepcopy(raw)
    instance._scaler_metadata = copy.deepcopy(artifact["scalers"])
    instance.supervised_ancestry = copy.deepcopy(ancestry)
    instance.training_kind = TRAINING_KINDS[instance.config.method]
    instance.feature_branch = "fresh_corresponding_cf_encoder_branch_zero_ssl"
    instance.embedding_dimension = instance.config.latent
    instance.method = instance.config.method


class NativeCFControlEncoder(legacy.NativeAcousticEncoder):
    def __init__(self, weights, device="cpu"):
        artifact = torch.load(weights, map_location="cpu", weights_only=True)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("kind") not in ENCODER_KINDS.values()
            or "model" in artifact
        ):
            raise ValueError("Only distinct CF control selected encoder kinds supported")
        _base(self, artifact, device)
        if artifact["kind"] != ENCODER_KINDS[self.config.method]:
            raise ValueError("Untrained/supervised control kind mismatch")
        with torch.device("meta"):
            encoder = CFTemporalEncoder(
                self.config.width, self.config.latent, self.config.blocks
            ).float()
        _check_state(artifact.get("encoder"), encoder.state_dict())
        encoder.load_state_dict(artifact["encoder"], strict=True, assign=True)
        self.encoder = encoder.eval().requires_grad_(False).to(self.device)


class NativeCFControlPredictor(NativeCFControlEncoder):
    def __init__(self, weights, device="cpu"):
        artifact = torch.load(weights, map_location="cpu", weights_only=True)
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("kind") != INFERENCE_KIND
            or "encoder" in artifact
        ):
            raise ValueError("Only native_cf_control_weights_only_inference_v1 supported")
        _base(self, artifact, device)
        with torch.device("meta"):
            model = CFControlModel(
                CFTemporalEncoder(
                    self.config.width, self.config.latent, self.config.blocks
                ).float(),
                QueryHead(self.config.latent, 5, self.config.latent).float(),
            )
        _check_state(artifact.get("model"), model.state_dict())
        model.load_state_dict(artifact["model"], strict=True, assign=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.encoder = self.model.encoder

    @torch.inference_mode()
    def forecast(self, x, observed, metadata, query):
        x, observed, metadata = legacy._inputs(x, observed, metadata, self.config.history)
        query = _query(query, metadata)
        values = []
        self.model.eval()
        for start in range(0, len(x), self.config.batch_size):
            end = min(start + self.config.batch_size, len(x))
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = self.scalers.channels(x[start:end], observed[start:end])
            if not np.isfinite(scaled).all():
                raise ValueError("Scaled observed inputs must remain finite")
            prediction = (
                self.model.forecast(
                    torch.as_tensor(scaled, device=self.device),
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
                raise ValueError("Finite native five-quantile forecasts required")
            values.append(prediction)
        return np.concatenate(values)


def load_encoder(weights, *, device="cpu"):
    return NativeCFControlEncoder(weights, device=device)


def load_inference(weights, *, device="cpu"):
    return NativeCFControlPredictor(weights, device=device)
