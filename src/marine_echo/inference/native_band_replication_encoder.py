"""Typed, load-only selected band encoder API; ancestral kinds are preserved."""

from collections.abc import Mapping

import torch

from marine_echo.inference import native_band_acoustic as original
from marine_echo.inference import native_band_replication_acoustic as replication
from marine_echo.inference.native_encoder import NativeAcousticEncoder
from marine_echo.models.native_band_temporal import NativeBandEncoder


class NativeBandReplicationEncoder(NativeAcousticEncoder):
    """Context-only selected SSL/control features, no supervised encoder loading."""

    def __init__(self, weights, device="cpu"):
        if str(device) not in ("cpu", "cuda:0"):
            raise ValueError("Device must be explicitly cpu or cuda:0.")
        artifact = torch.load(weights, weights_only=True, map_location="cpu")
        if not isinstance(artifact, Mapping):
            raise TypeError("Safe selected artifact mapping required.")
        kind = artifact.get("kind")
        if kind == original.SELECTED_KIND:
            loader = original
            self.artifact_version = 1
        elif kind == replication.SELECTED_KIND:
            loader = replication
            self.artifact_version = 2
        else:
            raise ValueError(
                "Only typed original/replication selected SSL/control encoders supported."
            )
        allowed = {
            "kind",
            "architecture",
            "config",
            "scalers",
            "bindings",
            "encoder",
            "evidence_kind",
            "correctness_smoke",
            "selected_pretrain_step",
        }
        if not set(artifact).issubset(allowed):
            raise ValueError("Unknown or supervised fields in selected encoder artifact.")
        replication._configuration(artifact, torch.device(device))
        loader._base(self, artifact, device)
        self.artifact_kind = kind
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
        loader._check_state(artifact.get("encoder"), encoder.state_dict())
        encoder.load_state_dict(artifact["encoder"], strict=True, assign=True)
        self.encoder = encoder.eval().requires_grad_(False).to(self.device)
