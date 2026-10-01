"""Single parameter-free nonlinear frequency-conditioning hypothesis.

The inherited shared temporal objectives, gradients, pooling and heads stay
unchanged. Only GELU(projected channel values + native metadata), before
observed-count channel pooling, differs from the immutable legacy encoder.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from marine_echo.models.native_temporal import (
    QUANTILES,
    NativeTemporalModel,
    QueryHead,
    SharedTemporalEncoder,
)
from marine_echo.models.sigreg import SlicedEppsPulley

ARCHITECTURE = "nonlinear_frequency_conditioned_v1"
__all__ = ["ARCHITECTURE", "QUANTILES", "NativeBandEncoder", "NativeBandTemporalModel", "QueryHead"]


class NativeBandEncoder(SharedTemporalEncoder):
    """Identical inherited constructor/tensors; GELU acts separately per channel."""

    def tokens(self, x, observed, metadata):
        if x.ndim != 3 or x.shape != observed.shape or x.shape[2] != 4:
            raise ValueError("Expected values/masks [B,T,4].")
        if metadata.shape != (x.shape[0], 4, 10) or not torch.isfinite(metadata).all():
            raise ValueError("Expected finite native metadata [B,4,10].")
        if x.shape[1] < 1 or not torch.isfinite(x[observed]).all():
            raise ValueError("Observed values must be finite and context nonempty.")
        observed = observed.bool()
        clean = torch.where(observed, x, 0)
        pad = (-x.shape[1]) % self.patch
        clean = F.pad(clean, (0, 0, 0, pad))
        observed = F.pad(observed, (0, 0, 0, pad))
        b, t, c = clean.shape
        values = clean.reshape(b, t // self.patch, self.patch, c).transpose(2, 3)
        masks = observed.reshape(b, t // self.patch, self.patch, c).transpose(2, 3)
        tokens = self.patch_projection(torch.cat([values, masks.to(x.dtype)], dim=-1))
        tokens = tokens + self.metadata_projection(metadata).unsqueeze(1)
        tokens = F.gelu(tokens)
        counts = masks.sum(-1).to(x.dtype)
        tokens = (tokens * counts.unsqueeze(-1)).sum(2) / counts.sum(2).clamp_min(1).unsqueeze(-1)
        valid = counts.sum(2) > 0
        positions = torch.arange(tokens.shape[1], device=x.device, dtype=x.dtype).unsqueeze(-1)
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10000) / self.width)
        )
        angles = positions * frequencies
        positional = torch.zeros(tokens.shape[1], self.width, device=x.device, dtype=x.dtype)
        positional[:, 0::2] = angles.sin()
        positional[:, 1::2] = angles[:, : positional[:, 1::2].shape[1]].cos()
        tokens = (tokens + positional) * valid.unsqueeze(-1)
        safe_valid = valid.clone()
        safe_valid[~valid.any(1), 0] = True  # explicit sentinel avoids all-masked softmax
        for block in self.blocks:
            tokens = block(tokens, src_key_padding_mask=~safe_valid)
            tokens = torch.where(valid.unsqueeze(-1), tokens, 0)
        return tokens, valid


class NativeBandTemporalModel(NativeTemporalModel):
    """Legacy head construction order and inherited losses on the revised encoder."""

    def __init__(self, width=192, latent=64, blocks=4, heads=4):
        nn.Module.__init__(self)
        self.encoder = NativeBandEncoder(width, latent, blocks, heads)
        self.readout = QueryHead(latent, 5, latent)
        self.predictor = QueryHead(latent, latent, width)
        self.reconstruction = nn.Linear(latent, 16)
        self.regularizer = SlicedEppsPulley()
