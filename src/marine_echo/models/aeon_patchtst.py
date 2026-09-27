"""Masked, channel-independent patch transformer for AEON supervised development.

Inspired by Nie et al., ICLR 2023, https://openreview.net/forum?id=Jbdc0vTOcol.
This bounded short-context variant uses six non-overlapping four-hour patches,
shared channel weights, explicit observed masks, and a persistence skip.
"""

from __future__ import annotations

import torch
from torch import nn


class AeonMaskedPatchTransformer(nn.Module):
    """Map preceding 24-by-4 source products to three five-quantile forecasts."""

    def __init__(
        self, *, width: int = 128, layers: int = 3, heads: int = 4,
        dropout: float = 0.1, baseline_scale: float = 1.0,
        baseline_shift: float = 0.0,
    ) -> None:
        super().__init__()
        if width <= 0 or layers < 1 or width % heads or not 0 <= dropout < 1:
            raise ValueError("Invalid AEON patch transformer geometry")
        self.baseline_scale = baseline_scale
        self.baseline_shift = baseline_shift
        self.patch_projection = nn.Linear(8, width)
        self.position = nn.Parameter(torch.zeros(1, 6, width))
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(4 * 6 * width), nn.Linear(4 * 6 * width, width * 2),
            nn.GELU(), nn.Linear(width * 2, 15),
        )

    def encode_channels(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (24, 4) or mask.shape != values.shape:
            raise ValueError("Expected matching AEON [batch,24,4] values and masks")
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Observed AEON inputs must be finite")
        observed = torch.where(mask, values, torch.zeros_like(values))
        channel_first = torch.stack((observed, mask.to(values.dtype)), dim=-1)
        patches = channel_first.permute(0, 2, 1, 3).reshape(-1, 6, 8)
        encoded = self.encoder(self.patch_projection(patches) + self.position)
        return encoded.reshape(values.shape[0], 4, 6, -1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        representation = self.encode_channels(values, mask)
        residual = self.head(representation.flatten(start_dim=1)).reshape(-1, 3, 5)
        baseline = values[:, -1, 0] * self.baseline_scale + self.baseline_shift
        return (residual + baseline[:, None, None]).sort(dim=-1).values
