"""Matched direct and temporal representation models for AEON hourly products.

The source-specific cohort, masks, normalization and partition gate live in the
training adapter. These modules consume only the preceding 24 source products.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from marine_echo.models.sigreg import SlicedEppsPulley


class AeonEncoder(nn.Module):
    """Encode a fixed 24-by-4 value/mask history into one representation."""

    def __init__(self, *, width: int = 128, layers: int = 3) -> None:
        super().__init__()
        if width <= 0 or layers < 2:
            raise ValueError("AEON encoder requires positive width and at least two layers.")
        modules: list[nn.Module] = [nn.Linear(24 * 4 * 2, width), nn.GELU()]
        for _ in range(layers - 1):
            modules.extend((nn.Linear(width, width), nn.GELU()))
        self.network = nn.Sequential(*modules)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (24, 4) or mask.shape != values.shape:
            raise ValueError("Expected matching [batch,24,4] values and masks.")
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Observed AEON input values must be finite.")
        observed = torch.where(mask, values, torch.zeros_like(values))
        features = torch.cat((observed, mask.to(values.dtype)), dim=-1).flatten(start_dim=1)
        return self.network(features)


class AeonDirect(nn.Module):
    """Direct supervised comparator with the same encoder and quantile head."""

    def __init__(self, *, width: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.encoder = AeonEncoder(width=width, layers=layers)
        self.head = nn.Linear(width, 3 * 5)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(values, mask)).reshape(-1, 3, 5)
        return raw.sort(dim=-1).values


class AeonTemporalSSL(nn.Module):
    """Predict the representation of the last six observed context products.

    The teacher sees all 24 *past* products in a TRAIN-only pretraining window.
    The student sees only the first 18 products. At supervised issuance, the
    encoder sees the full preceding 24 products, never a future target.
    """

    def __init__(
        self,
        *,
        mode: Literal["ema", "shared_sigreg"],
        width: int = 128,
        layers: int = 3,
        regularizer_weight: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("ema", "shared_sigreg"):
            raise ValueError("Unknown AEON temporal SSL mode.")
        self.mode = mode
        self.encoder = AeonEncoder(width=width, layers=layers)
        self.teacher = deepcopy(self.encoder) if mode == "ema" else None
        if self.teacher is not None:
            self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.regularizer = SlicedEppsPulley(num_slices=64, n_points=17)
        self.regularizer_weight = (
            regularizer_weight
            if regularizer_weight is not None
            else (0.03 if mode == "ema" else 0.04)
        )
        if self.regularizer_weight < 0:
            raise ValueError("Regularizer weight must be nonnegative.")
        self.head = nn.Linear(width, 3 * 5)

    @staticmethod
    def _short_context(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1:] != (24, 4) or mask.shape != values.shape:
            raise ValueError("Expected matching [batch,24,4] values and masks.")
        short_values = values.clone()
        short_mask = mask.clone()
        short_values[:, 18:] = 0
        short_mask[:, 18:] = False
        return short_values, short_mask

    def context_view(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        short_values, short_mask = self._short_context(values, mask)
        return self.encoder(short_values, short_mask)

    def teacher_view(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        module = self.teacher if self.teacher is not None else self.encoder
        if values.ndim != 3 or values.shape[1:] != (24, 4) or mask.shape != values.shape:
            raise ValueError("Expected matching [batch,24,4] values and masks.")
        target_values = values.clone()
        target_mask = mask.clone()
        target_values[:, :18] = 0
        target_mask[:, :18] = False
        return module(target_values, target_mask)

    def pretrain_loss(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if len(values) < 2:
            raise ValueError("Temporal SSL needs at least two TRAIN windows per batch.")
        predicted = self.predictor(self.context_view(values, mask))
        target = self.teacher_view(values, mask)
        prediction_loss = F.smooth_l1_loss(predicted, target.detach())
        regularized = predicted if self.mode == "ema" else target
        return prediction_loss + self.regularizer_weight * self.regularizer(regularized)

    @torch.no_grad()
    def update_teacher(self, *, momentum: float = 0.996) -> None:
        if self.teacher is None:
            raise ValueError("Shared-SIGReg uses one encoder and has no EMA teacher.")
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must lie in [0,1).")
        for teacher_parameter, online_parameter in zip(
            self.teacher.parameters(), self.encoder.parameters(), strict=True
        ):
            teacher_parameter.lerp_(online_parameter, 1 - momentum)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(values, mask)).reshape(-1, 3, 5)
        return raw.sort(dim=-1).values
