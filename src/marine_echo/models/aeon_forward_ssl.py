"""TRAIN-only forward acoustic EMA representation model for AEON products."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from marine_echo.models.aeon_ssl import AeonEncoder


class AeonForwardSSL(nn.Module):
    """Predict the representation of six subsequent TRAIN source products.

    The online encoder sees all 24 preceding products. The EMA target encoder
    sees only six subsequent products, padded into a disjoint 24-step view.
    At forecast issuance, forward() consumes the preceding products only.
    """

    def __init__(
        self,
        *,
        width: int = 128,
        layers: int = 3,
        variance_floor: float = 0.1,
        variance_weight: float = 0.04,
    ) -> None:
        super().__init__()
        if variance_floor <= 0 or variance_weight <= 0:
            raise ValueError("Forward EMA variance floor and weight must be positive.")
        self.encoder = AeonEncoder(width=width, layers=layers)
        self.teacher = deepcopy(self.encoder)
        self.teacher.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.head = nn.Linear(width, 15)
        self.variance_floor = variance_floor
        self.variance_weight = variance_weight

    @staticmethod
    def target_view(
        future: torch.Tensor, future_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            future.ndim != 3
            or future.shape[1:] != (6, 4)
            or future_mask.shape != future.shape
            or future_mask.dtype != torch.bool
            or not torch.isfinite(future[future_mask]).all()
        ):
            raise ValueError("Forward EMA target needs six masked source products.")
        view = torch.zeros((len(future), 24, 4), dtype=future.dtype, device=future.device)
        mask = torch.zeros((len(future), 24, 4), dtype=torch.bool, device=future.device)
        view[:, 18:] = torch.where(future_mask, future, 0)
        mask[:, 18:] = future_mask
        return view, mask

    def variance_floor_penalty(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.ndim != 2 or len(representation) < 2:
            raise ValueError("Forward EMA variance floor needs a batched representation.")
        standard_deviation = torch.sqrt(representation.var(dim=0, unbiased=True) + 1e-4)
        return F.relu(self.variance_floor - standard_deviation).mean()

    def pretrain_loss(
        self,
        past: torch.Tensor,
        past_mask: torch.Tensor,
        future: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_values, target_mask = self.target_view(future, future_mask)
        context = self.encoder(past, past_mask)
        prediction = self.predictor(context)
        with torch.no_grad():
            target = self.teacher(target_values, target_mask)
        online_future = self.encoder(target_values, target_mask)
        return (
            F.smooth_l1_loss(prediction, target)
            + self.variance_weight * self.variance_floor_penalty(prediction)
            + self.variance_weight * self.variance_floor_penalty(online_future)
        )

    @torch.no_grad()
    def update_teacher(self, *, momentum: float = 0.996) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("Forward EMA momentum must lie in [0,1).")
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online, 1 - momentum)

    def forward(self, past: torch.Tensor, past_mask: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.encoder(past, past_mask)).reshape(-1, 3, 5)
        return raw.sort(dim=-1).values
