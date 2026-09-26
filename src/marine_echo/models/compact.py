"""Matched compact direct, EMA-JEPA and shared-SIGReg acoustic models."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .sigreg import SlicedEppsPulley


@dataclass(frozen=True)
class ModelConfig:
    width: int = 96
    layers: int = 3
    heads: int = 4
    frequencies: int = 4
    range_bins: int = 64
    context_bins: int = 96

    def __post_init__(self) -> None:
        if self.width <= 0 or self.width % self.heads or self.layers <= 0:
            raise ValueError("Invalid Transformer width, head count or layer count.")
        if self.frequencies != 4 or self.range_bins != 64 or self.context_bins != 96:
            raise ValueError("This D1 adapter requires [96,4,64] context shape.")


@dataclass
class ForecastOutput:
    quantiles: torch.Tensor
    profile: torch.Tensor
    eligible: torch.Tensor


@dataclass
class ObjectiveOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    regularizer_loss: torch.Tensor
    predicted: torch.Tensor
    target: torch.Tensor
    eligible: torch.Tensor


class AcousticEncoder(nn.Module):
    """Four-time by eight-sample patches, with explicit values and masks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.patch = nn.Linear(4 * config.frequencies * 8 * 2, config.width)
        self.time_position = nn.Parameter(torch.zeros(24, config.width))
        self.range_position = nn.Parameter(torch.zeros(8, config.width))
        nn.init.trunc_normal_(self.time_position, std=0.02)
        nn.init.trunc_normal_(self.range_position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=2 * config.width,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.width)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 4 or mask.shape != values.shape:
            raise ValueError("Expected matching [B,T,4,64] values and mask.")
        batch, times, frequencies, ranges = values.shape
        if (
            times not in (4, 96)
            or frequencies != self.config.frequencies
            or ranges != self.config.range_bins
        ):
            raise ValueError("Unexpected acoustic window shape.")
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Valid acoustic input contains a non-finite value.")
        masked = torch.where(mask, values, torch.zeros_like(values))
        patches = masked.reshape(batch, times // 4, 4, frequencies, 8, 8).permute(0, 1, 4, 2, 3, 5)
        patch_mask = mask.reshape(batch, times // 4, 4, frequencies, 8, 8).permute(0, 1, 4, 2, 3, 5)
        patch_valid = patch_mask.any(dim=(3, 4, 5)).reshape(batch, -1)
        features = torch.cat(
            (
                patches.reshape(batch, times // 4, 8, -1),
                patch_mask.reshape(batch, times // 4, 8, -1).to(values.dtype),
            ),
            dim=-1,
        )
        tokens = self.patch(features)
        tokens = (
            tokens
            + self.time_position[: times // 4][None, :, None, :]
            + self.range_position[None, None, :, :]
        )
        tokens = tokens.reshape(batch, -1, self.config.width)
        padding = ~patch_valid
        padding[~patch_valid.any(dim=1), 0] = False
        encoded = self.final_norm(self.transformer(tokens, src_key_padding_mask=padding))
        return encoded, patch_valid


class HorizonPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(3, 8, config.width) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(config.width)

    def forward(self, context: torch.Tensor, context_valid: torch.Tensor) -> torch.Tensor:
        query = self.query.reshape(1, 24, -1).expand(context.shape[0], -1, -1)
        padding = ~context_valid.clone()
        padding[~context_valid.any(dim=1), 0] = False
        attended, _ = self.cross_attention(
            query, context, context, key_padding_mask=padding, need_weights=False
        )
        return self.norm(query + attended).reshape(context.shape[0], 3, 8, -1)


class ForecastHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.scalar = nn.Linear(width, 5)
        self.profile = nn.Linear(width, 4 * 8)

    def forward(self, predicted: torch.Tensor, eligible: torch.Tensor) -> ForecastOutput:
        raw = self.scalar(predicted.mean(dim=2))
        quantiles = torch.cat(
            (
                raw[..., :1],
                raw[..., :1] + torch.cumsum(F.softplus(raw[..., 1:]), dim=-1),
            ),
            dim=-1,
        )
        batch = predicted.shape[0]
        profile = (
            self.profile(predicted)
            .reshape(batch, 3, 8, 4, 8)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch, 3, 4, 64)
        )
        return ForecastOutput(quantiles=quantiles, profile=profile, eligible=eligible)


class DirectForecaster(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = AcousticEncoder(config)
        self.predictor = HorizonPredictor(config)
        self.head = ForecastHead(config.width)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor) -> ForecastOutput:
        tokens, valid = self.encoder(context, context_mask)
        predicted = self.predictor(tokens, valid)
        return self.head(predicted, valid.any(dim=1))


class TemporalJEPA(nn.Module):
    """Passive acoustic future-token predictor with explicit gradient mode."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        mode: Literal["ema", "shared_sigreg"],
        sigreg_weight: float = 0.04,
    ) -> None:
        super().__init__()
        if sigreg_weight < 0:
            raise ValueError("SIGReg weight must be nonnegative.")
        self.mode = mode
        self.encoder = AcousticEncoder(config)
        self.predictor = HorizonPredictor(config)
        self.head = ForecastHead(config.width)
        self.target_encoder = deepcopy(self.encoder) if mode == "ema" else self.encoder
        if mode == "ema":
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        self.sigreg = SlicedEppsPulley() if mode == "shared_sigreg" else None
        self.sigreg_weight = sigreg_weight

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        if self.mode != "ema" or not 0.0 <= momentum < 1.0:
            raise ValueError("EMA update requires EMA mode and momentum in [0,1).")
        for teacher, student in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            teacher.lerp_(student, 1.0 - momentum)

    def encode_future(
        self, future: torch.Tensor, future_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            future.ndim != 5
            or future.shape[1:] != (3, 4, 4, 64)
            or future_mask.shape != future.shape
        ):
            raise ValueError("Expected future [B,3,4,4,64] and matching mask.")
        batch = future.shape[0]
        target, valid = self.target_encoder(
            future.reshape(batch * 3, 4, 4, 64),
            future_mask.reshape(batch * 3, 4, 4, 64),
        )
        return target.reshape(batch, 3, 8, -1), valid.reshape(batch, 3, 8)

    def objective(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        future: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> ObjectiveOutput:
        encoded, context_valid = self.encoder(context, context_mask)
        predicted = self.predictor(encoded, context_valid)
        if self.mode == "ema":
            with torch.no_grad():
                target, target_valid = self.encode_future(future, future_mask)
        else:
            target, target_valid = self.encode_future(future, future_mask)
        sample_valid = context_valid.any(dim=1)
        support = target_valid & sample_valid[:, None, None]
        if not support.any():
            raise ValueError("No valid future target tokens in the batch.")
        prediction_loss = (predicted - target).square()[support].mean()
        regularizer = predicted.new_zeros(())
        if self.sigreg is not None and self.sigreg_weight > 0:
            regularizer = self.sigreg(target[support])
        return ObjectiveOutput(
            loss=prediction_loss + self.sigreg_weight * regularizer,
            prediction_loss=prediction_loss,
            regularizer_loss=regularizer,
            predicted=predicted,
            target=target,
            eligible=sample_valid,
        )

    def forecast(self, context: torch.Tensor, context_mask: torch.Tensor) -> ForecastOutput:
        encoded, valid = self.encoder(context, context_mask)
        return self.head(self.predictor(encoded, valid), valid.any(dim=1))
