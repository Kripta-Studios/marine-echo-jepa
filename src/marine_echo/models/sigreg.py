"""Sliced Epps-Pulley SIGReg adapted from stable-pretraining.

Source: galilai-group/stable-pretraining at
9aa93f8b6153eebb73f57d4853ccf8a13d848310,
stable_pretraining/methods/lejepa.py, EppsPulley and SlicedEppsPulley.
Copyright (c) 2024 rbalestr-lab, MIT license. This compact single-process
adaptation uses 128 projections by default and preserves the upstream
Epps-Pulley N scaling and Gaussian-weighted trapezoidal quadrature.
"""

from __future__ import annotations

import torch
from torch import nn


class SlicedEppsPulley(nn.Module):
    """Gaussian goodness-of-fit over seeded random one-dimensional projections."""

    def __init__(
        self, num_slices: int = 128, t_max: float = 3.0, n_points: int = 17
    ) -> None:
        super().__init__()
        if num_slices <= 0 or n_points < 3 or n_points % 2 != 1 or t_max <= 0:
            raise ValueError("Invalid SIGReg quadrature or projection configuration.")
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        phi = (-0.5 * t.square()).exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[0] = dt
        weights[-1] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            raise ValueError("SIGReg needs at least two embedding rows [N,D].")
        with torch.no_grad():
            generator = torch.Generator(device=embeddings.device).manual_seed(
                int(self.global_step.item())
            )
            directions = torch.randn(
                embeddings.shape[1],
                self.num_slices,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
            directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(
                1e-12
            )
            self.global_step.add_(1)
        projected = embeddings @ directions
        frequencies = projected.unsqueeze(-1) * self.t.to(dtype=embeddings.dtype)
        cosine = frequencies.cos().mean(dim=0)
        sine = frequencies.sin().mean(dim=0)
        error = (cosine - self.phi.to(dtype=embeddings.dtype)).square() + sine.square()
        return (
            (error @ self.weights.to(dtype=embeddings.dtype)) * embeddings.shape[0]
        ).mean()
