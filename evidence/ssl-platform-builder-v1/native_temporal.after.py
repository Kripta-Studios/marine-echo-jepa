"""Observation-aware shared temporal models for native integrated acoustic products.

No deployment identity is a feature. Shared SSL independently encodes real future
crops, without EMA or stop-gradient. Output latents are unconstrained for SIGReg.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from marine_echo.models.sigreg import SlicedEppsPulley

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (1, 3, 6)


def deterministic_adaptive_avg_pool1d(values: torch.Tensor, output_size: int) -> torch.Tensor:
    """Adaptive average bins using slice means, without native pooling backward.

    For input length L/output O, bin i is [floor(i*L/O), ceil((i+1)*L/O)).
    Integer arithmetic preserves overlapping bins, including O greater than L.
    Ordinary mean/stack autograd preserves the objective while avoiding CUDA's
    nondeterministic adaptive pooling backward. Reduction rounding can differ
    from the native kernel; the bins and mathematical gradients are identical.
    """
    if values.ndim not in (2, 3) or values.shape[-1] < 1:
        raise ValueError("Expected nonempty temporal values [C,L] or [B,C,L].")
    if type(output_size) is not int or output_size < 1:
        raise ValueError("Output size must be a positive integer.")
    length = values.shape[-1]
    return torch.stack(
        [
            values[
                ..., i * length // output_size : ((i + 1) * length + output_size - 1) // output_size
            ].mean(-1)
            for i in range(output_size)
        ],
        dim=-1,
    )


class SharedTemporalEncoder(nn.Module):
    """Patch4 shared channel projection, observation pooling, four temporal blocks.

    Feed-forward expansion six provides capacity in the temporal computation;
    default model size is measured by the runner. Empty crops return exact zero.
    Right padding and masked fill values cannot enter a projection or attention.
    """

    def __init__(self, width=192, latent=64, blocks=4, heads=4, patch=4):
        super().__init__()
        if min(width, latent, blocks, heads) < 1 or width % heads or patch != 4:
            raise ValueError("Invalid shared temporal dimensions or patch length.")
        self.width, self.latent, self.patch = width, latent, patch
        self.patch_projection = nn.Linear(patch * 2, width)
        self.metadata_projection = nn.Sequential(
            nn.Linear(10, width), nn.GELU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width, heads, width * 6, dropout=0, batch_first=True, norm_first=True
                )
                for _ in range(blocks)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent))

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

    def encode(self, x, observed, metadata):
        tokens, valid = self.tokens(x, observed, metadata)
        pooled = tokens.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
        latent = self.output(pooled)
        return torch.where(valid.any(1).unsqueeze(-1), latent, 0)

    def forward(self, x, observed, metadata):
        return self.encode(x, observed, metadata)


class QueryHead(nn.Module):
    def __init__(self, latent, output, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + 11, hidden), nn.GELU(), nn.Linear(hidden, output)
        )

    def forward(self, context, query):
        if query.shape != (context.shape[0], 3, 10) or not torch.isfinite(query).all():
            raise ValueError("Expected finite native query [B,3,10].")
        horizon = context.new_tensor(HORIZONS).view(1, 3, 1).expand(context.shape[0], -1, -1) / 6
        return self.net(torch.cat([context.unsqueeze(1).expand(-1, 3, -1), query, horizon], dim=-1))


class NativeTemporalModel(nn.Module):
    """All comparison heads initialize in a fixed order, regardless of method."""

    def __init__(self, width=192, latent=64, blocks=4, heads=4):
        super().__init__()
        self.encoder = SharedTemporalEncoder(width, latent, blocks, heads)
        self.readout = QueryHead(latent, 5, latent)
        self.predictor = QueryHead(latent, latent, width)
        self.reconstruction = nn.Linear(latent, 16)
        self.regularizer = SlicedEppsPulley()

    def forecast(self, x, observed, metadata, query):
        latent = self.encoder.encode(x, observed, metadata)
        return self.readout(latent, query).sort(dim=-1).values

    def shared_loss(
        self, x, observed, metadata, future, future_observed, query, *, weight=0.03, permuted=False
    ):
        context = self.encoder.encode(x, observed, metadata)
        targets = []
        for h in range(3):
            target_meta = metadata.clone()
            target_meta[:, :, -1] = query[:, h, -1].unsqueeze(-1)
            targets.append(self.encoder.encode(future[:, h], future_observed[:, h], target_meta))
        target = torch.stack(targets, dim=1)
        matched = target.roll(1, dims=0) if permuted else target
        prediction = (self.predictor(context, query) - matched).square().mean(dim=(0, 2)).mean()
        # Effective rows are correlated: one context + three overlapping crops/sample.
        regularization = self.regularizer(torch.cat([context, target.flatten(0, 1)], dim=0))
        return {
            "loss": prediction + weight * regularization,
            "prediction": prediction,
            "regularization": regularization,
        }

    def masked_loss(self, x, observed, metadata, *, generator):
        hide = (torch.rand(x.shape, generator=generator, device=x.device) < 0.25) & observed
        if not hide.any():
            candidates = observed.nonzero()
            if not len(candidates):
                raise ValueError("Masked SSL requires train observations.")
            hide[tuple(candidates[0])] = True
        tokens, _ = self.encoder.tokens(x, observed & ~hide, metadata)
        reconstruction = (
            self.reconstruction(self.encoder.output(tokens))
            .reshape(x.shape[0], -1, 4, 4)
            .transpose(2, 3)
            .reshape(x.shape[0], -1, 4)
        )
        reconstruction = reconstruction[:, : x.shape[1]]
        return (reconstruction[hide] - x[hide]).square().mean()

    def freeze_encoder(self):
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        for parameter in self.encoder.parameters():
            parameter.grad = None


# CF-JEPA adaptation retains the pinned author's multiscale architecture/loss.
# Source: WDSLab/CF-JEPA 5d3d2fd1273c283fbfa03249c078619245e84033.
# MIT License, Copyright (c) 2026 CF-JEPA authors.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


class CFTemporalEncoder(nn.Module):
    """Pinned source Encoder with a native observation/metadata input adapter.

    Convolution kernels 3/9/15, dilations 2**i, residual BN/pointwise blocks.
    Symmetric padding matches author training. Prefix-only forecasting has no
    observations later than issuance cutoff. Whole-crop masked mean readout
    replaces author's eight-bin flattening for the matched lightweight head.
    """

    def __init__(self, width=256, latent=128, blocks=5):
        super().__init__()
        self.input_fc = nn.Linear(48, width)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "dw": nn.ModuleList(
                            [
                                nn.Conv1d(
                                    width,
                                    width,
                                    k,
                                    dilation=2 ** min(i, 7),
                                    groups=width,
                                    bias=False,
                                )
                                for k in (3, 9, 15)
                            ]
                        ),
                        "bn1": nn.BatchNorm1d(width),
                        "pw": nn.Conv1d(width, width, 1, bias=False),
                        "bn2": nn.BatchNorm1d(width),
                    }
                )
                for i in range(blocks)
            ]
        )
        self.output_fc = nn.Linear(width, latent)

    def sequence(self, x, observed, metadata):
        clean = torch.where(observed, x, 0)
        meta = metadata.flatten(1).unsqueeze(1).expand(-1, x.shape[1], -1)
        h = self.input_fc(torch.cat([clean, observed.to(x.dtype), meta], dim=-1)).transpose(1, 2)
        valid = observed.any(-1).unsqueeze(1)
        h = torch.where(valid, h, 0)
        for block in self.blocks:
            out = sum(
                conv(
                    F.pad(
                        h,
                        (
                            ((conv.kernel_size[0] - 1) * conv.dilation[0]) // 2,
                            ((conv.kernel_size[0] - 1) * conv.dilation[0])
                            - ((conv.kernel_size[0] - 1) * conv.dilation[0]) // 2,
                        ),
                    )
                )
                for conv in block["dw"]
            )
            out = F.gelu(block["bn1"](out))
            out = F.gelu(block["bn2"](block["pw"](out)))
            h = torch.where(valid, h + out, 0)
        return self.output_fc(h.transpose(1, 2)) * valid.transpose(1, 2)

    def encode(self, x, observed, metadata):
        z = self.sequence(x, observed, metadata)
        return z.sum(1) / observed.any(-1).sum(1).clamp_min(1).unsqueeze(-1)


class CFNativeModel(nn.Module):
    """Explicit CF adaptation, not full paper reproduction. EMA is forecast encoder."""

    def __init__(self, width=256, latent=128, blocks=5, heads=4):
        super().__init__()
        self.online = CFTemporalEncoder(width, latent, blocks)
        self.encoder = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.readout = QueryHead(latent, 5, latent)
        self.predictors = nn.ModuleList([nn.Linear(latent, latent) for _ in HORIZONS])
        with torch.no_grad():
            for predictor in self.predictors:
                predictor.weight.copy_(torch.eye(latent) + torch.randn(latent, latent) * 0.01)
                predictor.bias.zero_()

    def forecast(self, x, observed, metadata, query):
        self.encoder.eval()
        return self.readout(self.encoder.encode(x, observed, metadata), query).sort(dim=-1).values

    def freeze_encoder(self):
        self.encoder.requires_grad_(False).eval()

    def cf_loss(self, x, observed, metadata, future, future_observed, query, *, step, total):
        self.encoder.eval()
        # Unique contiguous TRAIN suffix: steps1..4,5..6,7..9. Overlapping
        # horizon blocks must not duplicate intervals in the author trajectory.
        trajectory = torch.cat([x, future[:, 0], future[:, 1, 2:4], future[:, 2, 1:4]], dim=1)
        trajectory_mask = torch.cat(
            [
                observed,
                future_observed[:, 0],
                future_observed[:, 1, 2:4],
                future_observed[:, 2, 1:4],
            ],
            dim=1,
        )
        with torch.no_grad():
            full_target = self.encoder.sequence(trajectory, trajectory_mask, metadata)
        time_steps = trajectory.shape[1]
        crops = []
        prediction = x.new_zeros(())
        prediction_terms = 0
        # Pinned author crop bounds/four views, and three zones in the remaining
        # future. All roles are TRAIN-only; crops shift across temporal positions.
        for _ in range(4):
            low, high = (0.31, 0.686) if time_steps >= 50 else (0.6, 0.8)
            length = max(int(time_steps * np.random.uniform(low, high)), 4)
            start = int(np.random.randint(0, max(time_steps - length - 3, 0) + 1))
            end = start + length
            crop = self.online.sequence(
                trajectory[:, start:end], trajectory_mask[:, start:end], metadata
            )
            crops.append(crop)
            zone_size = (time_steps - end) // 3
            for h, predictor in enumerate(self.predictors):
                zone_start = end + h * zone_size
                zone_end = end + (h + 1) * zone_size if h < 2 else time_steps
                available = zone_end - zone_start
                if available < 1:
                    continue
                predicted = predictor(crop)
                if available >= length:
                    target_start = int(
                        np.random.randint(zone_start, max(zone_end - length, zone_start) + 1)
                    )
                    target = full_target[:, target_start : target_start + length]
                else:
                    target = full_target[:, zone_start:zone_end]
                    predicted = deterministic_adaptive_avg_pool1d(
                        predicted.transpose(1, 2), available
                    ).transpose(1, 2)
                prediction = prediction + F.l1_loss(
                    F.normalize(predicted, dim=-1), F.normalize(target, dim=-1)
                )
                prediction_terms += 1
        prediction = prediction / max(prediction_terms, 1)
        pooled = torch.cat(
            [deterministic_adaptive_avg_pool1d(z.transpose(1, 2), 8).mean(-1) for z in crops]
        )
        variance = F.relu(1 - pooled.std(0)).mean()
        centered = pooled - pooled.mean(0)
        cov = centered.T @ centered / (len(pooled) - 1)
        covariance = (cov.square().sum() - cov.diagonal().square().sum()) / cov.shape[0]
        invariance = x.new_zeros(())
        scales = [p for p in (2, 4, 8) if p <= min(z.shape[1] for z in crops)] or [1]
        for scale in scales:
            z = torch.stack(
                [
                    deterministic_adaptive_avg_pool1d(c.transpose(1, 2), scale).mean(-1)
                    for c in crops
                ]
            )
            invariance = invariance + (z - z.mean(0, keepdim=True)).square().mean() / len(scales)
        return (
            (1 - (step + 1) / total) * prediction
            + 0.038 * variance
            + 0.143 * covariance
            + 0.401 * invariance
        )

    @torch.no_grad()
    def update_ema(self, step, total):
        decay = 1 - (1 - 0.992) * (math.cos(math.pi * (step + 1) / total) + 1) / 2
        # Author behavior: EMA parameters only; target BN buffers remain initialized.
        for target, online in zip(self.encoder.parameters(), self.online.parameters(), strict=True):
            target.mul_(decay).add_(online, alpha=1 - decay)
