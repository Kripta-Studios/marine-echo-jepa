"""Finite train-only updates and weights-only checkpoint/resume."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from marine_echo.models.compact import DirectForecaster, ForecastOutput, TemporalJEPA

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class TrainBatch:
    context: torch.Tensor
    context_mask: torch.Tensor
    target_index: torch.Tensor | None = None
    target_profile: torch.Tensor | None = None
    target_profile_mask: torch.Tensor | None = None
    future: torch.Tensor | None = None
    future_mask: torch.Tensor | None = None
    source_partition: str = "train"
    source_file_ids: tuple[str, ...] = ()


def supervised_loss(forecast: ForecastOutput, batch: TrainBatch) -> torch.Tensor:
    if (
        batch.target_index is None
        or batch.target_profile is None
        or batch.target_profile_mask is None
    ):
        raise ValueError(
            "Direct training requires scalar and profile targets with masks."
        )
    if (
        batch.target_index.shape != forecast.quantiles.shape[:2]
        or batch.target_profile.shape != forecast.profile.shape
    ):
        raise ValueError("Target shapes differ from forecast output.")
    if not forecast.eligible.any():
        raise ValueError("No eligible context in supervised batch.")
    if not torch.isfinite(batch.target_index[forecast.eligible]).all():
        raise ValueError("Eligible scalar target is non-finite.")
    error = batch.target_index[..., None] - forecast.quantiles
    q = forecast.quantiles.new_tensor(QUANTILES)
    pinball = torch.maximum(q * error, (q - 1.0) * error)
    scalar_loss = pinball[forecast.eligible].mean()
    profile_valid = batch.target_profile_mask & forecast.eligible[:, None, None, None]
    if profile_valid.any():
        if not torch.isfinite(batch.target_profile[profile_valid]).all():
            raise ValueError("Eligible profile target is non-finite.")
        profile_loss = (
            (forecast.profile - batch.target_profile).square()[profile_valid].mean()
        )
    else:
        profile_loss = scalar_loss.new_zeros(())
    return scalar_loss + 0.1 * profile_loss


def run_steps(
    model: DirectForecaster | TemporalJEPA,
    optimizer: torch.optim.Optimizer,
    batches: list[TrainBatch],
    *,
    family: Literal["direct", "ema_jepa", "shared_sigreg"],
    start_step: int,
    updates: int,
    partition: str,
    allowed_train_file_ids: set[str] | None = None,
) -> list[float]:
    """Execute a declared finite number of updates; caller owns one GPU process."""
    if (
        partition != "train"
        or not batches
        or any(batch.source_partition != "train" for batch in batches)
    ):
        raise ValueError("Optimizer updates may read training batches only.")
    if any(batch.source_file_ids for batch in batches) and (
        allowed_train_file_ids is None
        or any(
            not set(batch.source_file_ids).issubset(allowed_train_file_ids)
            for batch in batches
        )
    ):
        raise ValueError(
            "A batch includes source files outside the frozen train allowlist."
        )
    if start_step < 0 or updates <= 0 or start_step + updates > 3_000:
        raise ValueError("Update count exceeds the bounded phase cap.")
    if family == "direct" and not isinstance(model, DirectForecaster):
        raise ValueError("Direct family needs DirectForecaster.")
    if family != "direct" and (
        not isinstance(model, TemporalJEPA)
        or model.mode != ("ema" if family == "ema_jepa" else "shared_sigreg")
    ):
        raise ValueError("JEPA family and gradient mode differ.")
    model.train()
    losses = []
    for step in range(start_step, start_step + updates):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        if family == "direct":
            assert isinstance(model, DirectForecaster)
            loss = supervised_loss(model(batch.context, batch.context_mask), batch)
        else:
            assert isinstance(model, TemporalJEPA)
            if batch.future is None or batch.future_mask is None:
                raise ValueError("JEPA update requires future train observations.")
            loss = model.objective(
                batch.context, batch.context_mask, batch.future, batch.future_mask
            ).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if family == "ema_jepa":
            assert isinstance(model, TemporalJEPA)
            model.update_teacher()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    protocol_sha256: str,
) -> None:
    if path.exists():
        raise FileExistsError("Checkpoint path is immutable and already exists.")
    if (
        step < 0
        or len(protocol_sha256) != 64
        or any(char not in "0123456789abcdef" for char in protocol_sha256)
    ):
        raise ValueError("Checkpoint step or protocol digest is invalid.")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staged_name = tempfile.mkstemp(
        prefix=path.name + ".stage.", dir=path.parent
    )
    os.close(handle)
    staged = Path(staged_name)
    try:
        torch.save(
            {
                "schema_version": 1,
                "step": step,
                "protocol_sha256": protocol_sha256,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            },
            staged,
        )
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    protocol_sha256: str,
) -> int:
    """Load a locally produced state dictionary; never unpickle an object graph."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("protocol_sha256") != protocol_sha256
    ):
        raise ValueError("Checkpoint schema or frozen protocol digest differs.")
    step = checkpoint["step"]
    if not isinstance(step, int) or not 0 <= step <= 3_000:
        raise ValueError("Checkpoint update count is invalid.")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_cpu"])
    if torch.cuda.is_available() and checkpoint["rng_cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    return step
