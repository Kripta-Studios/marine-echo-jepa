"""Read-only diagnosis of the stopped raw-response midpoint continuation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from marine_echo.models.compact import DirectForecaster
from marine_echo.training.raw_response_cohort import load_raw_cohorts
from marine_echo.training.raw_response_development import (
    BATCH,
    FROZEN_RUN_CONFIG,
    MIDPOINT,
    MODEL_CONFIG,
    QUANTILES,
    SEED,
    UPDATES,
    _checkpoint_identity,
    _checked_batch_label_count,
    _load_checkpoint,
)


def main() -> None:
    root = Path.cwd()
    review = root / "orchestration/reviews/V2_RAW_RESPONSE_PREFIT_CORRECTED_20260927.json"
    output = root / FROZEN_RUN_CONFIG["output_path"]
    fit, assessment = load_raw_cohorts(root, review_path=review)
    mid = output / f"direct-checkpoint-{MIDPOINT}.pt"
    final = output / f"direct-checkpoint-{UPDATES}.pt"
    if not mid.is_file() or not final.is_file() or (output / "result.json").exists():
        raise ValueError("Only the stopped midpoint/final pair may be diagnosed")
    saved_mid = torch.load(mid, map_location="cuda", weights_only=True)
    normalizer = saved_mid["identity"]["normalizer"]
    identity = _checkpoint_identity(fit, assessment, root, normalizer)
    x = torch.from_numpy(
        np.where(
            fit.context_mask,
            (fit.context - normalizer["context_mean"]) / normalizer["context_std"],
            0,
        ).astype(np.float32)
    )
    mask = torch.from_numpy(fit.context_mask)
    y = torch.from_numpy(
        np.where(
            fit.target_mask,
            (fit.targets - normalizer["target_mean"]) / normalizer["target_std"],
            0,
        ).astype(np.float32)
    )
    ymask = torch.from_numpy(fit.target_mask)
    torch.manual_seed(SEED)
    model = DirectForecaster(MODEL_CONFIG).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FROZEN_RUN_CONFIG["learning_rate"],
        weight_decay=FROZEN_RUN_CONFIG["weight_decay"],
        betas=tuple(FROZEN_RUN_CONFIG["adamw_betas"]), eps=FROZEN_RUN_CONFIG["adamw_eps"],
    )
    _load_checkpoint(mid, model, optimizer, step=MIDPOINT, identity=identity, device="cuda")
    levels = torch.tensor(QUANTILES, dtype=torch.float32, device="cuda")
    model.train()
    for step in range(MIDPOINT, UPDATES):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, step]))
        selected = rng.choice(len(fit.cutoffs), BATCH, replace=False)
        optimizer.param_groups[0]["lr"] = FROZEN_RUN_CONFIG["learning_rate"] * min(
            1.0, (step + 1) / FROZEN_RUN_CONFIG["warmup_updates"]
        )
        optimizer.zero_grad(set_to_none=True)
        forecast = model(x[selected].cuda(), mask[selected].cuda()).quantiles
        truth = y[selected].cuda()[:, :, None]
        valid = ymask[selected].cuda()[:, :, None]
        count = _checked_batch_label_count(valid)
        error = truth - forecast
        loss = torch.where(valid, torch.maximum(levels * error, (levels - 1) * error), 0).sum() / count / len(QUANTILES)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), FROZEN_RUN_CONFIG["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
    saved_final = torch.load(final, map_location="cuda", weights_only=True)
    differences = {
        key: float(torch.max(torch.abs(value - saved_final["model"][key])).item())
        for key, value in model.state_dict().items()
    }
    relative = {
        key: float(torch.max(torch.abs(value - saved_final["model"][key]) / torch.maximum(torch.abs(saved_final["model"][key]), torch.tensor(1e-7, device="cuda"))).item())
        for key, value in model.state_dict().items()
    }
    print(json.dumps({"max_abs_difference": max(differences.values()), "max_relative_difference": max(relative.values()), "worst_parameter": max(differences, key=differences.get), "allclose_1e_6_1e_7": all(torch.allclose(value, saved_final["model"][key], rtol=1e-6, atol=1e-7) for key, value in model.state_dict().items())}, indent=2))


if __name__ == "__main__":
    main()
