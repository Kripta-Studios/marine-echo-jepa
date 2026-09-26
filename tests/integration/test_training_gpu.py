"""Opt-in CUDA software smoke and bounded 100-update resource profile."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from marine_echo.models.compact import DirectForecaster, ModelConfig, TemporalJEPA
from marine_echo.training.loop import TrainBatch, load_checkpoint, run_steps, save_checkpoint


def _cuda_batch(batch_size: int) -> TrainBatch:
    device = torch.device("cuda")
    context = torch.randn(batch_size, 96, 4, 64, device=device)
    future = torch.randn(batch_size, 3, 4, 4, 64, device=device)
    return TrainBatch(
        context=context,
        context_mask=torch.ones_like(context, dtype=torch.bool),
        target_index=torch.randn(batch_size, 3, device=device),
        target_profile=torch.randn(batch_size, 3, 4, 64, device=device),
        target_profile_mask=torch.ones(batch_size, 3, 4, 64, device=device, dtype=torch.bool),
        future=future,
        future_mask=torch.ones_like(future, dtype=torch.bool),
    )


def _process_tree_rss() -> int:
    import psutil

    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    return sum(item.memory_info().rss for item in processes if item.is_running())


def _write_evidence(name: str, evidence: dict[str, object]) -> None:
    root = Path(__file__).resolve().parents[2] / "evidence" / "models"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


@pytest.mark.skipif(os.environ.get("MARINE_ECHO_GPU_SMOKE") != "1", reason="Opt-in owner GPU smoke")
def test_cuda_direct_ema_shared_backward_and_checkpoint(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "The requested local GPU is unavailable."
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    batch = _cuda_batch(2)
    config = ModelConfig(width=32, layers=1, heads=4)
    results: dict[str, object] = {}
    for family in ("direct", "ema_jepa", "shared_sigreg"):
        model = (
            DirectForecaster(config)
            if family == "direct"
            else TemporalJEPA(config, mode="ema" if family == "ema_jepa" else "shared_sigreg")
        ).cuda()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss = run_steps(
            model, optimizer, [batch], family=family, start_step=0, updates=1, partition="train"
        )
        torch.cuda.synchronize()
        result = {
            "loss": loss[0],
            "seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "observed_process_tree_rss_bytes": _process_tree_rss(),
        }
        assert result["peak_reserved_bytes"] < 10 * 1024**3
        if family == "direct":
            checkpoint = tmp_path / "direct_step_1.pt"
            save_checkpoint(checkpoint, model, optimizer, step=1, protocol_sha256="a" * 64)
            restored = DirectForecaster(config).cuda()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
            assert (
                load_checkpoint(checkpoint, restored, restored_optimizer, protocol_sha256="a" * 64)
                == 1
            )
        results[family] = result
        del model, optimizer
        torch.cuda.empty_cache()
    _write_evidence(
        "gpu_smoke.json",
        {
            "status": "SOFTWARE_SMOKE_ONLY",
            "data": "synthetic_fixture",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "families": results,
        },
    )


@pytest.mark.skipif(
    os.environ.get("MARINE_ECHO_PROFILE_GPU") != "1", reason="Opt-in bounded GPU profile"
)
def test_cuda_direct_100_update_profile() -> None:
    assert torch.cuda.is_available(), "The requested local GPU is unavailable."
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    config = ModelConfig()
    model = DirectForecaster(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = _cuda_batch(16)
    torch.cuda.reset_peak_memory_stats()
    step_seconds = []
    process_peak = 0
    start = time.perf_counter()
    for step in range(100):
        step_start = time.perf_counter()
        run_steps(
            model,
            optimizer,
            [batch],
            family="direct",
            start_step=step,
            updates=1,
            partition="train",
        )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_start)
        process_peak = max(process_peak, _process_tree_rss())
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    assert allocated < 10 * 1024**3
    assert reserved < 10 * 1024**3
    assert process_peak < 22 * 1024**3
    _write_evidence(
        "gpu_100_updates.json",
        {
            "status": "SOFTWARE_RESOURCE_PROFILE_ONLY",
            "data": "synthetic_fixture",
            "family": "direct",
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "batch_size": 16,
            "width": config.width,
            "layers": config.layers,
            "updates": 100,
            "elapsed_seconds": elapsed,
            "seconds_per_update_mean": elapsed / 100,
            "seconds_per_update_max": max(step_seconds),
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "observed_process_tree_rss_peak_bytes": process_peak,
            "estimated_3000_update_seconds_at_profile_mean": elapsed * 30,
        },
    )
