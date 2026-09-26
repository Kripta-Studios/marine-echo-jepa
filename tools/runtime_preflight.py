"""Inspect local runtime without installing packages or changing system settings."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"status": "NOT_INSTALLED"}
    try:
        result = subprocess.run([executable, *command[1:]], capture_output=True,
                                text=True, timeout=15, check=False)
        return {"status": "OK" if result.returncode == 0 else "ERROR",
                "returncode": result.returncode, "stdout": result.stdout[:4000],
                "stderr": result.stderr[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "ERROR", "reason": str(exc)}


def torch_probe() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {"status": "NOT_INSTALLED", "cuda_training_verified": False}
    report: dict[str, Any] = {"torch_version": torch.__version__,
                             "build_cuda": torch.version.cuda,
                             "cuda_training_verified": False}
    if not torch.cuda.is_available():
        report["status"] = "CUDA_UNAVAILABLE"
        return report
    try:
        report.update(device=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)),
                      compiled_arches=torch.cuda.get_arch_list())
        started = time.monotonic()
        torch.manual_seed(7)
        linear = torch.nn.Linear(32, 32).cuda()
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        x = torch.randn(4, 32, device="cuda")
        conv = torch.nn.Conv2d(4, 8, 3).cuda()
        image = torch.randn(2, 4, 16, 16, device="cuda", requires_grad=True)
        q = torch.randn(2, 4, 16, 16, device="cuda", requires_grad=True)
        attended = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        loss = linear(x).square().mean() + conv(image).square().mean() + attended.square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite kernel smoke loss.")
        report.update(status="CUDA_KERNEL_SMOKE_PASSED", cuda_training_verified=True,
                      elapsed_seconds=round(time.monotonic() - started, 3),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      note="Small FP32 kernel smoke only; full model/BF16/resume remain to test.")
    except Exception as exc:
        report.update(status="CUDA_KERNEL_SMOKE_FAILED", reason=str(exc)[:1000])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="Run a tiny FP32 GPU kernel test.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/preflight/runtime.json")
    args = parser.parse_args()
    disk = shutil.disk_usage(ROOT)
    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "free_disk_gib": round(disk.free / 1024**3, 2),
        "git": command_output(["git", "--version"]),
        "node": command_output(["node", "--version"]),
        "uv": command_output(["uv", "--version"]),
        "codex": command_output(["codex", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "torch": torch_probe() if args.torch else {"status": "NOT_REQUESTED"},
        "model_account_access": "NOT_VERIFIED_BY_THIS_SCRIPT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.torch and not report["torch"].get("cuda_training_verified", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
