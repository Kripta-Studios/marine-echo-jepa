"""Source-bound desktop operational runtime; original Resources science unchanged."""

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch

from marine_echo.training import native_desktop_resources_v2 as native_resources
from marine_echo.training import native_resources as original_resources
from marine_echo.training import native_ssl as original_core

OWNER_SHA256 = "18e29745c7595a9ea50bccf23e3124b06c57106929ac20c40a8f04776ff598e4"


def operational_sources():
    root = Path(native_resources.__file__).resolve().parents[3]
    return [
        Path(__file__).resolve(),
        Path(native_resources.__file__).resolve(),
        Path(original_resources.__file__).resolve(),
        Path(original_core.__file__).resolve(),
        native_resources.OWNER_PATH,
        root / "docs/adr/0024-owner-authorized-vlc-desktop-coexecution.md",
    ]


def validate_operational_authority():
    if native_resources.OWNER_SHA256 != OWNER_SHA256:
        raise ValueError("Operational resource owner pin differs")
    raw = native_resources.OWNER_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA256:
        raise ValueError("Exact owner operational authority changed")
    native_resources.validate_owner_resolution(json.loads(raw))


class Resources:
    def __init__(self, device, output):
        self.device, self.output = device, Path(output)
        self.peak_rss = self.peak_allocated = self.peak_reserved = 0
        self.gpu_elapsed = 0.0
        self.elapsed = 0.0
        self.started = time.monotonic()
        self.lock = None

    def __enter__(self):
        if str(self.device).startswith("cuda"):
            if not torch.cuda.is_available() or torch.version.cuda is None:
                raise RuntimeError("Verified CUDA PyTorch required; no CPU fallback or install.")
            # Shared local native-run lock, never delete stale/unowned locks automatically.
            folder = Path(__file__).resolve().parents[3] / "evidence/ssl-builder-v1"
            folder.mkdir(parents=True, exist_ok=True)
            self.lock = folder / "gpu-owner.lock"
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"pid": os.getpid(), "output": str(self.output.resolve())}, stream)
            try:
                native_resources.inspect_gpu_ownership(self.output)
                torch.cuda.set_per_process_memory_fraction(
                    min(
                        1.0,
                        0.98
                        * 10
                        * 2**30
                        / torch.cuda.get_device_properties(self.device).total_memory,
                    ),
                    self.device,
                )
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)
            except BaseException:
                self.lock.unlink()
                self.lock = None
                raise
        self.started = time.monotonic()
        return self

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        if self.peak_rss >= 22 * 2**30:
            raise RuntimeError("RSS resource limit exceeded.")
        if str(self.device).startswith("cuda"):
            self.peak_allocated = max(
                self.peak_allocated, torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved = max(
                self.peak_reserved, torch.cuda.max_memory_reserved(self.device)
            )
            if max(self.peak_allocated, self.peak_reserved) >= 10 * 2**30:
                raise RuntimeError("CUDA resource limit exceeded.")

    def snapshot(self):
        self.check()
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - self.started + self.elapsed
        return {
            "peak_rss_bytes": self.peak_rss,
            "peak_allocated_bytes": self.peak_allocated,
            "peak_reserved_bytes": self.peak_reserved,
            "elapsed_seconds": elapsed,
            "elapsed_gpu_seconds": elapsed if str(self.device).startswith("cuda") else 0,
            "gpu_time_definition": "synchronized elapsed device-owned runtime, including probes and evaluation",
        }

    def __exit__(self, *args):
        if self.lock is not None:
            self.lock.unlink()
