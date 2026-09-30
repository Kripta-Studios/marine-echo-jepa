"""ROOT ONLY: synthetic deterministic CUDA CF pooling/backward, no optimizer.

Run only after integrating the exact platform patch, with root GPU ownership.
This script is not executed by the builder. It neither loads a corpus nor saves
a checkpoint. Output is correctness evidence only, never a scientific score.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

BUILDER = Path(__file__).resolve().parents[2]
MAIN = BUILDER.parent / "marine-echo-jepa"
sys.path.insert(0, str(MAIN / "src"))
from marine_echo.models import native_temporal
from marine_echo.training import native_ssl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--root-gpu-owner", action="store_true", required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "patch-manifest.json").read_text())
    binding = next(c for c in manifest["changes"] if c["target"].endswith("native_temporal.py"))
    if (
        hashlib.sha256(Path(native_temporal.__file__).read_bytes()).hexdigest()
        != binding["after_sha256"]
    ):
        raise ValueError("Integrate exact pooling patch before root CUDA correctness.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Root must configure deterministic cuBLAS before starting Python.")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    evidence = Path(__file__).parent
    with native_ssl.Resources(torch.device(args.device), evidence) as resources:
        pool = native_temporal.deterministic_adaptive_avg_pool1d
        for dtype in (torch.float32, torch.float64):
            for length, output in ((3, 8), (7, 3), (11, 17), (31, 8)):
                torch.manual_seed(7)
                source = torch.randn(2, 3, length, dtype=dtype)
                expected_input = source.clone().requires_grad_()
                expected = F.adaptive_avg_pool1d(expected_input, output)
                weights = torch.randn_like(expected)
                (expected * weights).sum().backward()
                actual_input = source.cuda().requires_grad_()
                actual = pool(actual_input, output)
                (actual * weights.cuda()).sum().backward()
                tolerance = 3e-6 if dtype == torch.float32 else 3e-14
                torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(
                    actual_input.grad.cpu(), expected_input.grad, atol=tolerance, rtol=tolerance
                )
        torch.manual_seed(7)
        base = native_temporal.CFNativeModel(width=12, latent=8, blocks=1).cuda()
        x, future = torch.randn(4, 96, 4, device="cuda"), torch.randn(4, 3, 4, 4, device="cuda")
        metadata = torch.zeros(4, 4, 10, device="cuda")
        metadata[:, :, 3:5] = torch.tensor([0.04, 0.92], device="cuda")
        query = metadata[:, :1].repeat(1, 3, 1)
        query[:, :, -1] = torch.tensor([1, 3, 6], device="cuda")
        inputs = (
            x,
            torch.ones_like(x, dtype=torch.bool),
            metadata,
            future,
            torch.ones_like(future, dtype=torch.bool),
            query,
        )
        original = F.adaptive_avg_pool1d

        def rejected(*a, **k):
            raise AssertionError(
                "CF reached native adaptive pooling; repair did not cover its actual path."
            )

        F.adaptive_avg_pool1d = rejected
        try:
            instances, losses = [], []
            for _ in range(2):
                instance = copy.deepcopy(base)
                np.random.seed(23)
                loss = instance.cf_loss(*inputs, step=0, total=6000)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is None for p in instance.encoder.parameters())
                instances.append(instance)
                losses.append(loss)
            assert torch.equal(*losses)
            for a, b in zip(instances[0].parameters(), instances[1].parameters(), strict=True):
                assert (a.grad is None) == (b.grad is None)
                if a.grad is not None:
                    assert torch.equal(a.grad, b.grad)
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in instances[0].online.parameters()
            )
        finally:
            F.adaptive_avg_pool1d = original
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        print(
            json.dumps(
                {
                    "status": "PASSED",
                    "evidence_kind": "SYNTHETIC_CORRECTNESS_ONLY",
                    "optimizer_updates": 0,
                    "device": "cuda",
                    "cf_gradient_replay": "BIT_IDENTICAL",
                    "native_pooling_used_in_cf": False,
                    "resources": resources.snapshot(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
