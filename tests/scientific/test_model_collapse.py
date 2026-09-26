"""Planted collapse probes distinguish degenerate embeddings."""

from __future__ import annotations

import torch

from marine_echo.models.diagnostics import effective_rank


def test_effective_rank_detects_constant_rank_one_and_diverse() -> None:
    torch.manual_seed(23)
    constant = torch.ones(128, 16)
    rank_one = torch.randn(128, 1) @ torch.randn(1, 16)
    diverse = torch.randn(128, 16)
    assert effective_rank(constant) == 0.0
    assert 0.9 <= effective_rank(rank_one) <= 1.1
    assert effective_rank(diverse) > 10.0
