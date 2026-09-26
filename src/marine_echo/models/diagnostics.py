"""Collapse diagnostics for learned acoustic embeddings."""

from __future__ import annotations

import torch


def effective_rank(embeddings: torch.Tensor) -> float:
    """Entropy effective rank of centered singular values; zero for constants."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("Expected at least two embedding rows [N,D].")
    centered = embeddings.detach().float() - embeddings.detach().float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if total <= 1e-12:
        return 0.0
    probability = energy / total
    nonzero = probability[probability > 0]
    return float(torch.exp(-(nonzero * nonzero.log()).sum()).item())
