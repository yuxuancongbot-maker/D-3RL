"""Discrepancy utilities for D3RL oracle labels."""

from __future__ import annotations

import torch


def action_l1_discrepancy(source_action: torch.Tensor, refined_action: torch.Tensor) -> torch.Tensor:
    """Compute per-sample normalized L1 discrepancy between two action chunks.

    Returns a tensor of shape ``[B]``.
    """
    if source_action.shape != refined_action.shape:
        raise ValueError(
            f"Action shapes must match, got {tuple(source_action.shape)} and {tuple(refined_action.shape)}"
        )
    reduce_dims = tuple(range(1, source_action.ndim))
    action_dim = source_action.shape[-1]
    return torch.sum(torch.abs(source_action - refined_action), dim=reduce_dims) / float(action_dim)


def binary_oracle_labels(discrepancy: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
    """Generate binary oracle labels using ``median(discrepancy) * eta`` threshold."""
    threshold = torch.median(discrepancy) * float(eta)
    return (discrepancy >= threshold).long()
