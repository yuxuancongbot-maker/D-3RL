"""Supervised scheduler pretraining helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_TARGET_IF_SUFFICIENT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_TARGET_IF_NEEDS_REFINE = (0.0, 0.2, 0.3, 0.5)


def binary_labels_to_soft_targets(
    labels: torch.Tensor,
    target_if_sufficient=DEFAULT_TARGET_IF_SUFFICIENT,
    target_if_needs_refine=DEFAULT_TARGET_IF_NEEDS_REFINE,
) -> torch.Tensor:
    """Convert binary oracle labels to conservative scheduler soft targets.

    Label ``0`` means zero refinement is sufficient. Label ``1`` means non-zero
    refinement is needed and is conservatively biased toward stronger refinement.
    """
    labels = labels.long()
    sufficient = torch.tensor(target_if_sufficient, dtype=torch.float32, device=labels.device)
    needs_refine = torch.tensor(target_if_needs_refine, dtype=torch.float32, device=labels.device)
    targets = torch.where(labels[:, None] == 0, sufficient[None], needs_refine[None])
    return targets


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy for probability targets."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have same shape, got {logits.shape} and {targets.shape}")
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
