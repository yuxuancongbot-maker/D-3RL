"""Cost penalty schedules for lightweight PPO."""

from __future__ import annotations


def linear_cost_warmup(
    epoch: int,
    target: float,
    task_only_epochs: int,
    warmup_epochs: int,
) -> float:
    """Return the cost coefficient for an epoch.

    The coefficient is zero for the task-only phase and then linearly increases
    to ``target`` over ``warmup_epochs``.
    """
    if epoch < task_only_epochs:
        return 0.0
    if warmup_epochs <= 0:
        return float(target)
    progress = (epoch - task_only_epochs) / warmup_epochs
    return float(target) * min(1.0, max(0.0, progress))


def refinement_step_cost(refinement_steps, max_refinement_steps: int = 10):
    """Normalize actual refinement steps into [0, 1]."""
    return refinement_steps / float(max_refinement_steps)
