"""Safety guards for lightweight PPO fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SuccessGuard:
    """Detect unacceptable success degradation during cost optimization."""

    baseline_success: float
    tolerance: float = 0.05

    def is_violation(self, eval_success: float) -> bool:
        return eval_success < (self.baseline_success - self.tolerance)
