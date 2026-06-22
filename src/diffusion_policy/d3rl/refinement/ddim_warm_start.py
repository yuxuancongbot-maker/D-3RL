"""DDIM warm-start refinement interface."""

from __future__ import annotations

import torch


class DDIMWarmStartRefiner:
    """Adapter around a pretrained diffusion policy for warm-start refinement.

    Full implementation will connect to the migrated diffusion policy internals.
    This interface already encodes the D3RL contract: zero refinement returns the
    source action unchanged; non-zero refinement uses the diffusion refiner.
    """

    def __init__(self, refinement_policy):
        self.refinement_policy = refinement_policy

    @torch.no_grad()
    def refine(self, obs_dict, init_action: torch.Tensor, refinement_steps: int) -> torch.Tensor:
        if int(refinement_steps) == 0:
            return init_action
        if hasattr(self.refinement_policy, "refine_action"):
            return self.refinement_policy.refine_action(
                obs_dict=obs_dict,
                init_action=init_action,
                refinement_steps=int(refinement_steps),
            )
        raise NotImplementedError(
            "The migrated diffusion policy must expose refine_action(...) for non-zero refinement."
        )
