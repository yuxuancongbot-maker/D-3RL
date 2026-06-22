"""Compatibility workspace alias for D3RL/Ada Bridger training."""

from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace


class D3RLWorkspace(TrainAdaBridgerWorkspace):
    """Default D3RL workspace backed by the integrated Ada Bridger trainer.

    This preserves the cleaner ``D3RLWorkspace`` target while using the
    feature-complete scheduler/RL training loop migrated from the original
    Ada-BRIDGER implementation.
    """


__all__ = ["D3RLWorkspace"]
