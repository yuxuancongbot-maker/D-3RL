"""Standard supervised training workspace placeholder."""

from __future__ import annotations

from diffusion_policy.workspace.base import BaseWorkspace


class StandardTrainWorkspace(BaseWorkspace):
    """Workspace for standard diffusion/action-supervised training.

    The full training engine will be connected during the migration phases.
    """

    include_keys = ("global_step", "epoch")

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self.global_step = 0
        self.epoch = 0

    def run(self):
        raise NotImplementedError("Standard training engine migration is pending.")
