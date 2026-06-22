"""Legacy Hydra target aliases.

Old checkpoints/configs often contain ``_target_`` strings from the original
repository. This table maps them to new implementations once migration is
complete. During early migration the aliases document intended compatibility.
"""

from __future__ import annotations

LEGACY_TARGET_MAP = {
    "diffusion_policy.workspace.base_workspace.BaseWorkspace": "diffusion_policy.workspace.base.BaseWorkspace",
    "diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace.TrainDiffusionUnetLowdimWorkspace": "diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace.TrainDiffusionUnetLowdimWorkspace",
}


def resolve_legacy_target(target: str) -> str:
    """Return the migrated target for an old Hydra target string."""
    return LEGACY_TARGET_MAP.get(target, target)
