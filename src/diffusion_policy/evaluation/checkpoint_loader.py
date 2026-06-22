"""Checkpoint loading helpers.

The loader preserves the legacy payload contract while allowing new code to map
old Hydra ``_target_`` strings to refactored classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.legacy_aliases import resolve_legacy_target


def load_payload(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load a legacy/new checkpoint payload."""
    path = Path(path)
    return torch.load(path.open("rb"), pickle_module=dill, map_location=map_location)


def get_workspace_target(cfg: OmegaConf) -> str:
    """Resolve the workspace target from a config."""
    target = cfg.get("_target_", None)
    if target is None and "workspace" in cfg:
        target = cfg.workspace.get("_target_", None)
    if target is None:
        raise KeyError("Config does not contain a workspace _target_.")
    return resolve_legacy_target(str(target))


def instantiate_workspace_from_payload(payload: dict[str, Any], output_dir: str | None = None):
    """Instantiate a workspace from a checkpoint payload without loading state."""
    cfg = payload["cfg"]
    target = get_workspace_target(cfg)
    cls = hydra.utils.get_class(target)
    try:
        return cls(cfg, output_dir=output_dir)
    except TypeError:
        return cls(cfg)


def load_workspace_from_checkpoint(
    path: str | Path,
    output_dir: str | None = None,
    map_location: str | torch.device | None = None,
    **load_kwargs,
):
    """Load a workspace from a checkpoint path."""
    payload = load_payload(path, map_location=map_location)
    workspace = instantiate_workspace_from_payload(payload, output_dir=output_dir)
    workspace.load_payload(payload, **load_kwargs)
    return workspace, payload


def select_policy_from_workspace(workspace, use_ema: bool | None = None):
    """Return the policy/model used for evaluation."""
    if use_ema is None:
        cfg = getattr(workspace, "cfg", None)
        use_ema = bool(getattr(getattr(cfg, "training", None), "use_ema", False)) if cfg is not None else False
    if use_ema and hasattr(workspace, "ema_model"):
        return workspace.ema_model
    if hasattr(workspace, "model"):
        return workspace.model
    if hasattr(workspace, "policy"):
        return workspace.policy
    raise AttributeError("Workspace has no model, ema_model, or policy attribute.")
