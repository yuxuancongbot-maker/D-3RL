"""Utilities for lowdim and nested image observation containers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def is_modality_obs(obs: Any) -> bool:
    """Return True for image/multimodal observation dictionaries.

    Lowdim D3RL uses ``{"obs": Tensor[B,T,D]}``. Image runners and datasets use
    modality dictionaries such as ``{"image": Tensor[B,T,C,H,W], "agent_pos": ...}``,
    sometimes nested under an ``"obs"`` key.
    """
    if not isinstance(obs, Mapping):
        return False
    if "obs" in obs and isinstance(obs["obs"], torch.Tensor):
        return False
    if "obs" in obs and isinstance(obs["obs"], Mapping):
        return True
    return any(isinstance(v, (Mapping, torch.Tensor)) for v in obs.values())


def unwrap_obs(obs_dict: Mapping[str, Any]) -> Any:
    """Return nested modality obs when present, otherwise the original mapping."""
    if "obs" in obs_dict and isinstance(obs_dict["obs"], Mapping):
        return obs_dict["obs"]
    return obs_dict


def slice_obs_steps(obs: Any, n_obs_steps: int) -> Any:
    """Recursively slice the time dimension of observation tensors.

    Tensors with three or more dimensions are assumed to be ``[B,T,...]``. Tensors
    with fewer dimensions are returned unchanged to avoid changing scalar metadata.
    """
    if isinstance(obs, Mapping):
        return {k: slice_obs_steps(v, n_obs_steps) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor) and obs.dim() >= 3:
        return obs[:, :n_obs_steps]
    return obs


def index_obs_batch(obs: Any, idx: torch.Tensor) -> Any:
    """Recursively select batch indices from an observation container."""
    if isinstance(obs, Mapping):
        return {k: index_obs_batch(v, idx) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.index_select(0, idx.to(device=obs.device))
    return obs


def detach_clone_obs(obs: Any) -> Any:
    """Recursively detach and clone tensors in an observation container."""
    if isinstance(obs, Mapping):
        return {k: detach_clone_obs(v) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.detach().clone()
    return obs


def get_obs_item(obs: Any, index: int) -> Any:
    """Recursively take one batch item from an observation container."""
    if isinstance(obs, Mapping):
        return {k: get_obs_item(v, index) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs[index]
    return obs


def stack_obs_list(items: Sequence[Any]) -> Any:
    """Recursively stack a list of observation containers."""
    if len(items) == 0:
        raise ValueError("Cannot stack an empty observation list.")
    first = items[0]
    if isinstance(first, Mapping):
        return {k: stack_obs_list([item[k] for item in items]) for k in first.keys()}
    if isinstance(first, torch.Tensor):
        return torch.stack(list(items))
    return list(items)


def obs_to_device(obs: Any, device: torch.device | str) -> Any:
    """Recursively move tensor observations to a device."""
    if isinstance(obs, Mapping):
        return {k: obs_to_device(v, device) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.to(device=device)
    return obs


def obs_batch_size(obs: Any) -> int:
    """Return batch size for a tensor or nested observation dictionary."""
    if isinstance(obs, Mapping):
        for value in obs.values():
            return obs_batch_size(value)
        raise ValueError("Cannot infer batch size from an empty observation dict.")
    if isinstance(obs, torch.Tensor):
        return int(obs.shape[0])
    raise TypeError(f"Unsupported observation container: {type(obs)!r}")
