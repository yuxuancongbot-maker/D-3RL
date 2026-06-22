"""Dataset wrapper for D3RL oracle label files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from diffusion_policy.d3rl.scheduler.supervised_trainer import binary_labels_to_soft_targets


class D3RLOracleDataset(Dataset):
    """Load oracle data saved as a torch ``.pt`` dictionary.

    Expected keys:

    - ``obs``: ``[N, T_o, D_o]``
    - ``a_init``: ``[N, T_a, D_a]``
    - ``label``: binary labels, ``0`` sufficient, ``1`` needs refinement
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = torch.load(self.path)
        for key in ("obs", "a_init", "label"):
            if key not in self.data:
                raise KeyError(f"Oracle dataset missing required key: {key}")
        if len(self.data["obs"]) != len(self.data["a_init"]) or len(self.data["obs"]) != len(self.data["label"]):
            raise ValueError("Oracle dataset keys obs/a_init/label must have matching first dimension.")

    def __len__(self) -> int:
        return int(len(self.data["label"]))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        label = self.data["label"][idx].long()
        return {
            "obs": self.data["obs"][idx].float(),
            "a_init": self.data["a_init"][idx].float(),
            "label": label,
            "target": binary_labels_to_soft_targets(label.view(1)).squeeze(0),
        }
