from __future__ import annotations

import torch

from diffusion_policy.dataset.d3rl_oracle_dataset import D3RLOracleDataset
from diffusion_policy.d3rl.oracle.discrepancy import action_l1_discrepancy, binary_oracle_labels
from diffusion_policy.d3rl.scheduler.supervised_trainer import (
    binary_labels_to_soft_targets,
    soft_target_cross_entropy,
)


def test_discrepancy_and_labels():
    source = torch.zeros(3, 4, 2)
    refined = torch.zeros(3, 4, 2)
    refined[1] = 1.0
    refined[2] = 2.0
    disc = action_l1_discrepancy(source, refined)
    assert disc.tolist() == [0.0, 4.0, 8.0]
    labels = binary_oracle_labels(disc, eta=0.5)
    assert labels.tolist() == [0, 1, 1]


def test_soft_targets_and_loss():
    labels = torch.tensor([0, 1])
    targets = binary_labels_to_soft_targets(labels)
    assert torch.allclose(targets[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(targets[1], torch.tensor([0.0, 0.2, 0.3, 0.5]))
    logits = torch.zeros(2, 4)
    loss = soft_target_cross_entropy(logits, targets)
    assert loss.item() > 0


def test_oracle_dataset(tmp_path):
    path = tmp_path / "oracle.pt"
    torch.save(
        {
            "obs": torch.zeros(2, 3, 4),
            "a_init": torch.zeros(2, 5, 2),
            "label": torch.tensor([0, 1]),
        },
        path,
    )
    ds = D3RLOracleDataset(path)
    assert len(ds) == 2
    item = ds[1]
    assert item["obs"].shape == (3, 4)
    assert item["a_init"].shape == (5, 2)
    assert torch.allclose(item["target"], torch.tensor([0.0, 0.2, 0.3, 0.5]))
