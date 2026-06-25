"""Action-aware D3RL scheduler.

The scheduler directly predicts actual DDIM warm-start refinement steps from
``{0, 2, 5, 10}``, rather than predicting an abstract budget followed by a
separate step mapping.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActionAwareEncoder(nn.Module):
    """Encode observation history and source action into a joint feature."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        n_obs_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_obs_steps = n_obs_steps

        obs_input_dim = obs_dim * n_obs_steps
        action_input_dim = action_dim * action_horizon

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim

    def forward(self, obs: torch.Tensor, init_action: torch.Tensor) -> torch.Tensor:
        bsz = obs.shape[0]
        if obs.dim() == 3:
            obs = obs.reshape(bsz, -1)
        if init_action.dim() == 3:
            init_action = init_action.reshape(bsz, -1)

        obs_feat = self.obs_encoder(obs)
        action_feat = self.action_encoder(init_action)
        attended_feat, _ = self.cross_attention(
            query=action_feat.unsqueeze(1),
            key=obs_feat.unsqueeze(1),
            value=obs_feat.unsqueeze(1),
        )
        return self.fusion(torch.cat([attended_feat.squeeze(1), action_feat], dim=-1))


class D3RLScheduler(nn.Module):
    """Scheduler over actual refinement step choices."""

    DEFAULT_REFINEMENT_STEPS = [0, 1, 2, 5]

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        n_obs_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        refinement_steps: List[int] | None = None,
        initial_zero_bias: float = 3.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_obs_steps = n_obs_steps
        self.refinement_steps = refinement_steps or list(self.DEFAULT_REFINEMENT_STEPS)
        self.num_actions = len(self.refinement_steps)

        self.encoder = ActionAwareEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )
        with torch.no_grad():
            self.policy_head[-1].bias[0] = initial_zero_bias

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "refinement_steps_tensor",
            torch.tensor(self.refinement_steps, dtype=torch.long),
        )

    def forward(self, obs: torch.Tensor, init_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs, init_action)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action_distribution(self, obs: torch.Tensor, init_action: torch.Tensor) -> Categorical:
        logits, _ = self.forward(obs, init_action)
        return Categorical(logits=logits)

    def select_action(
        self,
        obs: torch.Tensor,
        init_action: torch.Tensor,
        deterministic: bool = False,
    ):
        logits, value = self.forward(obs, init_action)
        dist = Categorical(logits=logits)
        action_idx = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action_idx)
        refinement_steps = self.refinement_steps_tensor[action_idx]
        return refinement_steps, action_idx, log_prob, value.squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, init_action: torch.Tensor, action_idx: torch.Tensor):
        logits, value = self.forward(obs, init_action)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_idx)
        entropy = dist.entropy()
        return log_probs, entropy, value.squeeze(-1)
