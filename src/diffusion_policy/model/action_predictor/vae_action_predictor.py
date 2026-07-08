"""
VAE-based Action Predictor model adapted from NVIDIA I2SB implementation.

This module defines a lightweight conditional VAE for low-dimensional action prediction.
The encoder produces a Gaussian posterior over latents; the decoder maps latents and
flattened observations back to a future action chunk.
"""

# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Dict, Tuple, Optional
import os

import torch
import torch.nn as nn
import torch.distributions as dist
from torch_ema import ExponentialMovingAverage


def kl_divergence_normal(mean1: torch.Tensor, mean2: torch.Tensor,
                         std1: torch.Tensor, std2: torch.Tensor) -> torch.Tensor:
    """KL divergence between two factorized Gaussians."""
    kl = ((std2 + 1e-9).log() - (std1 + 1e-9).log()
          + (std1.pow(2) + (mean2 - mean1).pow(2)) / (2 * std2.pow(2) + 1e-9)
          - 0.5)
    return kl.sum(-1).mean()


class VAEConditionalMLP(nn.Module):
    """Simple conditional VAE with MLP encoder/decoder.

    Supports two architectures controlled by ``use_dropout``:
    - ``False`` (default): Linear-ReLU-Linear-ReLU  (no Dropout)
    - ``True``:            Linear-ReLU-Dropout-Linear-ReLU-Dropout (with Dropout=0.1)

    ``load_state_dict`` automatically detects which architecture the checkpoint
    was trained with and rebuilds the networks accordingly before loading.

    When ``prev_action_dim > 0``, the decoder receives the previous action chunk
    as additional conditioning (BRIDGER-style temporal feedback).
    """

    def __init__(self, action_dim: int, pred_horizon: int, global_cond_dim: int,
                 latent_dim: int, layer: int, use_dropout: bool = False,
                 prev_action_dim: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.latent_dim = latent_dim
        self._input_dim = global_cond_dim + action_dim * pred_horizon
        self._hidden_dim = layer
        self._decoder_input_dim = global_cond_dim + latent_dim + prev_action_dim
        self._prev_action_dim = prev_action_dim

        self._build_nets(use_dropout)
        self.encoder_mean = nn.Linear(layer, latent_dim)
        self.encoder_logstd = nn.Linear(layer, latent_dim)

    def _build_nets(self, use_dropout: bool):
        """Build encoder_net and decoder_net with or without Dropout."""
        hidden_dim = self._hidden_dim
        if use_dropout:
            self.encoder_net = nn.Sequential(
                nn.Linear(self._input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.decoder_net = nn.Sequential(
                nn.Linear(self._decoder_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.action_dim * self.pred_horizon),
            )
        else:
            self.encoder_net = nn.Sequential(
                nn.Linear(self._input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.decoder_net = nn.Sequential(
                nn.Linear(self._decoder_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim * self.pred_horizon),
            )

    def load_state_dict(self, state_dict, strict=True):
        """自动探测 checkpoint 架构（有无 Dropout，有无 prev_action）并重建网络后加载权重。"""
        has_dropout = any(k.startswith('encoder_net.3.') for k in state_dict)
        # Detect prev_action_dim from decoder input weight shape
        dec_w_key = 'decoder_net.0.weight'
        if dec_w_key in state_dict:
            ckpt_dec_in = state_dict[dec_w_key].shape[1]
            ckpt_prev_dim = ckpt_dec_in - self._decoder_input_dim + self._prev_action_dim
            if ckpt_prev_dim != self._prev_action_dim:
                self._prev_action_dim = ckpt_prev_dim
                self._decoder_input_dim = ckpt_dec_in
        self._build_nets(use_dropout=has_dropout)
        return super().load_state_dict(state_dict, strict=strict)

    def encoder(self, x: torch.Tensor) -> dist.Normal:
        h = self.encoder_net(x)
        mean = self.encoder_mean(h)
        logstd = self.encoder_logstd(h).clamp(min=-6.0, max=6.0)
        std = torch.exp(logstd)
        return dist.Normal(mean, std)

    def decoder(self, x: torch.Tensor, prev_action: torch.Tensor = None) -> torch.Tensor:
        if self._prev_action_dim > 0:
            if prev_action is not None:
                x = torch.cat([x, prev_action], dim=-1)
            else:
                # 第一步推理时无 prev_action，零填充
                zeros = torch.zeros(x.shape[0], self._prev_action_dim,
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, zeros], dim=-1)
        return self.decoder_net(x)


class VAEModel(nn.Module):
    """Minimal VAE wrapper for action prediction."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        obs_dim: int,
        obs_horizon: int,
        latent_dim: int,
        layer: int,
        use_ema: bool = True,
        pretrain: bool = False,
        ckpt_path: Optional[str] = None,
        prev_action_horizon: int = 0,
    ):
        super().__init__()

        self.pred_horizon = action_horizon
        self.action_dim = action_dim
        self.prev_action_horizon = prev_action_horizon
        self._prev_action_dim = action_dim * prev_action_horizon

        self.net = VAEConditionalMLP(
            action_dim=action_dim,
            pred_horizon=action_horizon,
            global_cond_dim=obs_dim * obs_horizon,
            latent_dim=latent_dim,
            layer=layer,
            prev_action_dim=self._prev_action_dim,
        )

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.99) if use_ema else None
        self.anneal_factor = 0.1  # 初始 KL 权重设为 0.1，避免早期坍塌
        self.prior_policy = 'gaussian'

        if pretrain and ckpt_path is not None:
            checkpoint = torch.load(os.path.join(ckpt_path, "model.pt"), map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            if self.ema is not None:
                self.ema.load_state_dict(checkpoint["ema"])

    def sample(self, cond: torch.Tensor, prev_action: torch.Tensor = None,
               x_prior=None, diffuse_step=None) -> torch.Tensor:
        """Sample actions given condition (flattened observations).

        Args:
            cond: [B, obs_dim * obs_horizon] flattened observations
            prev_action: optional [B, prev_action_horizon, action_dim] previous action chunk
        """
        num_sample = cond.shape[0]
        latent = torch.randn((num_sample, self.net.latent_dim), device=cond.device)
        decoder_input = torch.cat([cond, latent], dim=-1)
        if prev_action is not None and self._prev_action_dim > 0:
            prev_flat = prev_action.reshape(num_sample, -1).to(device=cond.device, dtype=cond.dtype)
        else:
            prev_flat = None
        action_flat = self.net.decoder(decoder_input, prev_action=prev_flat)
        return action_flat.reshape(-1, self.net.pred_horizon, self.net.action_dim)

    def get_loss(self, batch_dict: Dict[str, torch.Tensor],
                 loss_args: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        nobs = batch_dict['obs'].to(device).float().flatten(start_dim=1)
        naction = batch_dict['action'].to(device).float()

        # prev_action for decoder conditioning (optional, 零填充 fallback)
        prev_action = batch_dict.get('prev_action', None)
        if self._prev_action_dim > 0:
            if prev_action is not None:
                prev_flat = prev_action.to(device).float().flatten(start_dim=1)
            else:
                prev_flat = torch.zeros(nobs.shape[0], self._prev_action_dim,
                                        device=device, dtype=nobs.dtype)
        else:
            prev_flat = None

        latent_post_dist = self.net.encoder(torch.cat([nobs, naction.flatten(1)], dim=-1))
        latent_post_rsample = latent_post_dist.rsample()
        latent_post_mean = latent_post_dist.mean
        latent_post_std = latent_post_dist.stddev

        latent_prior_mean = torch.zeros_like(latent_post_mean, device=device)
        latent_prior_std = torch.ones_like(latent_post_std, device=device)

        action_rec = self.net.decoder(
            torch.cat([nobs, latent_post_rsample], dim=-1),
            prev_action=prev_flat,
        )
        rec_loss = torch.nn.functional.mse_loss(action_rec, naction.flatten(1)) * 1.0
        kl_loss = self.anneal_factor * kl_divergence_normal(latent_post_mean,
                                                            latent_prior_mean,
                                                            latent_post_std,
                                                            latent_prior_std)

        self.anneal_factor = min(self.anneal_factor + 0.001, 2.0)

        loss = rec_loss + kl_loss
        return loss, {'loss': loss, 'kl_loss': kl_loss, 'rec_loss': rec_loss}

    def log_info(self, writer, log, loss_info: Dict, optimizer, itr: int, num_itr: int):
        writer.add_scalar(itr, 'loss', loss_info['loss'].detach())
        writer.add_scalar(itr, 'kl_loss', loss_info['kl_loss'].detach())
        writer.add_scalar(itr, 'rec_loss', loss_info['rec_loss'].detach())
        log.info(
            "train_it {}/{} | lr:{} | loss:{} | kl_loss:{} | rec_loss:{}".format(
                1 + itr,
                num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.2f}".format(loss_info['loss'].item()),
                "{:+.2f}".format(loss_info['kl_loss'].item()),
                "{:+.2f}".format(loss_info['rec_loss'].item()),
            )
        )

    def save_model(self, ckpt_path: str, itr: int):
        torch.save({
            "net": self.net.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else {},
        }, os.path.join(ckpt_path, "model.pt"))
