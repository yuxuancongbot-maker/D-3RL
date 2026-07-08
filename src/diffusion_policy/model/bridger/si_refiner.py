"""
Stochastic Interpolants Refiner — 基于 BRIDGER 的最简 SI sampler

训练: 学习从 VAE prior (x0) 到 expert action (x1) 的速度场 v(x,t)
推理: 确定性 ODE 正向积分 (Euler), 从 x0 走到 x1

用法:
    model = SIRefiner(net=unet, t_min=0.001)
    loss = model.get_loss(x0=vae_action, x1=expert_action, cond=obs)
    refined = model.sample(x0=vae_action, cond=obs, num_steps=5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIRefiner(nn.Module):
    """Stochastic Interpolants 精炼器 (minimal: velocity-only, deterministic ODE).

    training:
        x_t = (1-t) * x0 + t * x1   ← linear interpolant (no noise in minimal version)
        target  v = x1 - x0          ← velocity = constant along linear path
        predict v_hat = net(x_t, t, cond)
        loss = MSE(v_hat, v)

    inference (ODE, Euler):
        x = x0
        for i in 1..N:
            t = i/N
            x += net(x, t, cond) * (1/N)
        return x   ← refined action at t=1
    """

    def __init__(self, net: nn.Module, t_min: float = 0.001):
        super().__init__()
        self.net = net
        self.t_min = t_min

    # ── training ──────────────────────────────────────────────────
    def get_loss(
        self,
        x0: torch.Tensor,   # prior action (VAE output),  [B, T, Da]
        x1: torch.Tensor,   # expert / target action,      [B, T, Da]
        cond: torch.Tensor, # global condition (obs flat),  [B, Dc]
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device

        # 每个样本独立采样 t ∈ [t_min, 1-t_min]
        t = torch.rand(B, device=device) * (1.0 - 2 * self.t_min) + self.t_min
        t_expanded = t.view(B, 1, 1)

        # 线性 interpolant: x_t = (1-t) * x0 + t * x1
        xt = (1 - t_expanded) * x0 + t_expanded * x1

        # 目标速度: v = x1 - x0
        v_target = x1 - x0

        # 预测
        v_pred = self.net(xt, timestep=t, global_cond=cond)

        return F.mse_loss(v_pred, v_target)

    # ── inference (ODE Euler integration) ─────────────────────────
    def sample(
        self,
        x0: torch.Tensor,       # prior action (VAE output),  [B, T, Da]
        cond: torch.Tensor,     # global condition (obs flat), [B, Dc]
        num_steps: int = 5,
    ) -> torch.Tensor:
        """ODE 正向积分: 从 x0 (t≈0) 走到 refined action (t=1).

        dx/dt = v_net(x, t, cond)
        用 Euler 积分, num_steps 步。
        """
        B = x0.shape[0]
        device = x0.device
        dt = 1.0 / num_steps
        x = x0

        for step in range(1, num_steps + 1):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            v = self.net(x, timestep=t, global_cond=cond)
            x = x + v * dt

        return x

    # ── helpers ───────────────────────────────────────────────────
    def to(self, device):
        self.net.to(device)
        return self


def create_si_net(
    input_dim: int,
    global_cond_dim: int,
    down_dims=(256, 512, 512),
    kernel_size=5,
    n_groups=8,
) -> nn.Module:
    """创建 SI velocity network (复用现有 ConditionalUnet1D)。"""
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    return ConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
    )
