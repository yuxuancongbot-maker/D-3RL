"""
InterpolantsConditionalUnet1D — 三头 UNet wrapper for Stochastic Interpolants

等价于 BRIDGER 的 InterpolantsConditionalUnet1D:
  - v_net: 预测速度场 velocity dx/dt
  - s_net: 预测 score ∇log p_t
  - b_net: 预测 drift b(x,t)

三个子网络共享相同的 ConditionalUnet1D 架构，
每个独立参数。
"""

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class InterpolantsConditionalUnet1D(nn.Module):
    """三头 UNet，分别预测速度场、score、drift。

    Args:
        input_dim: 动作维度 Da
        global_cond_dim: 条件维度 = obs_dim * obs_horizon
        diffusion_step_embed_dim: 时间嵌入维度
        down_dims: 各层通道数，长度决定 UNet 深度
        kernel_size: 卷积核大小
        n_groups: GroupNorm 组数
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        _make = lambda: ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.v_net = _make()
        self.s_net = _make()
        self.b_net = _make()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[SI-UNet] 3 × ConditionalUnet1D, params={n_params / 1e6:.1f}M")
