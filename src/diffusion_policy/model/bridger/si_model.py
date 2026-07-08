"""
Stochastic Interpolants 模型 (完整迁移自 BRIDGER)

训练: 学习从 prior action (x0) → expert action (x1) 的连续路径
推理: SDE 正向积分 (Euler-Maruyama) 从 x0 走到 x1

核心公式:
  x_t = I(t) * x0 + (1-I(t)) * x1 + γ(t) * z    (interpolant)
  v(x,t) = dx_t/dt  — 速度场 (velocity)
  s(x,t) = ∇log p_t  — score
  b(x,t)  — drift = v − γ̇·γ·s·ε

训练: 三个损失联合优化 (v_loss + s_loss + b_loss)
推理: dx = (b + ε·s)·dt + √(2·ε·dt) · dW  (forward SDE)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage

from diffusion_policy.model.bridger.si_unet import InterpolantsConditionalUnet1D


# ── helpers ──────────────────────────────────────────────────────
def _unsqueeze_xdim(t: torch.Tensor, xdim: list) -> torch.Tensor:
    """将 t [B] 展开为 [B, 1, 1, ...] 以匹配 x 的空间维度。"""
    for _ in range(len(xdim)):
        t = t.unsqueeze(-1)
    return t


# ──────────────────────────────────────────────────────────────────
# StochasticInterpolants
# ──────────────────────────────────────────────────────────────────
class StochasticInterpolants(nn.Module):
    """完整的 SI 模型：三头 UNet + 训练损失 + SDE 采样。

    Args:
        action_dim: 动作维度 Da
        obs_dim: 观测维度 Do
        obs_horizon: 观测历史长度 To
        interpolant_type: 'linear' | 'power3' | 'reverse_power3' | ...
        gamma_type: '(2t(t-1))^0.5' | '2^0.5*t(t-1)' | '(1-t)^2(2t)^0.5'
        epsilon_type: '1-t' | 't(t-1)' | '1-t^2' | '0'
        sde_type: 'vs' (velocity+score) | 'bs' (drift+score)
        beta_max: noise coefficient d in γ-diffusion
        net_kwargs: dict of UNet hyperparams (down_dims, kernel_size, n_groups)
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        obs_horizon: int,
        interpolant_type: str = "power3",
        gamma_type: str = "(2t(t-1))^0.5",
        epsilon_type: str = "1-t",
        sde_type: str = "vs",
        beta_max: float = 0.03,
        net_type: str = "unet",
        net_kwargs: dict | None = None,
    ):
        super().__init__()

        self.interpolant_type = interpolant_type
        self.gamma_type = gamma_type
        self.epsilon_type = epsilon_type
        self.sde_type = sde_type
        self.d = beta_max

        self.t_min = 0.001
        self.gamma_inv_max = 200.0

        # 构建网络
        net_kwargs = net_kwargs or {}
        if net_type == "transformer":
            self.net = self._build_transformer(action_dim, obs_dim, obs_horizon, net_kwargs)
        else:
            global_cond_dim = obs_dim * obs_horizon
            self.net = InterpolantsConditionalUnet1D(
                input_dim=action_dim,
                global_cond_dim=global_cond_dim,
                **net_kwargs,
            )

        # EMA
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=0.75)

        # 是否支持单次前向出 v+s+b（Transformer）还是需要分三次（UNet）
        self._combined_forward = hasattr(self.net, "forward_all")

    @staticmethod
    def _build_transformer(action_dim, obs_dim, obs_horizon, kw):
        from diffusion_policy.model.bridger.si_transformer import SITransformer
        return SITransformer(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=kw.pop("horizon", 16),
            n_obs_steps=obs_horizon,
            obs_dim=obs_dim,
            n_layer=kw.pop("n_layer", 8),
            n_head=kw.pop("n_head", 12),
            n_emb=kw.pop("n_emb", 768),
            n_cond_layers=kw.pop("n_cond_layers", 2),
            p_drop_emb=kw.pop("p_drop_emb", 0.1),
            p_drop_attn=kw.pop("p_drop_attn", 0.1),
            causal_attn=kw.pop("causal_attn", False),
        )

    # ── 数学工具 ──────────────────────────────────────────────────
    def _epsilon(self, t: torch.Tensor) -> torch.Tensor:
        if self.epsilon_type == "1-t":
            return 1.0 - t
        if self.epsilon_type == "t(t-1)":
            return t * (1.0 - t)
        if self.epsilon_type == "1-t^2":
            return 1.0 - t.pow(2)
        if self.epsilon_type == "0":
            return torch.zeros_like(t)
        raise NotImplementedError(f"epsilon_type={self.epsilon_type}")

    def _gamma(self, t: torch.Tensor) -> torch.Tensor:
        if self.gamma_type == "(2t(t-1))^0.5":
            return 1.4142 * torch.sqrt(t * (1.0 - t))
        if self.gamma_type == "2^0.5*t(t-1)":
            return 1.4142 * t * (1.0 - t)
        if self.gamma_type == "(1-t)^2(2t)^0.5":
            return 1.4142 * (1.0 - t).pow(2) * torch.sqrt(t)
        raise NotImplementedError(f"gamma_type={self.gamma_type}")

    def _gamma_der(self, t: torch.Tensor) -> torch.Tensor:
        if self.gamma_type == "(2t(t-1))^0.5":
            return (1.0 - 2.0 * t) / torch.sqrt(2.0 * (t - t.pow(2)) + 1e-4)
        if self.gamma_type == "2^0.5*t(t-1)":
            return 1.4142 * (1.0 - 2.0 * t)
        if self.gamma_type == "(1-t)^2(2t)^0.5":
            return 1.4142 * (
                2.0 * (t - 1.0) * torch.sqrt(t)
                + (1.0 - t).pow(2) / (2.0 * torch.sqrt(t + 1e-4))
            )
        raise NotImplementedError(f"gamma_type={self.gamma_type}")

    def _gamma_inv(self, t: torch.Tensor) -> torch.Tensor:
        g = self._gamma(t)
        return torch.clamp(1.0 / (g + 1e-4), 0.0, self.gamma_inv_max)

    def _interpolant(self, x0, x1, gamma, t):
        """构造 x_t = w_x0(t)·x0 + w_x1(t)·x1 + γ·z

        t 应为 [B, 1, 1, ...] 形状 (由 q_sample 预先 unsqueeze)。
        """
        z = self.d * torch.randn_like(x0)

        if self.interpolant_type == "linear":
            w0, w1 = 1.0 - t, t
        elif self.interpolant_type == "power3":
            w0 = (1.0 - t).pow(3)
            w1 = 1.0 - w0
        elif self.interpolant_type == "reverse_power3":
            w0 = 1.0 - t.pow(3)
            w1 = t.pow(3)
        elif self.interpolant_type == "power4":
            w0 = (1.0 - t).pow(4)
            w1 = 1.0 - w0
        elif self.interpolant_type == "reverse_power4":
            w0 = 1.0 - t.pow(4)
            w1 = t.pow(4)
        else:
            raise NotImplementedError(self.interpolant_type)

        return w0 * x0 + w1 * x1 + gamma * z, z

    def _interpolant_dev(self, x1, x0, t):
        """计算 ∂x_t/∂t"""
        xdim = list(x1.shape[1:])

        if self.interpolant_type == "linear":
            return x1 - x0
        if self.interpolant_type == "power3":
            tr = _unsqueeze_xdim(t, xdim)
            return 3.0 * (1.0 - tr).pow(2) * (x1 - x0)
        if self.interpolant_type == "reverse_power3":
            tr = _unsqueeze_xdim(t, xdim)
            return 3.0 * tr.pow(2) * (x1 - x0)
        if self.interpolant_type == "power4":
            tr = _unsqueeze_xdim(t, xdim)
            return 4.0 * (1.0 - tr).pow(3) * (x1 - x0)
        if self.interpolant_type == "reverse_power4":
            tr = _unsqueeze_xdim(t, xdim)
            return 4.0 * tr.pow(3) * (x1 - x0)
        raise NotImplementedError(self.interpolant_type)

    # ── 训练 ──────────────────────────────────────────────────────
    def q_sample(self, t, x0, x1):
        """从 q(x_t | x0, x1) 采样 (式 11)。"""
        xdim = list(x0.shape[1:])
        t_batch = _unsqueeze_xdim(t.clamp(self.t_min, 1.0 - self.t_min), xdim)
        gamma = _unsqueeze_xdim(self._gamma(t), xdim)
        xt, z = self._interpolant(x0, x1, gamma, t_batch)
        return xt.detach(), z

    def velocity_loss(self, t, xt, x0, x1, cond):
        t = t.clamp(self.t_min, 1.0 - self.t_min)
        partial_t = self._interpolant_dev(x1, x0, t)
        v = self.net.v_net(xt, timestep=t, global_cond=cond)
        p = partial_t.flatten(-2)
        vf = v.flatten(-2)
        return (0.5 * vf.norm(dim=-1).pow(2) - (p * vf).sum(dim=-1)).mean()

    def score_loss(self, t, xt, z, cond):
        t = t.clamp(self.t_min, 1.0 - self.t_min)
        s = self.net.s_net(xt, timestep=t, global_cond=cond)
        sf = s.flatten(-2)
        zf = z.flatten(-2)
        return (0.5 * sf.norm(dim=-1).pow(2) + (zf * sf).sum(dim=-1)).mean()

    def drift_loss(self, t, xt, x0, x1, z, cond):
        t = t.clamp(self.t_min, 1.0 - self.t_min)
        partial_t = self._interpolant_dev(x1, x0, t)
        gamma_der = self._gamma_der(t)
        b = self.net.b_net(xt, timestep=t, global_cond=cond)
        bf = b.flatten(-2)
        pf = partial_t.flatten(-2)
        zf = z.flatten(-2)
        xdim = list(bf.shape[1:])
        gd = _unsqueeze_xdim(gamma_der, xdim)
        return (0.5 * bf.norm(dim=-1).pow(2)
                - ((pf + gd * zf) * bf).sum(dim=-1)).mean()

    def get_loss(
        self, x0, x1, cond, return_info: bool = False
    ):
        """三损失之和。Transformer 用单次 forward_all 避免重复前向。"""
        B = x0.shape[0]
        device = x0.device
        t = torch.rand(B, device=device)

        xt, z = self.q_sample(t, x0, x1)

        if self._combined_forward:
            # Transformer: 一次 forward 拿三个输出
            # 但需要处理 cond 格式：Transformer 要 [B, obs_horizon, obs_dim] 不是 flatten
            v, s, b_pred = self.net.forward_all(xt, t, global_cond=cond)
            # 用相同的 batch forward 结果算三个 loss（避免重复前向）
            # velocity loss 用 v, score loss 用 s, drift loss 用 b_pred
            v_loss = self._compute_v_loss(t, xt, x0, x1, v)
            s_loss = self._compute_s_loss(t, xt, z, s)
            b_loss = self._compute_b_loss(t, xt, x0, x1, z, b_pred)
        else:
            v_loss = self.velocity_loss(t, xt, x0, x1, cond)
            s_loss = self.score_loss(t, xt, z, cond)
            b_loss = self.drift_loss(t, xt, x0, x1, z, cond)

        total = v_loss + s_loss + b_loss
        if return_info:
            return total, {"v_loss": v_loss.item(), "s_loss": s_loss.item(), "b_loss": b_loss.item()}
        return total

    def _compute_v_loss(self, t, xt, x0, x1, v):
        t_c = t.clamp(self.t_min, 1.0 - self.t_min)
        partial_t = self._interpolant_dev(x1, x0, t_c)
        vf = v.flatten(-2)
        pf = partial_t.flatten(-2)
        return (0.5 * vf.norm(dim=-1).pow(2) - (pf * vf).sum(dim=-1)).mean()

    def _compute_s_loss(self, t, xt, z, s):
        sf = s.flatten(-2)
        zf = z.flatten(-2)
        return (0.5 * sf.norm(dim=-1).pow(2) + (zf * sf).sum(dim=-1)).mean()

    def _compute_b_loss(self, t, xt, x0, x1, z, b):
        t_c = t.clamp(self.t_min, 1.0 - self.t_min)
        partial_t = self._interpolant_dev(x1, x0, t_c)
        gamma_der = self._gamma_der(t_c)
        bf = b.flatten(-2)
        pf = partial_t.flatten(-2)
        zf = z.flatten(-2)
        xdim = list(bf.shape[1:])
        gd = _unsqueeze_xdim(gamma_der, xdim)
        return (0.5 * bf.norm(dim=-1).pow(2) - ((pf + gd * zf) * bf).sum(dim=-1)).mean()

    # ── 推理 (SDE 正向积分) ───────────────────────────────────────
    @torch.no_grad()
    def sample(self, x0, cond, num_steps=5, record_traj=False):
        """正向 SDE 积分: 从 prior (t≈0) 走到 target (t=1)。

        Args:
            x0: prior action [B, T, Da]
            cond: global condition [B, Dc]
            num_steps: 积分步数 (k)
        Returns:
            refined action [B, T, Da]
        """
        delta_t = 1.0 / num_steps

        with self.ema.average_parameters():
            if self.sde_type == "vs":
                return self._sde_vs(x0, cond, delta_t, record_traj)
            else:
                return self._sde_bs(x0, cond, delta_t, record_traj)

    def _sde_vs(self, x, cond, delta_t, record_traj=False):
        """Velocity + Score SDE。Transformer 用单次 forward_all 拿 v+s。"""
        B = x.shape[0]
        device = x.device
        n_steps = int(1.0 / delta_t)

        combined = self._combined_forward  # True for Transformer

        for step in range(1, n_steps + 1):
            t_val = step / n_steps
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)

            gamma_t = self._gamma(t)
            dot_gamma_t = self._gamma_der(t)
            eps = self._epsilon(t[0])

            if combined:
                v, s, _ = self.net.forward_all(x, t, global_cond=cond)
            else:
                v = self.net.v_net(x, timestep=t, global_cond=cond)
                s = self.net.s_net(x, timestep=t, global_cond=cond)

            xdim = list(s.shape[1:])
            gamma_inv = _unsqueeze_xdim(self._gamma_inv(t), xdim)
            s = s * gamma_inv

            dgg = dot_gamma_t * gamma_t
            dgg = _unsqueeze_xdim(dgg, xdim)
            b = v - dgg * s * eps

            dW = self.d * torch.randn_like(x)
            noise_scale = delta_t * np.sqrt(2.0 * eps.item())
            score_eps = eps

            x = x + (b + score_eps * s) * delta_t + noise_scale * dW

        return x

    def _sde_bs(self, x, cond, delta_t, record_traj=False):
        """Drift + Score SDE。"""
        B = x.shape[0]
        device = x.device
        n_steps = int(1.0 / delta_t)

        for step in range(1, n_steps + 1):
            t_val = step / n_steps
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)

            eps = self._epsilon(t[0])

            b = self.net.b_net(x, timestep=t, global_cond=cond)
            s = self.net.s_net(x, timestep=t, global_cond=cond)

            xdim = list(s.shape[1:])
            gamma_inv = _unsqueeze_xdim(self._gamma_inv(t), xdim)
            s = s * gamma_inv

            dW = self.d * torch.randn_like(x)
            noise_scale = delta_t * np.sqrt(2.0 * eps.item())
            score_eps = eps

            x = x + (b + score_eps * s) * delta_t + noise_scale * dW

        return x

    # ── 保存 / 加载 ───────────────────────────────────────────────
    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(self, path: str):
        torch.save({"net": self.net.state_dict(), "ema": self.ema.state_dict()}, path)

    def load_checkpoint(self, path: str, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        self.net.load_state_dict(ckpt["net"])
        self.ema.load_state_dict(ckpt["ema"])
