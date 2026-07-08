"""
SI Transformer — 基于 DP Transformer 架构，适配 Stochastic Interpolants

架构:
  Encoder-Only (GPT-style):
    [time_token] [action_0] [action_1] ... [action_T-1]
      ↑ 所有 tokens 做双向 self-attention
    cond (obs) 作为一个额外 token 拼在前面，或通过 cross-attention

  Decoder (cross-attention):
    encoder: obs → memory tokens
    decoder: action tokens + time token → cross-attend to memory

默认使用 encoder-only，最简单、最快。

一次 forward 同时输出 v 和 s（双头）。
"""

from __future__ import annotations

import math, logging
from typing import Union, Optional

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class SITransformer(nn.Module):
    """DP-style Transformer for SI velocity + score prediction.

    n_obs_steps: how many obs frames to use as condition (default: uses all obs)
    causal_attn: True for autoregressive, False for bidirectional (default: False)
    """

    def __init__(
        self,
        input_dim: int,             # action_dim
        output_dim: int,            # action_dim (same)
        horizon: int,               # T — action prediction horizon
        n_obs_steps: int = None,    # obs window size
        obs_dim: int = 0,           # obs condition dim (per step)
        n_layer: int = 8,
        n_head: int = 12,
        n_emb: int = 512,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        n_cond_layers: int = 2,     # encoder layers for cond
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon
        T = horizon
        T_cond = 1  # time token
        obs_as_cond = obs_dim > 0
        if obs_as_cond:
            T_cond += n_obs_steps

        # ── input embedding ──
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ── condition embedding ──
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(obs_dim, n_emb)

        # ── cond encoder → memory ──
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        if n_cond_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn, activation='gelu',
                batch_first=True, norm_first=True,
            )
            self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_cond_layers)
        else:
            self.cond_encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb), nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # ── decoder: action tokens cross-attend to cond ──
        dec_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layer)

        # ── causal mask ──
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            # cross-attention causal mask (action token t can only see obs up to t)
            if obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing='ij')
                mem_mask = t >= (s - 1)  # time token is pos 0
                mem_mask = mem_mask.float().masked_fill(mem_mask == 0, float('-inf')).masked_fill(mem_mask == 1, float(0.0))
                self.register_buffer('memory_mask', mem_mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # ── output heads (v + s) ──
        self.ln_f = nn.LayerNorm(n_emb)
        self.v_head = nn.Linear(n_emb, output_dim)
        self.s_head = nn.Linear(n_emb, output_dim)

        # zero-init heads for stable training start
        nn.init.zeros_(self.v_head.weight); nn.init.zeros_(self.v_head.bias)
        nn.init.zeros_(self.s_head.weight); nn.init.zeros_(self.s_head.bias)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.obs_as_cond = obs_as_cond
        self.input_dim = input_dim

        self.apply(self._init_weights)
        logger.info(f"SITransformer: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")

    def _init_weights(self, module):
        ignore = (nn.Dropout, SinusoidalPosEmb, nn.TransformerEncoderLayer,
                  nn.TransformerDecoderLayer, nn.TransformerEncoder,
                  nn.TransformerDecoder, nn.ModuleList, nn.Mish, nn.Sequential)
        if isinstance(module, (nn.Linear,)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, SITransformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif not isinstance(module, ignore):
            pass  # some parameterless modules slip through

    def forward(self, sample, timestep, global_cond=None, **kwargs):
        """Forward pass (compatible with ConditionalUnet1D interface).

        Args:
            sample: [B, T, input_dim] — action trajectory at time t
            timestep: [B] — SI time (0 to 1)
            global_cond: [B, n_obs_steps, obs_dim] — observation condition

        Returns:
            v: [B, T, output_dim] — velocity
            s: [B, T, output_dim] — score
        """
        B = sample.shape[0]
        device = sample.device

        # ── 1. time embedding ──
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.float32, device=device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(device)
        timestep = timestep.float().expand(B)
        time_emb = self.time_emb(timestep).unsqueeze(1)  # [B, 1, n_emb]

        # ── 2. condition: time + obs → memory ──
        cond_embeddings = time_emb
        if self.obs_as_cond and global_cond is not None:
            obs_emb = self.cond_obs_emb(global_cond)  # [B, To, n_emb]
            cond_embeddings = torch.cat([cond_embeddings, obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        cond_pos = self.cond_pos_emb[:, :tc, :]
        memory = self.drop(cond_embeddings + cond_pos)
        memory = self.cond_encoder(memory)  # [B, T_cond, n_emb]

        # ── 3. action tokens → decoder → cross-attend to memory ──
        action_emb = self.input_emb(sample)  # [B, T, n_emb]
        t = action_emb.shape[1]
        pos = self.pos_emb[:, :t, :]
        x = self.drop(action_emb + pos)

        x = self.decoder(
            tgt=x, memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask,
        )  # [B, T, n_emb]

        # ── 4. output heads ──
        x = self.ln_f(x)
        v = self.v_head(x)
        s = self.s_head(x)
        return v, s, v  # v, s, b (b=v in vs mode)

    def forward_all(self, sample, timestep, global_cond=None):
        """Combined: v, s, b in one forward pass (for both training and inference)."""
        v, s, b = self.forward(sample, timestep, global_cond)
        return v, s, b

    # ── compatibility wrappers for StochasticInterpolants ──
    @property
    def v_net(self):
        return _HeadWrapper(self, 'v')

    @property
    def s_net(self):
        return _HeadWrapper(self, 's')

    @property
    def b_net(self):
        return self.v_net


class _HeadWrapper(nn.Module):
    def __init__(self, transformer, mode='v'):
        super().__init__()
        self.tf = transformer
        self.mode = mode
    def forward(self, x, timestep, global_cond=None, **kw):
        v, s = self.tf(x, timestep, global_cond, **kw)
        return v if self.mode == 'v' else s
