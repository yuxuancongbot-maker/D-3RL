"""
Ada-BRIDGER Policy: Adaptive Refinement via SDEdit with RL Scheduling

Combines a fast source policy (VAE) with a diffusion refinement model (SDEdit),
using a learned RL scheduler to dynamically allocate denoising steps.

Two variants:
- AdaBridgerPolicy: Base class for inference / oracle label collection
- AdaBridgerPolicyForRL: Extended for PPO training (returns intermediate results)
"""

import time
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.obs_utils import (
    index_obs_batch,
    is_modality_obs,
    slice_obs_steps,
    unwrap_obs,
)


class AdaBridgerPolicy(nn.Module):
    """
    Ada-BRIDGER 自适应精炼策略

    流程:
    1. Source policy (VAE) 产生初始动作 a_init
    2. Scheduler 根据 (obs, a_init) 决定精炼步数 k
    3. 若 k>0, 通过 SDEdit 精炼 a_init → a_refined
    4. 返回最终动作
    """

    def __init__(
        self,
        source_policy: nn.Module,
        refinement_policy: nn.Module,
        scheduler: nn.Module,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        refinement_steps: List[int] = None,
        max_refinement_steps: int = 5,
        scheduler_deterministic: bool = True,
        freeze_backbone: bool = True,
        # ---- 向后兼容 ----
        step_options: List[int] = None,
        ddim_steps: List[int] = None,
    ):
        super().__init__()

        self.source_policy = source_policy
        self.refinement_policy = refinement_policy
        self.scheduler = scheduler

        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        # 精炼步数直接对应 DDIM 推理步数 {0, 2, 5, 10}
        # 向后兼容：如果传入旧式 step_options/ddim_steps，使用 ddim_steps 作为精炼步数
        if refinement_steps is not None:
            self.refinement_steps = list(refinement_steps)
        elif ddim_steps is not None:
            self.refinement_steps = list(ddim_steps)
        elif step_options is not None:
            # 旧式 step_options → 尝试推断 DDIM 步数
            self.refinement_steps = list(step_options)
        else:
            self.refinement_steps = [0, 1, 2, 5]

        self.max_refinement_steps = max_refinement_steps
        self.scheduler_deterministic = scheduler_deterministic

        # 冻结 backbone
        if freeze_backbone:
            if self.source_policy is not None:
                for p in self.source_policy.parameters():
                    p.requires_grad = False
            if self.refinement_policy is not None:
                for p in self.refinement_policy.parameters():
                    p.requires_grad = False

        # 预注册 normalizer（确保 load_state_dict 时 key 存在）
        self._normalizer = LinearNormalizer()

        # 缓存 DDIMScheduler（避免每次 _sdedit_refine 都重建）
        self._ddim_scheduler = None

        # 推理统计
        self._inference_stats: List[Dict] = []
        self._skip_first_timing = True

    @property
    def step_options(self) -> List[int]:
        """向后兼容：返回 refinement_steps 的副本。"""
        return list(self.refinement_steps)

    @property
    def ddim_steps(self) -> List[int]:
        """向后兼容：返回 refinement_steps 的副本。"""
        return list(self.refinement_steps)

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置 normalizer（就地更新，保持模块注册不变）"""
        self._normalizer.load_state_dict(normalizer.state_dict())
        if hasattr(self.scheduler, 'set_normalizer'):
            self.scheduler.set_normalizer(normalizer)

    def state_dict(self, *args, **kwargs):
        """覆盖 state_dict：排除 _normalizer（运行时由 source policy 提供）"""
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith('_normalizer.')}

    def load_state_dict(self, state_dict, strict=True):
        """覆盖 load_state_dict：兼容旧 checkpoint 各种冗余/缺失 key 的情况

        注意：DictOfTensorMixin._load_from_state_dict 会无条件重建 params_dict，
        即使 state_dict 中不含对应 key 也会把已有数据清空。因此需要在调用
        super().load_state_dict 前保存 backbone 状态，之后恢复。
        """
        # 1) 保存 backbone 状态（避免被 super() 递归清空）
        source_sd = {k: v.clone() for k, v in self.source_policy.state_dict().items()}
        refine_sd = {k: v.clone() for k, v in self.refinement_policy.state_dict().items()}

        # 2) 过滤掉旧格式中常见的无关 key
        _ignore_prefixes = (
            '_normalizer.', 'normalizer.',
            'source_policy.', 'refinement_policy.',
        )
        _ignore_keys = ('_dummy_variable',)
        filtered = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in _ignore_prefixes)
            and k not in _ignore_keys
        }

        # 3) 加载 scheduler 权重（strict=False，backbone key 已被过滤）
        missing, unexpected = super().load_state_dict(filtered, strict=False)

        # 4) 恢复 backbone 状态（normalizer + 模型权重）
        self.source_policy.load_state_dict(source_sd, strict=False)
        self.refinement_policy.load_state_dict(refine_sd, strict=False)

        real_missing = [k for k in missing
                        if not any(k.startswith(p) for p in _ignore_prefixes)]
        if strict and real_missing:
            raise RuntimeError(
                f'Error(s) in loading state_dict for {type(self).__name__}:\n'
                f'\tMissing key(s): {real_missing}\n'
            )
        return missing, unexpected

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """每个 episode 开始时重置状态"""
        if hasattr(self.source_policy, 'reset'):
            self.source_policy.reset()
        self._inference_stats = []
        self._skip_first_timing = True
        # BRIDGER feedback: 保存上一步 refined action 供下一步 VAE 使用
        self._last_refined_action = None

    def _is_image_obs(self, obs_dict: Dict[str, torch.Tensor]) -> bool:
        return is_modality_obs(obs_dict) or (
            'obs' not in obs_dict and isinstance(unwrap_obs(obs_dict), dict)
        )

    def _get_scheduler_obs(self, obs_dict: Dict[str, torch.Tensor], source_result=None):
        """Return the observation container expected by the scheduler.

        Lowdim schedulers keep receiving ``Tensor[B, To, Do]`` exactly as before.
        Image source policies may expose encoded features as ``obs_feat``; reuse
        them only when the feature dimension exactly matches ``scheduler.obs_dim``.
        Otherwise, raw image modalities are allowed to fall back only to a
        scheduler that exposes ``encode_obs`` (``AdaSchedulerForImages``).
        """
        expected_dim = getattr(self.scheduler, 'obs_dim', None)
        if source_result is not None and 'obs_feat' in source_result:
            obs_feat = source_result['obs_feat'][:, :self.n_obs_steps]
            if expected_dim is None or obs_feat.shape[-1] == expected_dim:
                return obs_feat
            if not self._is_image_obs(obs_dict):
                raise ValueError(
                    f"Scheduler obs_dim={expected_dim} but source obs_feat dim="
                    f"{obs_feat.shape[-1]} for non-image observations."
                )

        if 'obs' in obs_dict and isinstance(obs_dict['obs'], torch.Tensor):
            obs = obs_dict['obs'][:, :self.n_obs_steps]
            if expected_dim is not None and obs.shape[-1] != expected_dim:
                raise ValueError(
                    f"Scheduler obs_dim={expected_dim} but lowdim obs dim={obs.shape[-1]}."
                )
            return obs

        obs = slice_obs_steps(unwrap_obs(obs_dict), self.n_obs_steps)
        if not hasattr(self.scheduler, 'encode_obs'):
            got = None
            if source_result is not None and 'obs_feat' in source_result:
                got = int(source_result['obs_feat'].shape[-1])
            raise ValueError(
                "Image observations require either source obs_feat with dim matching "
                f"scheduler.obs_dim={expected_dim}, or an image-capable scheduler "
                "with encode_obs(). "
                f"Got source obs_feat dim={got}."
            )
        return obs

    # ------------------------------------------------------------------
    # Core Inference
    # ------------------------------------------------------------------
    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        自适应动作预测

        Args:
            obs_dict: {'obs': [B, T_obs, obs_dim]}
            return_intermediate: 是否返回中间结果（用于 RL 训练）

        Returns:
            result dict 包含 'action' ([B, n_action_steps, action_dim])
            若 return_intermediate=True，还包含 scheduler 的决策信息
        """
        use_cuda_timing = (self.device.type == 'cuda' and torch.cuda.is_available())
        if use_cuda_timing:
            e_start = torch.cuda.Event(enable_timing=True)
            e_source = torch.cuda.Event(enable_timing=True)
            e_scheduler = torch.cuda.Event(enable_timing=True)
            e_refine = torch.cuda.Event(enable_timing=True)
            e_start.record()
        else:
            t_start = time.perf_counter()

        # 1) Source policy 产生初始动作（带 BRIDGER 反馈回路）
        prev_action = getattr(self, '_last_refined_action', None)
        if prev_action is not None:
            # 只取 prev_action_horizon 步（source policy 决定需要多少步）
            pa_horizon = getattr(self.source_policy, 'prev_action_horizon', self.horizon)
            prev_action = prev_action[:, :pa_horizon]
        with torch.no_grad():
            source_obs_dict = obs_dict
            if self._is_image_obs(obs_dict) and 'obs' not in obs_dict:
                # Image source policies in this repo accept either top-level
                # modalities (VAE) or {'obs': modalities} (Transformer).
                source_obs_dict = {'obs': obs_dict}
            try:
                source_result = self.source_policy.predict_action(
                    source_obs_dict, prev_action=prev_action)
            except AssertionError:
                source_result = self.source_policy.predict_action(
                    obs_dict, prev_action=prev_action)
        init_action = source_result['action_pred']  # [B, horizon, action_dim]

        if use_cuda_timing:
            e_source.record()
        else:
            t_source = time.perf_counter()

        # 2) Scheduler 决策
        scheduler_obs = self._get_scheduler_obs(obs_dict, source_result)
        steps, action_idx, log_prob, value = self.scheduler.select_action(
            scheduler_obs,
            init_action,
            deterministic=self.scheduler_deterministic,
        )

        # 每个样本使用各自的 k，避免多环境并行时全部共享 batch[0] 的决策
        steps = steps.to(init_action.device)
        if use_cuda_timing:
            e_scheduler.record()
        else:
            t_scheduler = time.perf_counter()

        # 3) SDEdit 精炼（按样本 k 分组）
        refined_action = init_action.clone()
        unique_steps = torch.unique(steps)
        with torch.no_grad():
            for k_tensor in unique_steps:
                k = int(k_tensor.item())
                if k <= 0:
                    continue
                batch_mask = (steps == k)
                if not torch.any(batch_mask):
                    continue
                idx = torch.nonzero(batch_mask, as_tuple=False).squeeze(-1)
                sub_obs_dict = index_obs_batch(obs_dict, idx)
                sub_init_action = init_action.index_select(0, idx)
                sub_refined = self._sdedit_refine(sub_obs_dict, sub_init_action, k)
                refined_action.index_copy_(0, idx, sub_refined)
        if use_cuda_timing:
            e_refine.record()
            e_refine.synchronize()
            source_time = e_start.elapsed_time(e_source) / 1000.0
            scheduler_time = e_source.elapsed_time(e_scheduler) / 1000.0
            refine_time = e_scheduler.elapsed_time(e_refine) / 1000.0
            total_time = e_start.elapsed_time(e_refine) / 1000.0
        else:
            t_refine = time.perf_counter()
            source_time = t_source - t_start
            scheduler_time = t_scheduler - t_source
            refine_time = t_refine - t_scheduler
            total_time = t_refine - t_start

        # 4) 截取执行部分，保护 NaN/Inf/极端值防止 MuJoCo 崩溃
        action = refined_action[:, :self.n_action_steps]
        action = torch.nan_to_num(action, nan=0.0, posinf=2.0, neginf=-2.0)
        action = torch.clamp(action, -5.0, 5.0)

        # 保存 refined action 供下一步 feedback
        self._last_refined_action = refined_action.detach().clone()

        # 记录推理统计（跳过首个样本，避免 CUDA warmup/JIT 污染均值）
        if self._skip_first_timing:
            self._skip_first_timing = False
        else:
            step_list = [int(v) for v in steps.detach().cpu().tolist()]
            self._inference_stats.append({
                'ks': step_list,
                'source_time': source_time,
                'scheduler_time': scheduler_time,
                'refine_time': refine_time,
                'total_time': total_time,
            })

        result = {'action': action}

        if return_intermediate:
            result.update({
                'init_action': init_action,
                'refinement_steps': steps,
                'scheduler_action_idx': action_idx,
                'scheduler_log_prob': log_prob,
                'scheduler_value': value,
            })

        return result

    def _sdedit_refine_image(
        self,
        obs_dict: Dict[str, torch.Tensor],
        init_action: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """DDIM warm-start refinement for image/multimodal diffusion policies."""
        refine_policy = self.refinement_policy
        noise_scheduler = refine_policy.noise_scheduler
        model = refine_policy.model
        normalizer = refine_policy.normalizer

        B = init_action.shape[0]
        device = init_action.device
        dtype = init_action.dtype
        Da = self.action_dim
        To = self.n_obs_steps
        T = self.horizon

        raw_obs = unwrap_obs(obs_dict)
        params = getattr(normalizer, 'params_dict', {})
        if all(key in params for key in raw_obs.keys()):
            nobs = normalizer.normalize(raw_obs)
        else:
            nobs = {
                key: normalizer[key].normalize(value) if key in params else value
                for key, value in raw_obs.items()
            }
        naction_init = normalizer['action'].normalize(init_action)

        use_global_cond = getattr(refine_policy, 'obs_as_global_cond', True)
        local_cond = None
        global_cond = None

        if use_global_cond:
            this_nobs = {
                key: value[:, :To].reshape(-1, *value.shape[2:])
                for key, value in nobs.items()
            }
            nobs_features = refine_policy.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            trajectory_init = naction_init
            condition_data = trajectory_init.clone()
            condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
        else:
            this_nobs = {
                key: value.reshape(-1, *value.shape[2:])
                for key, value in nobs.items()
            }
            nobs_features = refine_policy.obs_encoder(this_nobs).reshape(B, T, -1)
            trajectory_init = torch.cat([naction_init, nobs_features], dim=-1)
            condition_data = trajectory_init.clone()
            condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
            condition_mask[:, :To, Da:] = True

        num_inference_steps = int(k)
        if num_inference_steps <= 0:
            return init_action

        ddim_scheduler = self._get_ddim_scheduler(noise_scheduler)
        ddim_scheduler.set_timesteps(num_inference_steps)

        noise = torch.randn_like(trajectory_init, dtype=dtype, device=device)
        start_timestep = ddim_scheduler.timesteps[0].item()
        start_timestep_tensor = torch.tensor([start_timestep], device=device).expand(B)
        trajectory = ddim_scheduler.add_noise(trajectory_init, noise, start_timestep_tensor)

        for t in ddim_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = ddim_scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        naction_refined = trajectory[..., :Da]
        return normalizer['action'].unnormalize(naction_refined)

    # ------------------------------------------------------------------
    # SDEdit Refinement
    # ------------------------------------------------------------------
    def _sdedit_refine(
        self,
        obs_dict: Dict[str, torch.Tensor],
        init_action: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        精炼初始动作。

        如果 refinement_policy 是 SI refiner（有 sample 方法），
        直接用 SI ODE 正向积分。否则回退到 DDIM warm-start。
        """
        refine_policy = self.refinement_policy

        if refine_policy is not None and hasattr(refine_policy, 'obs_encoder') and self._is_image_obs(obs_dict):
            return self._sdedit_refine_image(obs_dict, init_action, k)

        # ── SI refiner fast-path ──────────────────────────────────
        if hasattr(refine_policy, 'sample'):
            obs = obs_dict['obs']
            B = obs.shape[0]
            nobs = refine_policy.normalizer['obs'].normalize(obs[:, :self.n_obs_steps])
            # UNet 需 flat [B, Dc]; Transformer 需 [B, To, Do]
            if getattr(refine_policy, '_combined_forward', False):
                global_cond = nobs  # [B, To, Do]
                obs_flat_for_vae = nobs.reshape(B, -1)
            else:
                global_cond = nobs.reshape(B, -1)  # [B, Dc]
                obs_flat_for_vae = global_cond
            # 训练用 VAE.model.sample() → 推理也保持一致（随机采样，不是确定性均值）
            prior_normalized = self.source_policy.model.sample(obs_flat_for_vae)
            refined = refine_policy.sample(
                x0=prior_normalized, cond=global_cond, num_steps=int(k),
            )
            return refine_policy.normalizer['action'].unnormalize(refined)

        # ── DDIM warm-start (legacy) ──────────────────────────────
        noise_scheduler = refine_policy.noise_scheduler
        model = refine_policy.model
        normalizer = refine_policy.normalizer

        B = init_action.shape[0]
        device = init_action.device
        Da = self.action_dim
        Do = self.obs_dim
        To = self.n_obs_steps
        T = self.horizon

        # 归一化
        naction_init = normalizer['action'].normalize(init_action)  # [B, T, Da]

        if 'obs' not in obs_dict or not isinstance(obs_dict['obs'], torch.Tensor):
            raise TypeError(
                "Lowdim DDIM warm-start expects obs_dict['obs'] to be a Tensor. "
                "For image observations, use a refiner policy with obs_encoder."
            )
        obs = obs_dict['obs']
        nobs = normalizer['obs'].normalize(obs)  # [B, To, Do]

        # 可选：将历史动作作为额外条件注入 SDEdit
        # 期望形状 [B, T_past, Da]
        past_action = obs_dict.get('past_action', None)
        npast_action = None
        if past_action is not None:
            if past_action.dim() == 2:
                past_action = past_action.unsqueeze(1)
            if past_action.shape[-1] == Da and past_action.numel() > 0:
                npast_action = normalizer['action'].normalize(past_action)

        # 检测条件模式
        use_global_cond = getattr(refine_policy, 'obs_as_global_cond', False)
        use_local_cond = getattr(refine_policy, 'obs_as_local_cond', False)
        use_inpainting = not use_global_cond and not use_local_cond

        # 准备条件
        local_cond = None
        global_cond = None
        condition_mask = None
        condition_data = None

        if use_global_cond:
            global_cond = nobs[:, :To].reshape(B, -1)
            trajectory_init = naction_init
            condition_data = trajectory_init.clone()
            condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
        elif use_local_cond:
            local_cond = torch.zeros(B, T, Do, device=device, dtype=nobs.dtype)
            local_cond[:, :To] = nobs[:, :To]
            trajectory_init = naction_init
            condition_data = trajectory_init.clone()
            condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
        else:
            # Inpainting: action + obs 拼接
            trajectory_init = torch.zeros(
                B, T, Da + Do, device=device, dtype=nobs.dtype
            )
            trajectory_init[:, :, :Da] = naction_init
            trajectory_init[:, :To, Da:] = nobs[:, :To]

            condition_data = trajectory_init.clone()
            condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
            condition_mask[:, :To, Da:] = True

        # past_action 条件：固定前若干步 action 通道
        if npast_action is not None and condition_data is not None and condition_mask is not None:
            Tpast = min(npast_action.shape[1], T)
            if use_inpainting:
                condition_data[:, :Tpast, :Da] = npast_action[:, :Tpast]
                condition_mask[:, :Tpast, :Da] = True
            else:
                condition_data[:, :Tpast, :] = npast_action[:, :Tpast]
                condition_mask[:, :Tpast, :] = True

        noise = torch.randn_like(trajectory_init)

        # 获取或创建缓存的 DDIM scheduler（仅首次创建）
        ddim_scheduler = self._get_ddim_scheduler(noise_scheduler)

        # 精炼步数直接就是 DDIM 推理步数（不再需要 k → DDIM 映射）
        num_inference_steps = int(k)
        if num_inference_steps <= 0:
            return init_action

        ddim_scheduler.set_timesteps(num_inference_steps)

        # 加噪
        start_timestep = ddim_scheduler.timesteps[0].item()
        start_timestep_tensor = torch.tensor(
            [start_timestep], device=device
        ).expand(B)
        trajectory = ddim_scheduler.add_noise(
            trajectory_init, noise, start_timestep_tensor
        )

        # 去噪循环
        for t in ddim_scheduler.timesteps:
            if condition_mask is not None:
                trajectory[condition_mask] = condition_data[condition_mask]

            model_output = model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = ddim_scheduler.step(
                model_output, t, trajectory
            ).prev_sample

        # 最终 inpainting 写回
        if condition_mask is not None:
            trajectory[condition_mask] = condition_data[condition_mask]

        # 提取 action 并反归一化
        naction_refined = trajectory[..., :Da]
        action_refined = normalizer['action'].unnormalize(naction_refined)

        return action_refined

    # ------------------------------------------------------------------
    # DDIM Scheduler Cache
    # ------------------------------------------------------------------
    def _get_ddim_scheduler(self, noise_scheduler):
        """返回缓存的 DDIMScheduler，首次调用时从训练 scheduler 构建。"""
        if self._ddim_scheduler is not None:
            return self._ddim_scheduler

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        if isinstance(noise_scheduler, DDIMScheduler):
            self._ddim_scheduler = noise_scheduler
        else:
            self._ddim_scheduler = DDIMScheduler(
                num_train_timesteps=noise_scheduler.config.num_train_timesteps,
                beta_start=noise_scheduler.config.beta_start,
                beta_end=noise_scheduler.config.beta_end,
                beta_schedule=noise_scheduler.config.beta_schedule,
                clip_sample=noise_scheduler.config.clip_sample,
                set_alpha_to_one=noise_scheduler.config.get(
                    'set_alpha_to_one', True
                ),
                prediction_type=noise_scheduler.config.prediction_type,
            )
        return self._ddim_scheduler

    # ------------------------------------------------------------------
    # Inference Stats
    # ------------------------------------------------------------------
    def get_inference_stats(self) -> Dict:
        """返回推理统计信息"""
        if not self._inference_stats:
            return {
                'avg_steps': 0,
                'avg_total_time': 0,
                'step_distribution': {},
            }

        ks = []
        for stat in self._inference_stats:
            if 'ks' in stat:
                ks.extend([int(v) for v in stat['ks']])
            elif 'k' in stat:
                ks.append(int(stat['k']))
        total_times = [s['total_time'] for s in self._inference_stats]

        # 步数分布
        counter = collections.Counter(ks)
        total = len(ks)
        step_dist = {k: v / total for k, v in counter.items()}

        return {
            'total_calls': len(self._inference_stats),
            'avg_steps': float(np.mean(ks)),
            'avg_total_time': float(np.mean(total_times)),
            'avg_source_time': float(
                np.mean([s['source_time'] for s in self._inference_stats])
            ),
            'avg_scheduler_time': float(
                np.mean([s['scheduler_time'] for s in self._inference_stats])
            ),
            'avg_refine_time': float(
                np.mean([s['refine_time'] for s in self._inference_stats])
            ),
            'step_distribution': step_dist,
        }

    # ------------------------------------------------------------------
    # Device helper
    # ------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class AdaBridgerPolicyForRL(AdaBridgerPolicy):
    """
    用于 RL 训练的 Ada-BRIDGER 策略

    在基类基础上增加:
    - evaluate_scheduler_actions(): PPO 更新所需的动作评估
    - 训练期间默认 scheduler_deterministic=False（采样探索）
    """

    def evaluate_scheduler_actions(
        self,
        obs: torch.Tensor,
        init_actions: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估 scheduler 动作（PPO 更新时使用）

        Args:
            obs: [N, n_obs_steps, obs_dim] 或 [N, n_obs_steps * obs_dim]
            init_actions: [N, horizon, action_dim]
            action_idx: [N] 之前采样的动作索引

        Returns:
            log_probs: [N]
            entropy: [N]
            values: [N]
        """
        return self.scheduler.evaluate_actions(obs, init_actions, action_idx)
 