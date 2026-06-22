from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply

class EnhancedDiffusionUnetImagePolicy(BaseImagePolicy):
    """
    增强版 Diffusion UNet Image Policy
    
    主要特性：
    1. 支持图像观测输入 (通过 obs_encoder 编码)
    2. 支持从初始动作 (init_action) 开始推理 (Diffusion Refinement)
    """
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
            obs_encoder: nn.Module,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps: int = None,
            init_trajectory_steps: int = None, # 初始轨迹优化的步数
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 256,
            down_dims: tuple = (256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            **kwargs):
        super().__init__()

        # 参数校验
        assert obs_as_global_cond, "Image policy currently only supports obs_as_global_cond=True"
        
        self.shape_meta = shape_meta
        self.noise_scheduler = noise_scheduler
        self.obs_encoder = obs_encoder
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # 获取维度信息
        action_dim = shape_meta['action']['shape'][0]
        self.action_dim = action_dim
        
        # 这里的 obs_feature_dim 是经过 CNN 编码后的特征维度
        obs_feature_dim = obs_encoder.output_shape[0]
        
        # 这里的 input_dim 仅仅是 action_dim，因为 obs 是作为 global_cond 传入的
        input_dim = action_dim

        # 全局条件维度 = obs_feature_dim * n_obs_steps
        global_cond_dim = obs_feature_dim * n_obs_steps

        # 构建 ConditionalUnet1D
        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # 推理步数设置
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 初始轨迹 Refinement 步数 (默认取总步数的 1/4 或至少 10 步)
        if init_trajectory_steps is None:
            init_trajectory_steps = max(num_inference_steps // 4, 10)
        self.init_trajectory_steps = init_trajectory_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data: torch.Tensor, 
            condition_mask: torch.Tensor,
            global_cond: torch.Tensor,
            init_trajectory: torch.Tensor = None,  # 初始动作轨迹
            start_timestep: int = None,
            generator: torch.Generator = None,
            **kwargs
            ) -> torch.Tensor:
        """
        条件采样 (Diffusion Loop)
        
        Args:
            condition_data: 用于 In-painting 的条件数据 (通常是空的或者部分动作)
            condition_mask: 指示哪些部分是已知的
            global_cond: 图像编码后的特征向量
            init_trajectory: [B, T, action_dim] 初始动作猜测
        """
        model = self.model
        scheduler = self.noise_scheduler

        # 1. 初始化轨迹
        if init_trajectory is not None:
            # === 增强模式：从 init_trajectory 开始 ===
            num_inference_steps = self.init_trajectory_steps
            scheduler.set_timesteps(num_inference_steps)
            
            # 确定起始 timestep
            if start_timestep is None:
                start_timestep = scheduler.timesteps[0].item()
            
            # 生成噪声
            noise = torch.randn(
                size=init_trajectory.shape, 
                dtype=init_trajectory.dtype,
                device=init_trajectory.device,
                generator=generator
            )
            
            # 加噪：将 init_trajectory 加噪到 start_timestep 的水平
            start_timestep_tensor = torch.tensor([start_timestep], device=init_trajectory.device)
            trajectory = scheduler.add_noise(
                init_trajectory, 
                noise, 
                start_timestep_tensor
            )
        else:
            # === 标准模式：从纯高斯噪声开始 ===
            num_inference_steps = self.num_inference_steps
            scheduler.set_timesteps(num_inference_steps)
            
            # trajectory shape: [B, T, action_dim]
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator
            )

        # 2. 去噪循环
        for t in scheduler.timesteps:
            # 2.1 应用 In-painting 条件 (如果有)
            # 对于纯 Image Policy，通常 condition_mask 全为 0，这步其实不起作用，除非 fix 了某些动作
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2.2 预测噪声/样本
            # 注意：Image Policy 中 input 只有 action，obs 是 global_cond
            model_output = model(
                sample=trajectory, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )

            # 2.3 调度器步进
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        # 最终强制应用条件
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, 
            obs_dict: Dict[str, torch.Tensor],
            init_action: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs_dict: 包含图像数据的字典
            init_action: [B, horizon, action_dim] 来自 Action Predictor 的初始动作
        """
        assert 'past_action' not in obs_dict # not implemented yet
        
        # 1. 归一化并编码观测数据 (Images -> Features)
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_encoder.output_shape[0]
        
        # 这里的 obs_encoder 会处理多视角的图像并融合
        # input: nobs_dict, output: [B, n_obs_steps, feature_dim]
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...])
        nobs_features = self.obs_encoder(this_nobs)
        
        # Flatten 为 global_cond: [B, n_obs_steps * feature_dim]
        global_cond = nobs_features.reshape(B, -1)

        # 2. 准备初始轨迹 (Init Trajectory)
        init_trajectory = None
        if init_action is not None:
            # 归一化 action
            init_action_normalized = self.normalizer['action'].normalize(init_action)
            
            # 处理 Horizon 不匹配问题 (填充或截断)
            init_horizon = init_action_normalized.shape[1]
            if init_horizon != T:
                if init_horizon < T:
                    # 填充: 重复最后一帧
                    padding = init_action_normalized[:, -1:].expand(-1, T - init_horizon, -1)
                    init_action_normalized = torch.cat([init_action_normalized, padding], dim=1)
                else:
                    # 截断
                    init_action_normalized = init_action_normalized[:, :T]
            
            init_trajectory = init_action_normalized

        # 3. 准备采样所需的形状变量
        # Image Policy 的 condition_data 通常就是 action 的形状
        shape = (B, T, Da)
        cond_data = torch.zeros(size=shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # 4. 执行采样
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            global_cond=global_cond,
            init_trajectory=init_trajectory, # 传入初始轨迹
            **self.kwargs
        )
        
        # 5. 反归一化
        naction_pred = nsample
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 6. 提取需要的动作步数
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 归一化输入
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # 处理观测数据 -> global_cond
        # 只取前 n_obs_steps 帧
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...])
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(batch_size, -1)

        # 采样噪声
        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # 采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()

        # 加噪
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # 预测噪声
        pred = self.model(
            sample=noisy_trajectory, 
            timestep=timesteps, 
            local_cond=None, 
            global_cond=global_cond
        )

        # 计算 Loss
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss