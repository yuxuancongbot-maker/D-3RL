"""
Ada-BRIDGER: Action-Aware Scheduler Network

元认知调度器 (Action-Aware Scheduler):
- 输入: 当前观测 O_t + 初始动作预测 a_init
- 输出: 精炼步数 k ∈ {0, 2, 5, 10}

核心创新:
- 不仅看路(观测)，还看自己写好的"驾驶计划"(初始动作)
- 能有效识别 OOD 场景，提高系统鲁棒性
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActionAwareEncoder(nn.Module):
    """
    动作感知编码器
    
    将观测和初始动作编码为联合特征向量
    """
    
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
        
        # 观测编码器
        obs_input_dim = obs_dim * n_obs_steps
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 动作编码器 - 编码初始动作轨迹
        action_input_dim = action_dim * action_horizon
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # 交叉注意力融合观测和动作
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self, 
        obs: torch.Tensor, 
        init_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, n_obs_steps, obs_dim] 或 [B, n_obs_steps * obs_dim]
            init_action: [B, action_horizon, action_dim] 或 [B, action_horizon * action_dim]
            
        Returns:
            features: [B, hidden_dim] 融合特征
        """
        B = obs.shape[0]
        
        # 展平输入
        if obs.dim() == 3:
            obs = obs.reshape(B, -1)
        if init_action.dim() == 3:
            init_action = init_action.reshape(B, -1)
        
        # 编码
        obs_feat = self.obs_encoder(obs)  # [B, hidden_dim]
        action_feat = self.action_encoder(init_action)  # [B, hidden_dim]
        
        # 交叉注意力: 用动作特征查询观测特征
        obs_feat_seq = obs_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        action_feat_seq = action_feat.unsqueeze(1)  # [B, 1, hidden_dim]
        
        attended_feat, _ = self.cross_attention(
            query=action_feat_seq,
            key=obs_feat_seq,
            value=obs_feat_seq,
        )  # [B, 1, hidden_dim]
        attended_feat = attended_feat.squeeze(1)  # [B, hidden_dim]
        
        # 融合
        fused = torch.cat([attended_feat, action_feat], dim=-1)
        features = self.fusion(fused)
        
        return features


class AdaScheduler(nn.Module):
    """
    自适应调度器 (Ada-Scheduler)

    输入观测和初始动作，输出精炼步数的分类分布。
    精炼步数直接对应 DDIM 推理步数（0, 2, 5, 10），
    不再使用中间 budget level 做二次映射。
    支持离散动作空间 {0, 2, 5, 10}
    """

    # 预定义的精炼步数选项（直接对应 DDIM 推理步数）
    # 保持 [0, 1, 2, 5] 以兼容已训练的 4-way scheduler checkpoint
    REFINEMENT_STEPS = [0, 1, 2, 5]

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        n_obs_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        refinement_steps: List[int] = None,
        # ---- 向后兼容 ----
        step_options: List[int] = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.n_obs_steps = n_obs_steps

        # 可配置的精炼步数（直接就是 DDIM 推理步数）
        # 向后兼容：step_options 在 refinement_steps 未提供时作为回退
        if refinement_steps is not None:
            self.refinement_steps = list(refinement_steps)
        elif step_options is not None:
            self.refinement_steps = list(step_options)
        else:
            self.refinement_steps = list(self.REFINEMENT_STEPS)
        self.num_actions = len(self.refinement_steps)

        # 动作感知编码器
        self.encoder = ActionAwareEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # 策略头 (Actor) - 输出步数分布
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

        # 初始化：让调度器一开始倾向选择最大精炼步数（DDPM 天花板）
        # PPO 训练会逐渐学会在安全状态下降级到 k=0
        with torch.no_grad():
            # 给最大 k 的 logit 加一个正偏置（最后一个 action）
            self.policy_head[-1].bias[-1] = 3.0  # 初始时 ~95% 概率选最大 k

        # 价值头 (Critic) - 输出状态价值
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 注册精炼步数选项为buffer (用于快速查找)
        self.register_buffer(
            'refinement_steps_tensor',
            torch.tensor(self.refinement_steps, dtype=torch.long)
        )

    @property
    def step_options(self) -> List[int]:
        """向后兼容：返回 refinement_steps 的副本。"""
        return list(self.refinement_steps)

    @property
    def step_options_tensor(self) -> torch.Tensor:
        """向后兼容：返回 refinement_steps_tensor。"""
        return self.refinement_steps_tensor
    
    def forward(
        self, 
        obs: torch.Tensor, 
        init_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回策略logits和价值估计
        
        Args:
            obs: [B, n_obs_steps, obs_dim]
            init_action: [B, action_horizon, action_dim]
            
        Returns:
            logits: [B, num_actions] 策略logits
            value: [B, 1] 价值估计
        """
        features = self.encoder(obs, init_action)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value
    
    def get_action_distribution(
        self, 
        obs: torch.Tensor, 
        init_action: torch.Tensor
    ) -> Categorical:
        """
        获取动作分布
        """
        logits, _ = self.forward(obs, init_action)
        return Categorical(logits=logits)
    
    def get_value(
        self, 
        obs: torch.Tensor, 
        init_action: torch.Tensor
    ) -> torch.Tensor:
        """
        获取状态价值
        """
        _, value = self.forward(obs, init_action)
        return value
    
    def select_action(
        self, 
        obs: torch.Tensor, 
        init_action: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择精炼步数
        
        Args:
            obs: [B, n_obs_steps, obs_dim]
            init_action: [B, action_horizon, action_dim]
            deterministic: 是否确定性选择（使用argmax）
            
        Returns:
            steps: 选择的精炼步数
            action_idx: [B] 动作索引
            log_prob: [B] 对数概率
        """
        logits, value = self.forward(obs, init_action)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action_idx = logits.argmax(dim=-1)
        else:
            action_idx = dist.sample()
        
        log_prob = dist.log_prob(action_idx)

        # 转换为实际精炼步数（直接对应 DDIM 推理步数）
        steps = self.refinement_steps_tensor[action_idx]
        
        return steps, action_idx, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        init_action: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的log概率和熵（用于PPO更新）
        
        Returns:
            log_prob: [B] 对数概率
            entropy: [B] 策略熵
            value: [B] 价值估计
        """
        logits, value = self.forward(obs, init_action)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)


class AdaSchedulerForImages(AdaScheduler):
    """Image/multimodal wrapper for :class:`AdaScheduler`.

    The base scheduler still operates on encoded observation features
    ``[B, T, Df]``. This wrapper only adapts raw modality observations from image
    runners, e.g. ``{"image": [B,T,C,H,W], "agent_pos": [B,T,D]}``, through an
    existing ``MultiImageObsEncoder``.
    """

    def __init__(
        self,
        obs_encoder: nn.Module,
        action_dim: int,
        action_horizon: int,
        n_obs_steps: int,
        obs_feature_dim: int = None,
        hidden_dim: int = 256,
        num_layers: int = 2,
        refinement_steps: List[int] = None,
        step_options: List[int] = None,
        freeze_encoder: bool = True,
        **kwargs,
    ):
        if obs_feature_dim is None:
            obs_feature_dim = obs_encoder.output_shape()[0]
        super().__init__(
            obs_dim=obs_feature_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            n_obs_steps=n_obs_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            refinement_steps=refinement_steps,
            step_options=step_options,
        )
        self.obs_encoder_module = obs_encoder
        self.normalizer = None

        if freeze_encoder:
            self.obs_encoder_module.eval()
            for param in self.obs_encoder_module.parameters():
                param.requires_grad = False

    def set_normalizer(self, normalizer):
        """Attach the policy/dataset normalizer used for raw image observations."""
        self.normalizer = normalizer

    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode modality observations ``[B,T,...]`` to features ``[B,T,Df]``."""
        if "obs" in obs_dict and isinstance(obs_dict["obs"], dict):
            obs_dict = obs_dict["obs"]
        if self.normalizer is not None:
            params = getattr(self.normalizer, "params_dict", {})
            if all(key in params for key in obs_dict.keys()):
                obs_dict = self.normalizer.normalize(obs_dict)
            else:
                # Some image source policies normalize encoded obs under an ``obs`` key;
                # keep raw modality tensors unchanged rather than breaking lowdim/image
                # mixed checkpoints. Image pixels from wrappers are already in [0, 1].
                obs_dict = {
                    key: self.normalizer[key].normalize(value) if key in params else value
                    for key, value in obs_dict.items()
                }

        batch_size = None
        n_steps = None
        flat = {}
        for key, value in obs_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() >= 3:
                batch_size = value.shape[0]
                n_steps = min(value.shape[1], self.n_obs_steps)
                flat[key] = value[:, :n_steps].reshape(batch_size * n_steps, *value.shape[2:])
            elif value.dim() == 2:
                batch_size = value.shape[0]
                n_steps = 1
                flat[key] = value
            else:
                flat[key] = value

        if batch_size is None or n_steps is None:
            raise ValueError("Image scheduler received no tensor observations to encode.")

        grad_ctx = torch.no_grad() if not any(p.requires_grad for p in self.obs_encoder_module.parameters()) else torch.enable_grad()
        with grad_ctx:
            features = self.obs_encoder_module(flat)
        return features.reshape(batch_size, n_steps, -1)

    def forward(
        self,
        obs_dict,
        init_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs_dict, torch.Tensor):
            obs_features = obs_dict
        else:
            obs_features = self.encode_obs(obs_dict)
        if obs_features.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Encoded image obs dim={obs_features.shape[-1]} does not match "
                f"scheduler.obs_dim={self.obs_dim}. Align the source/refiner encoder "
                "feature size or set scheduler.obs_feature_dim accordingly."
            )
        return super().forward(obs_features, init_action)
