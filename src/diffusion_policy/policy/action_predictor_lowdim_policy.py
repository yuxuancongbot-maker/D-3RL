"""
Action Predictor Low-Dimensional Policy
使用交叉注意力Transformer的动作预测策略，用于独立训练
"""

from typing import Dict
import time
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.action_predictor.action_predictor_transformer import ActionPredictorTransformer


class ActionPredictorLowdimPolicy(BaseLowdimPolicy):
    """
    Action Predictor策略
    
    训练时：
        - 输入：上一个action chunk的专家动作序列 + 当前观察
        - 输出：预测的动作序列
        - 损失：预测动作与真实专家动作的MSE
    
    推理时：
        - 输入：上一次的预测动作（或零初始化）+ 当前观察
        - 输出：预测的动作序列
    """
    
    def __init__(
        self,
        model,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        prev_action_horizon: int = None,
        backend: str = 'transformer',
        # 推理参数
        **kwargs
    ):
        super().__init__()

        # In some Hydra setups (esp. when overriding backend at runtime), `model`
        # may still be an OmegaConf DictConfig instead of an instantiated module.
        # Make VAE mode robust by instantiating here.
        if backend == 'vae' and not isinstance(model, nn.Module):
            model = hydra.utils.instantiate(model)

        self.model = model
        self.normalizer = LinearNormalizer()
        self.backend = backend

        # 防御性修正：某些旧 checkpoint / Hydra 配置可能把 backend 写成 transformer，
        # 但实际 model 是 VAEModel。此时强制切回 vae 分支，避免走错推理路径。
        if self.backend != 'vae' and hasattr(self.model, 'sample') and hasattr(self.model, 'get_loss'):
            self.backend = 'vae'
        
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.prev_action_horizon = prev_action_horizon if prev_action_horizon else n_action_steps
        self.kwargs = kwargs
        self._debug_profile_printed = False
        
        # 用于推理时保存上一次的预测动作
        self._prev_action = None
    
    # ========= inference  ============
    def reset(self):
        """重置状态（新episode开始时调用）"""
        self._prev_action = None
    
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        prev_action: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        推理：预测动作序列
        
        Args:
            obs_dict: 必须包含 "obs" key，shape: [B, T_obs, obs_dim]
            prev_action: 可选，上一个action chunk，shape: [B, T_prev, action_dim]
                        如果为None，使用内部保存的上一次预测或零初始化
        
        Returns:
            result: 包含 "action" 和 "action_pred" key
        """
        assert 'obs' in obs_dict
        
        use_cuda_timing = (obs_dict['obs'].is_cuda and torch.cuda.is_available())
        if use_cuda_timing:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e3 = torch.cuda.Event(enable_timing=True)
            e0.record()
        else:
            t0 = time.perf_counter()

        # 归一化观察
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim

        if use_cuda_timing:
            e1.record()
        else:
            t1 = time.perf_counter()
        
        device = self.device
        dtype = self.dtype
        
        if self.backend == 'vae':
            # VAE 采样：obs 条件 + 可选的 prev_action 反馈
            obs = nobs[:, :To]
            cond = obs.reshape(B, -1)

            # 处理 prev_action（BRIDGER 反馈回路）
            nprev_action = None
            if prev_action is not None:
                nprev_action = self.normalizer['action'].normalize(prev_action)
            elif self._prev_action is not None:
                nprev_action = self._prev_action
            if nprev_action is not None:
                nprev_action = nprev_action.to(device=device, dtype=dtype)

            self.model.eval()
            with torch.no_grad():
                naction_pred = self.model.sample(cond=cond, prev_action=nprev_action)
            if use_cuda_timing:
                e2.record()
            else:
                t2 = time.perf_counter()
            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            action = action_pred[:, :self.n_action_steps]
            # 保存 VAE 预测的前 prev_action_horizon 步供下一步使用
            pa_horizon = getattr(self.model, 'prev_action_horizon', 0) or self.prev_action_horizon
            if pa_horizon > 0:
                self._prev_action = naction_pred[:, :pa_horizon].clone()
            else:
                self._prev_action = None
        else:
            # transformer 分支，保留原逻辑
            if prev_action is not None:
                nprev_action = self.normalizer['action'].normalize(prev_action)
            elif self._prev_action is not None:
                nprev_action = self._prev_action
            else:
                nprev_action = torch.zeros(
                    B, self.prev_action_horizon, self.action_dim,
                    device=device, dtype=dtype
                )

            nprev_action = nprev_action.to(device=device, dtype=dtype)
            obs = nobs[:, :To]

            self.model.eval()
            with torch.no_grad():
                naction_pred = self.model(nprev_action, obs)
            if use_cuda_timing:
                e2.record()
            else:
                t2 = time.perf_counter()

            action_pred = self.normalizer['action'].unnormalize(naction_pred)
            action = action_pred[:, :self.n_action_steps]
            self._prev_action = naction_pred.clone()

        if use_cuda_timing:
            e3.record()
            e3.synchronize()
            normalize_ms = e0.elapsed_time(e1)
            model_ms = e1.elapsed_time(e2)
            post_ms = e2.elapsed_time(e3)
        else:
            t3 = time.perf_counter()
            normalize_ms = (t1 - t0) * 1000.0
            model_ms = (t2 - t1) * 1000.0
            post_ms = (t3 - t2) * 1000.0

        if not self._debug_profile_printed:
            try:
                model_param_device = next(self.model.parameters()).device
            except StopIteration:
                model_param_device = 'n/a'
            print(
                f"  [SourceProfile] backend={self.backend}, B={B}, obs_device={obs_dict['obs'].device}, "
                f"model_device={model_param_device}, normalize={normalize_ms:.3f}ms, "
                f"model={model_ms:.3f}ms, post={post_ms:.3f}ms"
            )
            self._debug_profile_printed = True
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'action_pred_normalized': naction_pred
        }
        
        return result
    
    def predict_action_with_condition(
        self,
        obs_dict: Dict[str, torch.Tensor],
        condition_action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        带条件的动作预测（用于联合推理时接收diffusion policy的反馈）
        
        Args:
            obs_dict: 观察
            condition_action: 从diffusion policy获得的动作，作为"历史动作"条件
        
        Returns:
            result: 预测结果
        """
        return self.predict_action(obs_dict, prev_action=condition_action)
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """设置归一化器"""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def load_state_dict(self, state_dict, strict=True):
        """覆盖 load_state_dict：VAE 模式下自动探测 Dropout 架构并重建网络."""
        if self.backend == 'vae' and hasattr(self.model, 'net'):
            # encoder_net.3.weight → Dropout 版；encoder_net.2.weight → 无 Dropout 版
            has_dropout = any(k.startswith('model.net.encoder_net.3.') for k in state_dict)
            self.model.net._build_nets(use_dropout=has_dropout)
        return super().load_state_dict(state_dict, strict=strict)

    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算训练损失
        
        batch需要包含：
            - obs: [B, horizon, obs_dim] - 观察序列
            - action: [B, horizon, action_dim] - 动作序列
            - prev_action: [B, prev_action_horizon, action_dim] - 上一个action chunk
        
        Returns:
            loss: MSE损失
        """
        nbatch = self.normalizer.normalize(batch)

        if self.backend == 'vae':
            # 防御：若仍是 DictConfig（缓存/安装旧包），即时实例化
            if not hasattr(self.model, 'get_loss'):
                to_build = self.model
                # 若是 DictConfig 或 dict，且缺 _target_，用当前维度构造一个 VAEModel 配置
                if isinstance(to_build, (DictConfig, dict)):
                    has_target = False
                    try:
                        has_target = ('_target_' in to_build)
                    except Exception:
                        pass
                    if not has_target:
                        to_build = {
                            '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
                            'action_dim': self.action_dim,
                            'action_horizon': self.horizon,
                            'obs_dim': self.obs_dim,
                            'obs_horizon': self.n_obs_steps,
                            'latent_dim': 32,
                            'layer': 256,
                            'use_ema': True,
                            'pretrain': False,
                            'ckpt_path': None,
                        }
                self.model = hydra.utils.instantiate(to_build)
                if not hasattr(self.model, 'get_loss'):
                    raise RuntimeError(f"VAE model instantiation failed: get_loss missing; model type={type(self.model)}")

            obs = nbatch['obs'][:, :self.n_obs_steps]
            # 确保模型和输入在同一设备
            self.model.to(obs.device)
            pred_horizon = getattr(self.model, 'pred_horizon', self.horizon)
            action = nbatch['action'][:, :pred_horizon]
            # 传递 prev_action（如果可用）
            batch_for_vae = {'obs': obs, 'action': action}
            if 'prev_action' in nbatch:
                batch_for_vae['prev_action'] = nbatch['prev_action']
            loss, _ = self.model.get_loss(
                batch_for_vae,
                loss_args={'prior_policy': 'action'},
                device=obs.device
            )
        else:
            obs = nbatch['obs']
            action = nbatch['action']
            prev_action = nbatch['prev_action']

            obs_input = obs[:, :self.n_obs_steps]
            target_action = action[:, :self.model.pred_horizon]
            pred_action = self.model(prev_action, obs_input)
            loss = F.mse_loss(pred_action, target_action)
        
        return loss
    
    def get_optimizer_groups(self, weight_decay: float = 1e-3):
        """获取优化器参数组"""
        return self.model.get_optim_groups(weight_decay)


class ActionPredictorDatasetWrapper:
    """
    数据集包装器，用于生成Action Predictor训练所需的prev_action
    """
    
    def __init__(
        self,
        base_dataset,
        prev_action_horizon: int,
        n_action_steps: int,
        pad_before: int = 0
    ):
        """
        Args:
            base_dataset: 基础数据集
            prev_action_horizon: 上一个action chunk的长度
            n_action_steps: 每次执行的动作步数
            pad_before: 序列开始前的填充步数
        """
        self.base_dataset = base_dataset
        self.prev_action_horizon = prev_action_horizon
        self.n_action_steps = n_action_steps
        self.pad_before = pad_before
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        获取带有prev_action的样本
        
        在实际场景中，prev_action是时间上在当前样本之前的动作序列
        这里我们通过采样相邻的数据来模拟
        """
        # 获取基础样本
        sample = self.base_dataset[idx]
        
        # 获取action
        action = sample['action']  # [horizon, action_dim]
        
        # 创建prev_action
        # 在训练中，我们使用当前序列之前的动作
        # 简化实现：使用action序列开头的数据作为prev_action（带有适当的shift）
        # 或者用零填充表示episode开始
        
        horizon = action.shape[0]
        action_dim = action.shape[1]
        
        # 计算prev_action
        # 假设action已经包含了足够的历史信息
        # prev_action代表"之前一个action chunk"执行的动作
        # 这里我们使用一个简化策略：使用序列开头的部分
        
        if horizon >= self.prev_action_horizon:
            # 可以直接从action中截取（假设数据集已经提供了足够的历史）
            prev_action = action[:self.prev_action_horizon].clone()
        else:
            # 需要填充
            prev_action = torch.zeros(self.prev_action_horizon, action_dim, dtype=action.dtype)
            prev_action[-horizon:] = action
        
        sample['prev_action'] = prev_action
        
        return sample
    
    def get_normalizer(self, **kwargs):
        """获取归一化器"""
        normalizer = self.base_dataset.get_normalizer(**kwargs)
        # prev_action使用与action相同的归一化
        normalizer['prev_action'] = normalizer['action']
        # 图像数据集的normalizer可能没有obs键，默认用恒等归一化
        if 'obs' not in normalizer.params_dict:
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
    
    def get_validation_dataset(self):
        """获取验证数据集"""
        val_base = self.base_dataset.get_validation_dataset()
        return ActionPredictorDatasetWrapper(
            val_base,
            self.prev_action_horizon,
            self.n_action_steps,
            self.pad_before
        )
