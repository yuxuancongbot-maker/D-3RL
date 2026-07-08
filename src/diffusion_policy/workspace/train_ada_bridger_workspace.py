"""
Ada-BRIDGER RL Training Workspace

第二阶段: 强化学习训练调度器
1. 加载预训练的 Source Policy 和 Refinement Model (冻结)
2. 使用 PPO + PEGrad 训练调度器
3. 在环境中收集 rollout 数据
4. 评估自适应策略的成功率和控制频率
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import collections
from typing import Dict, List, Optional

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy, AdaBridgerPolicyForRL
from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler
from diffusion_policy.model.ada_bridger.pegrad_optimizer import PEGradOptimizer, PPOWithPEGrad

OmegaConf.register_new_resolver("eval", eval, replace=True)


class RolloutBuffer:
    """
    Rollout 数据缓冲区
    """
    
    def __init__(self):
        self.obs = []
        self.init_actions = []
        self.scheduler_actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.refinement_steps = []
    
    def add(
        self,
        obs: torch.Tensor,
        init_action: torch.Tensor,
        scheduler_action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        refinement_steps: int,
    ):
        self.obs.append(obs)
        self.init_actions.append(init_action)
        self.scheduler_actions.append(scheduler_action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.refinement_steps.append(refinement_steps)
    
    def get(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """获取数据并转换为tensor"""
        return {
            'obs': torch.stack(self.obs).to(device),
            'init_actions': torch.stack(self.init_actions).to(device),
            'actions': torch.stack(self.scheduler_actions).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'values': torch.stack(self.values).to(device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'refinement_steps': torch.tensor(self.refinement_steps, dtype=torch.long, device=device),
        }
    
    def clear(self):
        self.__init__()
    
    def __len__(self):
        return len(self.rewards)


class TrainAdaBridgerWorkspace(BaseWorkspace):
    """
    Ada-BRIDGER RL 训练工作区
    
    训练流程:
    1. 加载预训练的 Source Policy (Action Predictor) 和 Refinement Policy (Diffusion)
    2. 创建 AdaScheduler 并用 PPO + PEGrad 训练
    3. 在 PushT 环境中收集 rollout
    4. 使用 PEGrad 平衡任务奖励和效率惩罚
    """
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir=None):
        OmegaConf.set_struct(cfg, False)
        super().__init__(cfg, output_dir=output_dir)
        
        # 设置随机种子
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # ========== 1. 加载预训练的 Source Policy ==========
        print("Loading Source Policy...")
        source_policy = self._load_pretrained_policy(
            cfg.source_policy_checkpoint,
            cfg.source_policy_config,
        )
        source_policy.eval()
        for param in source_policy.parameters():
            param.requires_grad = False
        if cfg.get('past_action_visible', False) and getattr(source_policy, 'backend', None) == 'vae':
            print("[Warning] past_action_visible=True 但 source backend=vae；当前实现中该开关不会影响策略输入。")
        
        # ========== 2. 加载预训练的 Refinement Policy ==========
        print("Loading Refinement Policy...")
        refinement_policy = self._load_pretrained_policy(
            cfg.refinement_policy_checkpoint,
            cfg.refinement_policy_config,
        )
        refinement_policy.eval()
        for param in refinement_policy.parameters():
            param.requires_grad = False
        
        # ========== 3. 创建 AdaScheduler ==========
        print("Creating Ada-Scheduler...")
        # 统一的精炼步数配置（直接对应 DDIM 推理步数）
        refinement_steps = list(cfg.scheduler.get(
            'refinement_steps',
            cfg.scheduler.get('step_options', [0, 1, 2, 5])
        ))
        # 向后兼容：如果旧 config 同时提供 step_options/ddim_steps 且无 refinement_steps，
        # 使用 ddim_steps 作为实际的精炼步数
        if 'refinement_steps' not in cfg.scheduler and 'ddim_steps' in cfg.scheduler:
            refinement_steps = list(cfg.scheduler.ddim_steps)

        if cfg.scheduler.max_refinement_steps not in refinement_steps:
            raise ValueError(
                f"scheduler.max_refinement_steps={cfg.scheduler.max_refinement_steps} "
                f"不在 scheduler.refinement_steps={refinement_steps} 中"
            )

        scheduler = AdaScheduler(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            action_horizon=cfg.horizon,
            n_obs_steps=cfg.n_obs_steps,
            hidden_dim=cfg.scheduler.hidden_dim,
            num_layers=cfg.scheduler.num_layers,
            refinement_steps=refinement_steps,
        )

        # 加载预训练的 Scheduler（Stage 3 输出）
        pretrain_ckpt = cfg.scheduler.get('pretrain_checkpoint', None)
        if pretrain_ckpt is not None and str(pretrain_ckpt).lower() != 'null':
            if not os.path.exists(pretrain_ckpt):
                raise FileNotFoundError(
                    f"Pretrain checkpoint not found: {pretrain_ckpt}\n"
                    f"如果想从零开始训练，请设置 scheduler.pretrain_checkpoint=null"
                )
            print(f"Loading pretrained scheduler from: {pretrain_ckpt}")
            pretrain_data = torch.load(pretrain_ckpt, map_location=cfg.training.device)
            scheduler.load_state_dict(pretrain_data['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler with val_acc={pretrain_data.get('val_acc', 'N/A')}")
        else:
            print("  No pretrained scheduler, training from scratch")

        self._max_refinement_steps = cfg.scheduler.max_refinement_steps
        self._max_step_action_idx = (
            refinement_steps.index(self._max_refinement_steps)
            if self._max_refinement_steps in refinement_steps
            else None
        )

        # ----- KL 正则化：保存 Stage 3 调度器的冻结副本 -----
        stage4_cfg = cfg.get('stage4', {})
        reg_cfg = stage4_cfg.get('regularization', {})
        self._kl_to_pretrained = reg_cfg.get('kl_to_pretrained', False)
        self._kl_coef = reg_cfg.get('kl_coef', 0.05)
        self._frozen_pretrained_scheduler = None
        if self._kl_to_pretrained:
            self._frozen_pretrained_scheduler = copy.deepcopy(scheduler)
            self._frozen_pretrained_scheduler.eval()
            for p in self._frozen_pretrained_scheduler.parameters():
                p.requires_grad = False
            print("  ✓ Created frozen copy of pretrained scheduler for KL regularization")

        # ----- Cost warmup -----
        reward_cfg = stage4_cfg.get('reward', {})
        self._cost_coef_target = reward_cfg.get('cost_coef_target', 0.05)
        self._task_only_epochs = reward_cfg.get('task_only_epochs', 10)
        self._cost_warmup_epochs = reward_cfg.get('cost_warmup_epochs', 20)

        # ----- Success guard -----
        guard_cfg = stage4_cfg.get('guard', {})
        self._guard_enable = guard_cfg.get('enable', False)
        self._guard_eval_every = guard_cfg.get('eval_every', 5)
        self._guard_success_tolerance = guard_cfg.get('success_tolerance', 0.05)
        self._guard_rollback_on_drop = guard_cfg.get('rollback_on_drop', True)
        self._best_success = -float('inf')
        self._best_scheduler_state = None
        self._pretrain_success = None

        # ----- Stage 4 模式 -----
        self._stage4_mode = stage4_cfg.get('mode', 'lightweight_ppo')
        
        # ========== 4. 创建 Ada-BRIDGER Policy ==========
        self.policy = AdaBridgerPolicyForRL(
            source_policy=source_policy,
            refinement_policy=refinement_policy,
            scheduler=scheduler,
            horizon=cfg.horizon,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            n_action_steps=cfg.n_action_steps,
            n_obs_steps=cfg.n_obs_steps,
            refinement_steps=refinement_steps,
            max_refinement_steps=cfg.scheduler.max_refinement_steps,
            scheduler_deterministic=False,
            freeze_backbone=True,
        )

        # ========== 5. 创建优化器 ==========
        base_optimizer = torch.optim.Adam(
            scheduler.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )

        if self._stage4_mode == 'lightweight_ppo':
            # lightweight_ppo：标准 Adam + clip PPO，不使用 PEGrad 多目标投影
            self.pegrad_optimizer = PEGradOptimizer(
                optimizer=base_optimizer,
                task_weight=1.0,
                cost_weight=0.0,
                project_cost_to_task=False,
                clip_grad_norm=cfg.ppo.max_grad_norm,
            )
        else:
            self.pegrad_optimizer = PEGradOptimizer(
                optimizer=base_optimizer,
                task_weight=cfg.pegrad.task_weight,
                cost_weight=cfg.pegrad.cost_weight,
                project_cost_to_task=cfg.pegrad.project_cost_to_task,
                clip_grad_norm=cfg.pegrad.clip_grad_norm,
            )

        # ========== 6. 创建 PPO 训练器 ==========
        self.ppo_trainer = PPOWithPEGrad(
            policy=self.policy,
            optimizer=self.pegrad_optimizer,
            clip_epsilon=cfg.ppo.clip_epsilon,
            value_loss_coef=cfg.ppo.value_loss_coef,
            entropy_coef=cfg.ppo.entropy_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            efficiency_coef=cfg.ppo.efficiency_coef,
            max_refinement_steps=cfg.scheduler.max_refinement_steps,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
            n_epochs=cfg.ppo.n_epochs,
            batch_size=cfg.ppo.batch_size,
        )

        # ----- 保存 KL 系数供 run() 使用 -----
        self._scheduler_params = list(scheduler.parameters())
        
        self.global_step = 0
        self.epoch = 0
    
    def _load_pretrained_policy(self, checkpoint_path: str, config_path: Optional[str] = None):
        """加载预训练的策略"""
        import dill
        
        print(f"  Loading from: {checkpoint_path}")
        
        # 加载 checkpoint (dill 不支持 weights_only 参数)
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        
        # 从 checkpoint 获取配置
        cfg = payload['cfg']
        
        print(f"  Checkpoint config: {cfg.policy._target_}")
        
        # 实例化策略
        policy = hydra.utils.instantiate(cfg.policy)
        
        print(f"  Instantiated policy type: {type(policy).__name__}")
        if hasattr(policy, 'backend'):
            print(f"  Policy backend: {policy.backend}")
        if hasattr(policy, 'model'):
            print(f"  Policy model type: {type(policy.model).__name__}")
        
        # 加载权重（允许部分不匹配）
        if 'model' in payload['state_dicts']:
            try:
                policy.load_state_dict(payload['state_dicts']['model'], strict=True)
                print(f"  ✓ Loaded weights successfully (strict mode)")
            except RuntimeError as e:
                print(f"  [Warning] Strict loading failed, trying non-strict mode...")
                error_msg = str(e)
                if len(error_msg) > 500:
                    print(f"  Error (truncated): {error_msg[:500]}...")
                else:
                    print(f"  Error: {error_msg}")
                missing, unexpected = policy.load_state_dict(payload['state_dicts']['model'], strict=False)
                if missing:
                    print(f"  ⚠ Missing keys: {len(missing)}")
                    print(f"    First 5: {list(missing)[:5]}")
                if unexpected:
                    print(f"  ⚠ Unexpected keys: {len(unexpected)}")
                    print(f"    First 5: {list(unexpected)[:5]}")
                print(f"  ✓ Loaded weights with non-strict mode")
        elif 'policy' in payload['state_dicts']:
            try:
                policy.load_state_dict(payload['state_dicts']['policy'], strict=True)
                print(f"  ✓ Loaded weights successfully (strict mode)")
            except RuntimeError as e:
                print(f"  [Warning] Strict loading failed, trying non-strict mode...")
                error_msg = str(e)
                if len(error_msg) > 500:
                    print(f"  Error (truncated): {error_msg[:500]}...")
                else:
                    print(f"  Error: {error_msg}")
                missing, unexpected = policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
                if missing:
                    print(f"  ⚠ Missing keys: {len(missing)}")
                    print(f"    First 5: {list(missing)[:5]}")
                if unexpected:
                    print(f"  ⚠ Unexpected keys: {len(unexpected)}")
                    print(f"    First 5: {list(unexpected)[:5]}")
                print(f"  ✓ Loaded weights with non-strict mode")
        else:
            print(f"  [Error] No 'model' or 'policy' key in state_dicts!")
            print(f"  Available keys: {list(payload['state_dicts'].keys())}")
        
        # 加载 normalizer
        if 'normalizer' in payload['state_dicts']:
            try:
                policy.normalizer.load_state_dict(payload['state_dicts']['normalizer'], strict=True)
                print(f"  ✓ Loaded normalizer successfully")
            except RuntimeError as e:
                print(f"  [Warning] Normalizer strict loading failed, using non-strict mode")
                policy.normalizer.load_state_dict(payload['state_dicts']['normalizer'], strict=False)
                print(f"  ✓ Loaded normalizer with non-strict mode")

        if hasattr(policy, 'device'):
            print(f"  Policy device: {policy.device}")
        
        return policy
    
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # 恢复训练
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)
        
        # ========== 配置环境 ==========
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
        )
        
        # 配置归一化器 (从 source policy 获取)
        normalizer = self.policy.source_policy.normalizer
        self.policy.set_normalizer(normalizer)
        
        # ========== 配置日志 ==========
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        try:
            wandb.config.update({"output_dir": self.output_dir})
        except Exception:
            pass  # wandb 可能不允许更新 output_dir
        
        # Checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        
        # 设备
        device = torch.device(cfg.training.device)
        self.policy.to(device)
        
        # ========== RL 训练循环 ==========
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                # ----- Cost warmup: 计算当前 λ_cost -----
                lambda_cost = self._compute_cost_coef(epoch_idx)
                step_log['train/lambda_cost'] = lambda_cost

                # ========== 收集 Rollouts ==========
                self.policy.scheduler.train()
                rollout_info = self._collect_rollouts(
                    env_runner=env_runner,
                    n_rollouts=cfg.training.n_rollouts_per_epoch,
                    device=device,
                    lambda_cost=lambda_cost,
                )

                step_log.update({
                    'train/mean_reward': np.mean(rollout_info['episode_rewards']),
                    'train/mean_length': np.mean(rollout_info['episode_lengths']),
                    'train/mean_success': np.mean(rollout_info['episode_successes']),
                    'train/mean_steps': np.mean(rollout_info['avg_refinement_steps']),
                })

                # 步数分布
                step_counts = rollout_info['step_distribution']
                for k, v in step_counts.items():
                    step_log[f'train/step_{k}_ratio'] = v

                # ========== PPO 更新 ==========
                if len(rollout_info['buffer']) > 0:
                    buffer_data = rollout_info['buffer'].get(device)

                    # 计算 GAE
                    with torch.no_grad():
                        last_obs = buffer_data['obs'][-1:]
                        last_init_action = buffer_data['init_actions'][-1:]
                        _, _, _, next_value = self.policy.scheduler.select_action(
                            last_obs[:, :cfg.n_obs_steps],
                            last_init_action,
                            deterministic=True,
                        )

                    advantages, returns = self.ppo_trainer.compute_gae(
                        rewards=buffer_data['rewards'],
                        values=buffer_data['values'],
                        dones=buffer_data['dones'],
                        next_value=next_value,
                    )

                    # PPO 更新（含可选的 KL 正则化）
                    update_info = self._ppo_update(
                        buffer_data=buffer_data,
                        advantages=advantages,
                        returns=returns,
                        cfg=cfg,
                    )

                    step_log.update({
                        'train/policy_loss': update_info['policy_loss'],
                        'train/value_loss': update_info['value_loss'],
                        'train/entropy': update_info['entropy'],
                        'train/conflict_ratio': update_info['conflict_ratio'],
                    })

                # ========== 评估 + Success Guard ==========
                if (epoch_idx + 1) % cfg.training.eval_every == 0:
                    self.policy.scheduler.eval()
                    prev_deterministic = self.policy.scheduler_deterministic
                    self.policy.scheduler_deterministic = True
                    eval_log = env_runner.run(self.policy)
                    self.policy.scheduler_deterministic = prev_deterministic
                    step_log.update(eval_log)

                    # 推理统计
                    inference_stats = self.policy.get_inference_stats()
                    step_log['eval/avg_refinement_steps'] = inference_stats.get('avg_steps', 0)
                    if 'avg_total_time' in inference_stats:
                        step_log['eval/avg_inference_time_ms'] = inference_stats['avg_total_time'] * 1000

                    # Success guard
                    eval_success = float(eval_log.get('test/mean_score', 0.0))
                    self._apply_success_guard(eval_success, epoch_idx, step_log)
                
                # ========== 日志 ==========
                step_log['global_step'] = self.global_step
                step_log['epoch'] = self.epoch
                
                wandb.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                
                # ========== Checkpoint ==========
                if (epoch_idx + 1) % cfg.training.checkpoint_every == 0:
                    # 保存最新
                    self.save_checkpoint()

                    # TopK (normalize / → _ in keys for format string compatibility)
                    if 'test/mean_score' in step_log:
                        # str.format() 不支持 / 在 keyword arg 中
                        step_log['test_mean_score'] = step_log['test/mean_score']
                        topk_ckpt_path = topk_manager.get_ckpt_path(step_log)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                
                self.global_step += 1
                self.epoch += 1
                
                # 打印进度
                print(f"Epoch {epoch_idx+1}/{cfg.training.num_epochs} | "
                      f"Reward: {step_log.get('train/mean_reward', 0):.3f} | "
                      f"Success: {step_log.get('train/mean_success', 0):.3f} | "
                      f"Avg Steps: {step_log.get('train/mean_steps', 0):.2f}")
    
    def _collect_rollouts(
        self,
        env_runner,
        n_rollouts: int,
        device: torch.device,
        lambda_cost: float = 0.0,
    ) -> Dict:
        """
        在环境中收集 rollout 数据用于 PPO 训练
        
        策略：使用修改后的 env_runner.run() 逻辑来收集数据
        """
        cfg = self.cfg
        buffer = RolloutBuffer()
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        avg_refinement_steps_list = []
        step_counts = collections.defaultdict(int)
        
        # 获取环境相关参数
        env = env_runner.env
        n_envs = len(env_runner.env_fns)
        
        for rollout_idx in range(n_rollouts):
            self.policy.reset()
            
            # 初始化环境（可能因上次崩溃残留而失败）
            init_fn_dill = env_runner.env_init_fn_dills[rollout_idx % len(env_runner.env_init_fn_dills)]
            try:
                env.call_each('run_dill_function', args_list=[(init_fn_dill,)] * n_envs)
            except Exception:
                env.reset()  # 清除 pending 状态后重试
                env.call_each('run_dill_function', args_list=[(init_fn_dill,)] * n_envs)

            obs = env.reset()
            past_action_for_policy = None
            done_arr = np.zeros(n_envs, dtype=bool)
            episode_length = np.zeros(n_envs, dtype=np.int32)
            episode_steps = [[] for _ in range(n_envs)]
            rollout_data = [[] for _ in range(n_envs)]
            episode_max_reward = [None for _ in range(n_envs)]
            
            while (not np.all(done_arr)) and (np.max(episode_length) < env_runner.max_steps):
                # 准备观测
                # PushT keypoints env 会在 obs 后拼接 visibility mask (obs_dim*2)
                # 需要检测并只取前半部分（真正的 obs）
                raw_obs = obs[:, :env_runner.n_obs_steps].astype(np.float32)
                expected_obs_dim = cfg.obs_dim
                actual_obs_dim = raw_obs.shape[-1]
                if actual_obs_dim == expected_obs_dim * 2:
                    # PushT keypoints: obs + mask 拼接，只取前半
                    raw_obs = raw_obs[..., :expected_obs_dim]
                np_obs_dict = {
                    'obs': raw_obs
                }
                if cfg.get('past_action_visible', False) and (past_action_for_policy is not None):
                    np_obs_dict['past_action'] = past_action_for_policy.astype(np.float32)
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(device=device))
                
                # 使用返回中间结果的方式调用策略
                with torch.no_grad():
                    result = self.policy.predict_action(obs_dict, return_intermediate=True)
                
                # 提取数据
                action = result['action']
                init_action = result['init_action']
                refinement_steps = result['refinement_steps']
                
                # 记录调度器决策（用于后续 PPO 更新）
                for env_idx in range(n_envs):
                    if done_arr[env_idx]:
                        continue

                    rollout_data[env_idx].append({
                        'obs': obs_dict['obs'][env_idx].clone(),
                        'init_action': init_action[env_idx].clone(),
                        'scheduler_action_idx': result['scheduler_action_idx'][env_idx].clone() if result['scheduler_action_idx'].dim() > 0 else result['scheduler_action_idx'].clone(),
                        'scheduler_log_prob': result['scheduler_log_prob'][env_idx].clone() if result['scheduler_log_prob'].dim() > 0 else result['scheduler_log_prob'].clone(),
                        'scheduler_value': result['scheduler_value'][env_idx].clone() if result['scheduler_value'].dim() > 0 else result['scheduler_value'].clone(),
                        'refinement_steps': int(refinement_steps[env_idx].item()) if refinement_steps.dim() > 0 else int(refinement_steps.item()),
                    })
                    
                    # 记录步数
                    step_value = int(refinement_steps[env_idx].item()) if refinement_steps.dim() > 0 else int(refinement_steps.item())
                    episode_steps[env_idx].append(step_value)
                    step_counts[step_value] = step_counts.get(step_value, 0) + 1
                
                # 执行动作
                np_action = action.detach().cpu().numpy()
                action_for_env = np_action[:, env_runner.n_latency_steps:]
                # 为下一步策略输入缓存“策略动作空间”下的历史动作
                past_action_for_policy = action_for_env.copy()
                
                if getattr(env_runner, 'abs_action', False):
                    action_for_env = env_runner.undo_transform_action(action_for_env)

                # 动作安全保护：清洗 NaN/Inf，并做范围裁剪，避免 MuJoCo 数值爆炸
                if not np.isfinite(action_for_env).all():
                    action_for_env = np.nan_to_num(
                        action_for_env,
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0,
                    )
                
                # Step 环境（捕获 MuJoCo 物理崩溃，重置 env 并跳过该 rollout）
                try:
                    obs, reward, done_arr_new, info = env.step(action_for_env)
                except Exception as e:
                    print(f"  [WARN] env.step failed: {e}, resetting all envs")
                    done_arr_new = np.ones(n_envs, dtype=bool)
                    reward = [0.0] * n_envs
                    info = [{}] * n_envs
                    try:
                        obs = env.reset()
                    except Exception:
                        pass  # reset 也可能失败，那就跳过整个 rollout
                done_arr = done_arr | done_arr_new
                episode_length += (~done_arr).astype(np.int32) * action_for_env.shape[1]

                # 记录该 episode 的最大即时奖励（回退用）
                reward_arr = np.asarray(reward)
                if reward_arr.size > 0:
                    for env_idx in range(n_envs):
                        reward_env = reward_arr[env_idx]
                        reward_val = float(np.max(reward_env)) if np.ndim(reward_env) > 0 else float(reward_env)
                        if episode_max_reward[env_idx] is None:
                            episode_max_reward[env_idx] = reward_val
                        else:
                            episode_max_reward[env_idx] = max(episode_max_reward[env_idx], reward_val)
            
            # Episode 结束，获取最终奖励（步骤崩溃时回退到 per-step 奖励）
            try:
                final_rewards = env.call('get_attr', 'reward')
            except Exception:
                final_rewards = None
            gamma = cfg.ppo.gamma

            for env_idx in range(n_envs):
                if final_rewards and (final_rewards[env_idx] is not None) and (len(final_rewards[env_idx]) > 0):
                    max_reward = float(np.max(final_rewards[env_idx]))
                elif episode_max_reward[env_idx] is not None:
                    max_reward = float(episode_max_reward[env_idx])
                else:
                    max_reward = 0.0
                success = float(max_reward > 0.5)

                n_decisions = len(rollout_data[env_idx])
                if n_decisions > 0:
                    for i, data in enumerate(rollout_data[env_idx]):
                        # 任务奖励（reward-to-go）
                        task_reward = (gamma ** (n_decisions - 1 - i)) * max_reward

                        # 成本惩罚：r_total = r_task - λ_cost * cost(k)
                        r_step = int(data['refinement_steps'])
                        cost = r_step / max(self._max_refinement_steps, 1)
                        total_reward = task_reward - lambda_cost * cost

                        is_done = (i == n_decisions - 1)

                        buffer.add(
                            obs=data['obs'],
                            init_action=data['init_action'],
                            scheduler_action=data['scheduler_action_idx'],
                            log_prob=data['scheduler_log_prob'],
                            value=data['scheduler_value'],
                            reward=total_reward,
                            done=is_done,
                            refinement_steps=data['refinement_steps'],
                        )

                episode_rewards.append(max_reward)
                episode_lengths.append(int(episode_length[env_idx]))
                episode_successes.append(success)
                avg_refinement_steps_list.append(np.mean(episode_steps[env_idx]) if episode_steps[env_idx] else 0)
        
        # 计算步数分布
        total_steps = sum(step_counts.values())
        step_distribution = {k: v / total_steps for k, v in step_counts.items()} if total_steps > 0 else {}
        
        return {
            'buffer': buffer,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'avg_refinement_steps': avg_refinement_steps_list,
            'step_distribution': step_distribution,
        }

    # ------------------------------------------------------------------
    # Cost warmup
    # ------------------------------------------------------------------
    def _compute_cost_coef(self, epoch_idx: int) -> float:
        """计算当前 epoch 的 λ_cost（cost warmup）。

        前 task_only_epochs 个 epoch 仅关注任务奖励（λ=0），
        之后在 cost_warmup_epochs 内线性增加到 target。
        """
        if epoch_idx < self._task_only_epochs:
            return 0.0
        progress = min(1.0, (epoch_idx - self._task_only_epochs) / max(1, self._cost_warmup_epochs))
        return self._cost_coef_target * progress

    # ------------------------------------------------------------------
    # PPO update with optional KL regularization
    # ------------------------------------------------------------------
    def _ppo_update(self, buffer_data, advantages, returns, cfg):
        """执行 PPO 更新，可选 KL 到 pretrained scheduler。"""
        if self._stage4_mode == 'lightweight_ppo':
            return self._ppo_update_lightweight(buffer_data, advantages, returns, cfg)
        else:
            return self.ppo_trainer.update(
                obs=buffer_data['obs'],
                init_actions=buffer_data['init_actions'],
                actions=buffer_data['actions'],
                old_log_probs=buffer_data['log_probs'],
                advantages=advantages,
                returns=returns,
                refinement_steps=buffer_data['refinement_steps'],
            )

    def _ppo_update_lightweight(self, buffer_data, advantages, returns, cfg):
        """轻量级 PPO 更新（无 PEGrad，可选 KL 正则化）。"""
        obs = buffer_data['obs']
        init_actions = buffer_data['init_actions']
        actions = buffer_data['actions']
        old_log_probs = buffer_data['log_probs']

        # 标准化 advantage
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_loss = 0.0
        n_updates = 0

        for _ in range(cfg.ppo.n_epochs):
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), cfg.ppo.batch_size):
                end = start + cfg.ppo.batch_size
                idx = indices[start:end]

                batch_obs = obs[idx]
                batch_init = init_actions[idx]
                batch_act = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = adv[idx]
                batch_ret = returns[idx]

                new_log_probs, entropy, values = self.policy.evaluate_scheduler_actions(
                    batch_obs, batch_init, batch_act
                )

                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo.clip_epsilon, 1.0 + cfg.ppo.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_ret)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + cfg.ppo.value_loss_coef * value_loss
                        + cfg.ppo.entropy_coef * entropy_loss)

                # KL 正则化到 pretrained scheduler
                kl_loss_val = 0.0
                if self._kl_to_pretrained and self._frozen_pretrained_scheduler is not None:
                    train_logits, _ = self.policy.scheduler.forward(batch_obs, batch_init)
                    with torch.no_grad():
                        frozen_logits, _ = self._frozen_pretrained_scheduler.forward(batch_obs, batch_init)
                    kl_loss_val = nn.functional.kl_div(
                        nn.functional.log_softmax(train_logits, dim=-1),
                        nn.functional.softmax(frozen_logits, dim=-1),
                        reduction='batchmean',
                    )
                    loss = loss + self._kl_coef * kl_loss_val

                self.pegrad_optimizer.zero_grad()
                loss.backward()
                if cfg.ppo.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self._scheduler_params, cfg.ppo.max_grad_norm
                    )
                self.pegrad_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl_loss += (kl_loss_val.item() if isinstance(kl_loss_val, torch.Tensor) else kl_loss_val)
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'conflict_ratio': total_kl_loss / max(n_updates, 1),  # repurpose as KL loss
        }

    # ------------------------------------------------------------------
    # Success guard
    # ------------------------------------------------------------------
    def _apply_success_guard(self, eval_success: float, epoch_idx: int, step_log: dict):
        """Success guard: 检测成功率下降并回滚。"""
        if not self._guard_enable:
            return

        if self._pretrain_success is None:
            self._pretrain_success = eval_success
            self._best_success = eval_success
            self._best_scheduler_state = {
                k: v.clone() for k, v in self.policy.scheduler.state_dict().items()
            }
            step_log['guard/pretrain_success'] = self._pretrain_success
            return

        step_log['guard/pretrain_success'] = self._pretrain_success
        step_log['guard/best_success'] = self._best_success

        if eval_success > self._best_success:
            self._best_success = eval_success
            self._best_scheduler_state = {
                k: v.clone() for k, v in self.policy.scheduler.state_dict().items()
            }
            print(f"  [Guard] New best success: {eval_success:.4f}")
            return

        drop = self._best_success - eval_success
        if drop > self._guard_success_tolerance and self._guard_rollback_on_drop:
            print(f"  [Guard] Success dropped by {drop:.4f} (> {self._guard_success_tolerance}), "
                  f"rolling back to best (success={self._best_success:.4f})")
            if self._best_scheduler_state is not None:
                self.policy.scheduler.load_state_dict(self._best_scheduler_state)
            step_log['guard/rollback'] = 1
        else:
            step_log['guard/rollback'] = 0


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath('config')),
    config_name='train_ada_bridger_workspace'
)
def main(cfg):
    workspace = TrainAdaBridgerWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
