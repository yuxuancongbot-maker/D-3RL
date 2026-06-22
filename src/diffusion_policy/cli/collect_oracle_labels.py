"""
Oracle 标签自动收集脚本

自动化流程：
1. 在环境中运行 episodes
2. 对每个状态，分别测试 k=0 (VAE only) 和 k=10 (VAE + SDEdit) 的结果
3. 根据结果自动打标签
4. 保存为 Scheduler 监督预训练的数据集

用法：
python collect_oracle_labels.py \
    --source_checkpoint <vae_checkpoint> \
    --refine_checkpoint <diffusion_checkpoint> \
    --output_dir data/oracle_labels \
    --n_episodes 100
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import argparse
import torch
import torch.nn as nn
import numpy as np
import hydra
import dill
import copy
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import defaultdict

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


def create_robomimic_env(dataset_path, obs_keys, n_obs_steps, n_action_steps, max_steps, abs_action=True):
    """
    创建 robomimic 环境（Can, Lift, Square 等）
    
    返回: (env, rotation_transformer)
    - rotation_transformer: abs_action=True 时用于转换动作格式
    """
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    
    dataset_path = os.path.expanduser(dataset_path)
    
    # 从数据集获取环境元数据
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    
    rotation_transformer = None
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    
    # 初始化观测模态
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})
    
    # 创建基础环境
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    
    # 包装为 RobomimicLowdimWrapper
    wrapped_env = RobomimicLowdimWrapper(
        env=robomimic_env,
        obs_keys=obs_keys,
        init_state=None,
        render_hw=(128, 128),
        render_camera_name='agentview'
    )
    
    # 包装为 MultiStepWrapper
    env = MultiStepWrapper(
        wrapped_env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )
    
    return env, rotation_transformer


def undo_transform_action(action, rotation_transformer):
    """
    将 rotation_6d 格式的动作转换回 axis_angle 格式
    
    Args:
        action: numpy array, shape [..., 10] (pos:3 + rot6d:6 + gripper:1)
        rotation_transformer: RotationTransformer 实例
    
    Returns:
        action: numpy array, shape [..., 7] (pos:3 + axis_angle:3 + gripper:1)
    """
    if rotation_transformer is None:
        return action
    
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)
    
    d_rot = action.shape[-1] - 4  # 6 for rotation_6d
    pos = action[..., :3]
    rot = action[..., 3:3+d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    
    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    
    return uaction


def load_policy(ckpt_path: str, device: str = 'cuda:0'):
    """加载策略（VAE 或 Diffusion）- 使用与 eval.py 一致的方式"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 检测是否为 VAE（需要特殊处理）
    state_dict = payload['state_dicts']['model']
    looks_like_vae = any('net.encoder_net' in k for k in state_dict.keys())
    
    try:
        OmegaConf.set_struct(cfg.policy, False)
    except:
        pass
    
    if looks_like_vae:
        # VAE 需要重新配置 model
        cfg.policy.backend = 'vae'
        cfg.policy.model = {
            '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
            'action_dim': cfg.policy.action_dim,
            'action_horizon': cfg.policy.horizon,
            'obs_dim': cfg.policy.obs_dim,
            'obs_horizon': cfg.policy.n_obs_steps,
            'latent_dim': 32,
            'layer': 256,
            'use_ema': True,
            'pretrain': False,
            'ckpt_path': None,
        }
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(state_dict, strict=False)
    else:
        # Diffusion Policy - 使用与 eval.py 完全一致的方式
        # 通过 workspace 加载，确保所有组件正确初始化
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
            policy = workspace.ema_model
    
    # 检查 normalizer 是否已正确加载
    normalizer_loaded = (
        hasattr(policy, 'normalizer') and 
        policy.normalizer is not None and
        len(policy.normalizer.params_dict) > 0 and
        'action' in policy.normalizer.params_dict
    )
    
    if normalizer_loaded:
        print(f"  已加载 normalizer (keys: {list(policy.normalizer.params_dict.keys())})")
    else:
        print(f"  警告: normalizer 未正确加载!")
    
    policy.eval()
    policy.to(device)
    
    return policy, cfg


def create_test_policy(source_policy, refine_policy, k_value, cfg, device):
    """创建固定 k 值的测试策略"""
    from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler
    
    # 创建一个总是返回固定 k 的 Scheduler
    class FixedKScheduler(nn.Module):
        def __init__(self, k, step_options):
            super().__init__()
            self.k = k
            self.step_options = step_options
            self.k_idx = step_options.index(k) if k in step_options else 0
            
            # 模拟 AdaScheduler 的接口
            self.register_buffer('step_options_tensor', 
                                torch.tensor(step_options, dtype=torch.long))
        
        def select_action(self, obs, init_action, deterministic=True):
            B = obs.shape[0]
            device = obs.device
            
            k_idx = torch.full((B,), self.k_idx, dtype=torch.long, device=device)
            steps = torch.full((B,), self.k, dtype=torch.long, device=device)
            log_prob = torch.zeros(B, device=device)
            value = torch.zeros(B, device=device)
            
            return steps, k_idx, log_prob, value
    
    step_options = [0, 1, 2, 5]
    ddim_steps = [0, 2, 5, 10]
    scheduler = FixedKScheduler(k_value, step_options).to(device)
    
    policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=cfg.policy.horizon,
        obs_dim=cfg.policy.obs_dim,
        action_dim=cfg.policy.action_dim,
        n_action_steps=cfg.policy.n_action_steps,
        n_obs_steps=cfg.policy.n_obs_steps,
        step_options=step_options,
        ddim_steps=ddim_steps,
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    policy.set_normalizer(refine_policy.normalizer)
    policy.to(device)
    policy.eval()
    
    return policy


def collect_trajectory_data(env, policy, max_steps=300):
    """收集单个 trajectory 的数据"""
    trajectory = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'infos': [],
    }
    
    obs = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    
    done = False
    step = 0
    
    while not done and step < max_steps:
        # 准备观测 - 需要 [B, T_obs, obs_dim] 格式
        if isinstance(obs, dict):
            obs_tensor = {}
            for k, v in obs.items():
                v_tensor = torch.from_numpy(v).float().to(policy.device)
                if v_tensor.dim() == 1:
                    v_tensor = v_tensor.unsqueeze(0).unsqueeze(0)
                elif v_tensor.dim() == 2:
                    v_tensor = v_tensor.unsqueeze(0)
                obs_tensor[k] = v_tensor
        else:
            obs_np = np.array(obs)
            obs_t = torch.from_numpy(obs_np).float().to(policy.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0).unsqueeze(0)
            elif obs_t.dim() == 2:
                obs_t = obs_t.unsqueeze(0)
            obs_tensor = {'obs': obs_t}
        
        trajectory['obs'].append(copy.deepcopy(obs))
        
        # 预测动作
        with torch.no_grad():
            result = policy.predict_action(obs_tensor)
        action = result['action'][0].cpu().numpy()
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['dones'].append(done)
        trajectory['infos'].append(info)
        
        step += 1
    
    return trajectory


def run_parallel_comparison(env_runner, source_policy, refine_policy, cfg, device, n_episodes=50):
    """
    并行比较 k=0 和 k=10 的效果
    
    策略：运行完整 episode，记录关键指标
    """
    oracle_data = []
    
    # 创建两个测试策略
    policy_k0 = create_test_policy(source_policy, refine_policy, 0, cfg, device)
    policy_k10 = create_test_policy(source_policy, refine_policy, 10, cfg, device)
    
    print(f"\n开始收集 Oracle 标签数据 ({n_episodes} episodes)...")
    print("策略: 对每个 episode 的每个状态，比较 k=0 和 k=10 的累积回报")
    
    # 运行 k=0 的 episodes
    print("\n[1/2] 运行 k=0 (VAE only) episodes...")
    results_k0 = env_runner.run(policy_k0)
    success_rate_k0 = results_k0.get('test/mean_score', 0)
    print(f"  k=0 成功率: {success_rate_k0:.2%}")
    
    # 运行 k=10 的 episodes
    print("\n[2/2] 运行 k=10 (VAE + SDEdit) episodes...")
    results_k10 = env_runner.run(policy_k10)
    success_rate_k10 = results_k10.get('test/mean_score', 0)
    print(f"  k=10 成功率: {success_rate_k10:.2%}")
    
    return {
        'k0_success_rate': success_rate_k0,
        'k10_success_rate': success_rate_k10,
        'k0_results': results_k0,
        'k10_results': results_k10,
    }


def collect_step_level_labels(env, source_policy, refine_policy, cfg, device, 
                               n_episodes=100, max_steps=300, rotation_transformer=None):
    """
    收集 step 级别的 Oracle 标签
    
    使用 MultiStepWrapper 环境：
    - reset() 返回 (n_obs_steps, obs_dim) 的观测
    - step() 接受 (n_action_steps, action_dim) 的动作，返回 (n_obs_steps, obs_dim) 的观测
    
    Args:
        rotation_transformer: robomimic abs_action 时用于转换 rotation_6d -> axis_angle
    
    方法：
    1. 用 k=10 运行 episode（更稳定）
    2. 同时记录 k=0 会产生的动作
    3. 比较两者的动作差异
    """
    oracle_samples = []
    stats = defaultdict(int)
    
    # 获取配置参数
    n_obs_steps = getattr(source_policy, 'n_obs_steps', 2)
    horizon = getattr(source_policy, 'horizon', 16)
    n_action_steps = getattr(refine_policy, 'n_action_steps', 8)
    
    print(f"\n收集 step 级别标签 ({n_episodes} episodes)...")
    print(f"  n_obs_steps: {n_obs_steps}, horizon: {horizon}, n_action_steps: {n_action_steps}")
    
    # 检查 normalizer 是否一致
    if hasattr(source_policy, 'normalizer') and source_policy.normalizer is not None:
        src_action_stats = source_policy.normalizer['action'].get_output_stats()
        print(f"  VAE normalizer action range: [{src_action_stats['min'].min().item():.3f}, {src_action_stats['max'].max().item():.3f}]")
    
    if hasattr(refine_policy, 'normalizer') and refine_policy.normalizer is not None:
        ref_action_stats = refine_policy.normalizer['action'].get_output_stats()
        print(f"  Diffusion normalizer action range: [{ref_action_stats['min'].min().item():.3f}, {ref_action_stats['max'].max().item():.3f}]")
    
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        # reset 返回已堆叠的观测: (n_obs_steps, obs_dim)
        obs = env.reset()
        source_policy.reset()
        
        done = False
        step = 0
        ep_success = False
        ep_rewards = []  # 收集所有 reward
        
        while not done and step < max_steps:
            # obs 已经是 (n_obs_steps, obs_dim)
            if isinstance(obs, dict):
                if 'obs' in obs:
                    obs_np = np.array(obs['obs'], dtype=np.float32)
                else:
                    obs_np = np.array(next(iter(obs.values())), dtype=np.float32)
            else:
                obs_np = np.array(obs, dtype=np.float32)

            # PushT keypoints env returns obs+mask concatenation.
            # For lowdim policies, keep only the first half (obs without mask).
            if obs_np.shape[-1] == getattr(source_policy, 'obs_dim', obs_np.shape[-1]) * 2:
                obs_np = obs_np[..., :source_policy.obs_dim]
            
            # 添加 batch 维度: [1, n_obs_steps, obs_dim]
            obs_tensor = {
                'obs': torch.from_numpy(obs_np).float().unsqueeze(0).to(device)
            }
            
            with torch.no_grad():
                # 获取 VAE 动作
                source_result = source_policy.predict_action(obs_tensor)
                action_k0 = source_result['action_pred'][0]  # [horizon, action_dim]
                
                # 获取精修后的动作
                action_k10 = refine_action_sdedit(
                    refine_policy, obs_tensor, action_k0, k=10
                )
            
            # 计算动作差异（作为"需要精修"的信号）
            action_diff = (action_k0 - action_k10).abs().mean().item()
            
            # 执行 k=10 的动作（更安全）- 取前 n_action_steps
            action_to_execute = action_k10[:n_action_steps].cpu().numpy()
            
            # 如果是 abs_action 模式，需要将 rotation_6d 转换回 axis_angle
            if rotation_transformer is not None:
                action_to_execute = undo_transform_action(action_to_execute, rotation_transformer)
            
            # 保存样本
            sample = {
                'obs': obs_tensor['obs'][0].cpu(),  # [n_obs_steps, obs_dim]
                'init_action': action_k0.cpu(),      # [horizon, action_dim]
                'action_diff': action_diff,
                'step_in_episode': step,
            }
            oracle_samples.append(sample)
            
            # 执行 - MultiStepWrapper 的 step 接受 (n_action_steps, action_dim)
            obs, reward, done, info = env.step(action_to_execute)
            ep_rewards.append(reward)
            
            step += 1
        
        # robomimic 使用 max_reward 判断成功：max_reward >= 0.99 认为成功
        max_reward = max(ep_rewards) if ep_rewards else 0.0
        ep_success = (max_reward >= 0.99)
        
        stats['total_episodes'] += 1
        stats['success_episodes'] += int(ep_success)
    
    print(f"\n收集完成!")
    print(f"  总 episodes: {stats['total_episodes']}")
    print(f"  成功 episodes: {stats['success_episodes']}")
    print(f"  总样本数: {len(oracle_samples)}")
    
    return oracle_samples, stats


def refine_action_sdedit(refine_policy, obs_dict, init_action, k=10):
    """
    使用 SDEdit 精修动作
    
    使用 DDIM scheduler 以支持低步数推理
    k 参数控制精修强度：k 越大，添加的噪声越多，精修程度越高
    
    自动检测 diffusion policy 的条件模式：
    - obs_as_global_cond: obs 通过 FiLM 全局调制（robomimic 默认）
    - obs_as_local_cond: obs 通过 local encoder 注入
    - inpainting 模式: obs 与 action 拼接为输入通道（PushT lowdim 默认）
    """
    if k == 0:
        return init_action
    
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    
    scheduler = refine_policy.noise_scheduler
    model = refine_policy.model
    normalizer = refine_policy.normalizer
    
    B = 1
    device = init_action.device
    Da = refine_policy.action_dim
    Do = refine_policy.obs_dim
    To = refine_policy.n_obs_steps
    T = refine_policy.horizon
    
    # 归一化 action 和 obs
    naction_init = normalizer['action'].normalize(init_action.unsqueeze(0))  # (1, T, Da)
    
    if 'obs' in obs_dict:
        obs = obs_dict['obs']
    else:
        obs = next(iter(obs_dict.values()))
    nobs = normalizer['obs'].normalize(obs)  # (1, To, Do)
    
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
        global_cond = nobs[:, :To].reshape(B, -1)  # (B, To*Do)
        # trajectory 仅包含 action
        noise = torch.randn_like(naction_init)
        trajectory_init = naction_init
    elif use_local_cond:
        local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=nobs.dtype)
        local_cond[:, :To] = nobs[:, :To]
        # trajectory 仅包含 action
        noise = torch.randn_like(naction_init)
        trajectory_init = naction_init
    else:
        # Inpainting 模式: action + obs 拼接
        trajectory_init = torch.zeros(size=(B, T, Da + Do), device=device, dtype=nobs.dtype)
        trajectory_init[:, :, :Da] = naction_init
        trajectory_init[:, :To, Da:] = nobs[:, :To]
        
        # condition_mask: obs 部分在前 To 步为已知
        condition_data = trajectory_init.clone()
        condition_mask = torch.zeros_like(trajectory_init, dtype=torch.bool)
        condition_mask[:, :To, Da:] = True
        
        noise = torch.randn_like(trajectory_init)
    
    # 创建 DDIM scheduler（支持低步数推理）
    if not isinstance(scheduler, DDIMScheduler):
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler.config.num_train_timesteps,
            beta_start=scheduler.config.beta_start,
            beta_end=scheduler.config.beta_end,
            beta_schedule=scheduler.config.beta_schedule,
            clip_sample=scheduler.config.clip_sample,
            set_alpha_to_one=scheduler.config.get('set_alpha_to_one', True),
            prediction_type=scheduler.config.prediction_type,
        )
    else:
        ddim_scheduler = scheduler
    
    # 设置推理步数（显式映射）
    _k_to_ddim = {0: 0, 1: 2, 2: 5, 5: 10}
    num_inference_steps = _k_to_ddim.get(k, max(int(k * 2), 1))
    ddim_scheduler.set_timesteps(num_inference_steps)
    
    # 加噪
    start_timestep = ddim_scheduler.timesteps[0].item()
    start_timestep_tensor = torch.tensor([start_timestep], device=device)
    trajectory = ddim_scheduler.add_noise(trajectory_init, noise, start_timestep_tensor)
    
    # 去噪循环
    for t in ddim_scheduler.timesteps:
        # inpainting: 每步写回已知 obs
        if use_inpainting and condition_mask is not None:
            trajectory[condition_mask] = condition_data[condition_mask]
        
        model_output = model(
            sample=trajectory,
            timestep=t,
            local_cond=local_cond,
            global_cond=global_cond
        )
        trajectory = ddim_scheduler.step(model_output, t, trajectory).prev_sample
    
    # inpainting: 最终强制写回已知 obs
    if use_inpainting and condition_mask is not None:
        trajectory[condition_mask] = condition_data[condition_mask]
    
    # 提取 action 部分并反归一化
    naction_refined = trajectory[..., :Da]
    action_refined = normalizer['action'].unnormalize(naction_refined)[0]
    
    return action_refined


def assign_labels(oracle_samples, threshold=0.1):
    """
    根据 action_diff 分配标签
    
    启发式规则：
    - 如果 k=0 和 k=10 的动作差异小，说明 k=0 够用 -> label=0
    - 如果差异大，说明需要精修 -> label=10
    """
    labels = []
    
    # 计算 action_diff 的分布
    diffs = [s['action_diff'] for s in oracle_samples]
    median_diff = np.median(diffs)
    mean_diff = np.mean(diffs)
    
    print(f"\n动作差异统计:")
    print(f"  均值: {mean_diff:.4f}")
    print(f"  中位数: {median_diff:.4f}")
    print(f"  阈值: {threshold}")
    
    for sample in oracle_samples:
        # 用动态阈值（中位数的某个比例）
        dynamic_threshold = max(threshold, median_diff * 0.5)
        
        if sample['action_diff'] < dynamic_threshold:
            label = 0  # VAE 够用
        else:
            label = 3  # 对应 step_options 中 k=5 的索引
        
        sample['label'] = label
        labels.append(label)
    
    # 统计
    label_counts = defaultdict(int)
    for l in labels:
        label_counts[l] += 1
    
    print(f"\n标签分布:")
    step_options = [0, 1, 2, 5]
    for l, c in sorted(label_counts.items()):
        k_value = step_options[l] if l < len(step_options) else l
        print(f"  k={k_value}: {c} ({c/len(labels):.1%})")
    
    return oracle_samples


def create_single_env(cfg, n_obs_steps=2, n_action_steps=8, max_steps=300):
    """
    创建用于数据收集的单个环境
    
    包装为 MultiStepWrapper 以便与 policy 的输出格式兼容
    
    返回: (env, rotation_transformer)
    - rotation_transformer: robomimic abs_action 时用于转换动作格式，否则为 None
    """
    # 检查任务类型并创建对应环境
    task_name = cfg.task.get('name', 'pusht_lowdim')
    rotation_transformer = None
    
    if 'pusht' in task_name:
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        raw_env = PushTKeypointsEnv(
            legacy=True,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            **kp_kwargs
        )
        # 包装为 MultiStepWrapper
        env = MultiStepWrapper(
            raw_env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
    elif any(name in task_name for name in ['can', 'lift', 'square', 'transport', 'toolhang', 'tool_hang']):
        # Robomimic 任务
        dataset_path = cfg.task.get('dataset_path', None)
        if dataset_path is None:
            # 尝试从 env_runner 配置获取
            dataset_path = cfg.task.env_runner.get('dataset_path', None)
        
        if dataset_path is None:
            raise ValueError(f"无法找到 robomimic 数据集路径，请检查配置")
        
        obs_keys = cfg.task.get('obs_keys', ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'])
        abs_action = cfg.task.get('abs_action', True)
        
        env, rotation_transformer = create_robomimic_env(
            dataset_path=dataset_path,
            obs_keys=obs_keys,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_steps=max_steps,
            abs_action=abs_action
        )
    else:
        # 通用方案：尝试从配置创建
        raise NotImplementedError(f"Task {task_name} 暂不支持，请手动添加环境创建逻辑")
    
    return env, rotation_transformer


def main():
    parser = argparse.ArgumentParser(description='Oracle 标签自动收集')
    parser.add_argument('--source_checkpoint', type=str, required=True,
                        help='VAE source policy checkpoint')
    parser.add_argument('--refine_checkpoint', type=str, required=True,
                        help='Diffusion refinement policy checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/oracle_labels',
                        help='输出目录路径')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='收集的 episode 数量')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='每个 episode 的最大步数（默认：pusht=300, robomimic=400）')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='动作差异阈值')
    args = parser.parse_args()
    
    device = args.device
    
    # 加载策略
    print("加载 Source Policy (VAE)...")
    source_policy, source_cfg = load_policy(args.source_checkpoint, device)
    
    print("加载 Refinement Policy (Diffusion)...")
    refine_policy, refine_cfg = load_policy(args.refine_checkpoint, device)
    
    # 如果 VAE 没有有效的 normalizer，使用 Diffusion 的 normalizer
    vae_has_normalizer = (
        hasattr(source_policy, 'normalizer') and 
        source_policy.normalizer is not None and
        len(source_policy.normalizer.params_dict) > 0 and
        'action' in source_policy.normalizer.params_dict
    )
    diff_has_normalizer = (
        hasattr(refine_policy, 'normalizer') and 
        refine_policy.normalizer is not None and
        len(refine_policy.normalizer.params_dict) > 0 and
        'action' in refine_policy.normalizer.params_dict
    )
    
    if not vae_has_normalizer:
        if diff_has_normalizer:
            print("  VAE 没有有效的 normalizer，使用 Diffusion 的 normalizer")
            source_policy.set_normalizer(refine_policy.normalizer)
        else:
            print("  警告: 两个 policy 都没有有效的 normalizer!")
    
    # 获取配置参数
    n_obs_steps = getattr(source_policy, 'n_obs_steps', 2)
    n_action_steps = getattr(refine_policy, 'n_action_steps', 8)
    
    # 确定 max_steps
    task_name = refine_cfg.task.get('name', 'pusht_lowdim')
    if args.max_steps is not None:
        max_steps = args.max_steps
    elif any(name in task_name for name in ['can', 'lift', 'square', 'transport', 'toolhang', 'tool_hang']):
        max_steps = 400  # robomimic 默认
    else:
        max_steps = 300  # pusht 默认
    
    print(f"任务: {task_name}, max_steps: {max_steps}")
    
    # 创建单个环境
    print("创建环境...")
    env, rotation_transformer = create_single_env(
        refine_cfg, 
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps
    )
    
    # 收集 step 级别标签
    oracle_samples, stats = collect_step_level_labels(
        env, source_policy, refine_policy, refine_cfg, device,
        n_episodes=args.n_episodes,
        max_steps=max_steps,
        rotation_transformer=rotation_transformer
    )
    
    # 分配标签
    oracle_samples = assign_labels(oracle_samples, threshold=args.threshold)
    
    # 保存
    output_path = os.path.join(args.output_dir, 'oracle_labels.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_data = {
        'samples': oracle_samples,
        'stats': dict(stats),
        'config': {
            'source_checkpoint': args.source_checkpoint,
            'refine_checkpoint': args.refine_checkpoint,
            'n_episodes': args.n_episodes,
            'threshold': args.threshold,
        }
    }
    
    torch.save(save_data, output_path)
    print(f"\n数据已保存到: {output_path}")
    print(f"样本数量: {len(oracle_samples)}")


if __name__ == '__main__':
    main()
