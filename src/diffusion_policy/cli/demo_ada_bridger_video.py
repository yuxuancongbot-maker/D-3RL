#!/usr/bin/env python
"""
推理演示视频生成脚本 (IROS 2026)

支持三种策略类型：
  1. Ada-BRIDGER  —— 自适应精炼步调度，显示 k 分布和组件耗时
  2. DDPM         —— 标准 100 步去噪扩散
  3. DDIM         —— 加速去噪（可指定步数）

支持 PushT 和 Transport (robomimic) 环境。

用法 (Ada-BRIDGER):
  python demo_ada_bridger_video.py \
      --checkpoint data/outputs/ablation_pusht/full_seed42/checkpoints/latest.ckpt \
      --output_dir data/demo_video --n_episodes 3

用法 (DDPM 扩散策略):
  python demo_ada_bridger_video.py \
      --checkpoint weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
      --output_dir data/demo_video_ddpm

用法 (DDIM — 指定推理步数):
  python demo_ada_bridger_video.py \
      --checkpoint weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
      --output_dir data/demo_video_ddim --steps 8

用法 (Transport Ada-BRIDGER):
  python demo_ada_bridger_video.py \
      --checkpoint data/outputs/ada_bridger_transport/checkpoints/latest.ckpt \
      --output_dir data/demo_video --n_episodes 3
"""

import os
import sys
import pathlib
import argparse
import copy
from collections import Counter

# 服务器无显示器时使用 dummy 驱动
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import torch
import cv2
import dill
import hydra
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply

# ═══════════════════════════════════════════════════════════════
#  环境工厂：根据任务类型创建对应的环境
# ═══════════════════════════════════════════════════════════════

TASK_PUSHT = 'pusht'
TASK_ROBOMIMIC = 'robomimic'  # transport / can / square / lift 等

POLICY_ADA_BRIDGER = 'ada_bridger'
POLICY_DIFFUSION = 'diffusion'   # DDPM / DDIM


def detect_policy_type(cfg):
    """从 checkpoint config 的 _target_ 判断策略类型"""
    target = cfg.get('_target_', '')
    if 'ada_bridger' in target.lower():
        return POLICY_ADA_BRIDGER
    return POLICY_DIFFUSION


def detect_task_type(cfg):
    """从 checkpoint config 自动检测任务类型"""
    task_name = cfg.get('task_name', '')
    if 'pusht' in task_name.lower():
        return TASK_PUSHT
    # transport / can / lift / square 等都走 robomimic
    return TASK_ROBOMIMIC


def create_pusht_env(cfg):
    """创建 PushT 环境（无 VideoRecordingWrapper，直接同步调用）"""
    import pygame  # noqa: F811
    pygame.init()
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    return MultiStepWrapper(
        PushTKeypointsEnv(
            legacy=True,
            keypoint_visible_rate=cfg.get('keypoint_visible_rate', 1.0),
            render_action=True,
            render_size=96,
        ),
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=300,
    )


def create_robomimic_env(cfg, render_hw=(512, 512), render_camera_name='agentview'):
    """
    创建 robomimic 环境（transport / can / square / lift 等）。
    不包裹 VideoRecordingWrapper，我们自行逐帧抓取。
    """
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

    dataset_path = os.path.expanduser(cfg.task.dataset_path)
    obs_keys = list(cfg.task.obs_keys)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    abs_action = cfg.task.get('abs_action', False)
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False

    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,   # 需要 offscreen 才能渲染高清帧
        use_image_obs=False,
    )

    max_steps = cfg.task.env_runner.get('max_steps', 700)
    return MultiStepWrapper(
        RobomimicLowdimWrapper(
            env=robomimic_env,
            obs_keys=obs_keys,
            init_state=None,
            render_hw=render_hw,
            render_camera_name=render_camera_name,
        ),
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=max_steps,
    )


def _undo_transform_action(action, rotation_transformer):
    """
    将 rotation_6d 编码的动作转回 axis_angle 以供 robosuite 执行。
    复刻自 RobomimicLowdimRunner.undo_transform_action。
    """
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)
    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


# ═══════════════════════════════════════════════════════════════
#  HUD 渲染
# ═══════════════════════════════════════════════════════════════

def draw_hud(frame, info, render_size=512):
    """
    在帧上绘制半透明 HUD 信息面板。
    根据 policy_type 自动适配 Ada-BRIDGER 和 DDPM/DDIM 两种布局。
    """
    H, W = frame.shape[:2]
    scale = H / 512.0  # 相对 512 的缩放比

    # ── 字体设置 ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 0.55 * scale
    font_scale_body = 0.45 * scale
    font_scale_large = 1.2 * scale
    thickness = max(1, int(1.5 * scale))
    line_h = int(22 * scale)

    # ── 左上：推理频率（大字） ──
    hz = info.get('hz', 0)
    hz_text = f"{hz:.1f} Hz"
    hz_color = (0, 255, 0) if hz >= 10 else (0, 255, 255) if hz >= 5 else (0, 0, 255)

    # 半透明背景
    overlay = frame.copy()
    box_w = int(200 * scale)
    box_h = int(50 * scale)
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, hz_text, (int(15 * scale), int(42 * scale)),
                font, font_scale_large, hz_color, thickness + 1, cv2.LINE_AA)

    # ── 左侧：时间分解面板 ──
    panel_y_start = int(70 * scale)
    policy_type = info.get('policy_type', POLICY_ADA_BRIDGER)
    if policy_type == POLICY_DIFFUSION:
        sched_name = info.get('scheduler_name', 'DDPM')
        n_steps = info.get('num_inference_steps', '?')
        panel_lines = [
            f"{sched_name}  ({n_steps} steps)",
            f"Denoise:   {info.get('denoise_ms', 0):6.1f} ms",
            f"Total:     {info.get('total_ms', 0):6.1f} ms",
        ]
    else:
        panel_lines = [
            f"refinement_steps = {info.get('k', '?')}",
            f"Source:    {info.get('source_ms', 0):6.1f} ms",
            f"Scheduler: {info.get('scheduler_ms', 0):6.1f} ms",
            f"Refine:    {info.get('refine_ms', 0):6.1f} ms",
            f"Total:     {info.get('total_ms', 0):6.1f} ms",
        ]
    # 背景
    panel_h = len(panel_lines) * line_h + int(12 * scale)
    panel_w = int(230 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, panel_y_start - int(5 * scale)),
                  (8 + panel_w, panel_y_start + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(panel_lines):
        y = panel_y_start + int(15 * scale) + i * line_h
        color = (255, 255, 255) if i > 0 else (0, 255, 255)
        cv2.putText(frame, line, (int(15 * scale), y),
                    font, font_scale_body, color, thickness, cv2.LINE_AA)

    # ── 右上：Episode 进度 ──
    step = info.get('step', 0)
    max_steps = info.get('max_steps', 300)
    reward = info.get('reward', 0)

    progress_lines = [
        f"Step: {step}/{max_steps}",
        f"Reward: {reward:.3f}",
    ]
    overlay = frame.copy()
    prog_w = int(165 * scale)
    prog_h = len(progress_lines) * line_h + int(12 * scale)
    prog_x = W - prog_w - 8
    cv2.rectangle(overlay, (prog_x, 8),
                  (prog_x + prog_w, 8 + prog_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(progress_lines):
        y = int(25 * scale) + i * line_h
        cv2.putText(frame, line, (prog_x + int(10 * scale), y),
                    font, font_scale_body, (255, 255, 255), thickness, cv2.LINE_AA)

    # ── 底部：步数分布条 ──
    k_dist = info.get('k_distribution', {})
    if k_dist:
        bar_h = int(30 * scale)
        bar_y = H - bar_h - int(8 * scale)
        bar_x = int(8 * scale)
        bar_w = W - 2 * bar_x

        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x, bar_y - int(18 * scale)),
                      (bar_x + bar_w, bar_y + bar_h + int(5 * scale)),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 标题
        cv2.putText(frame, "k distribution", (bar_x, bar_y - int(5 * scale)),
                    font, font_scale_body * 0.85, (200, 200, 200), thickness, cv2.LINE_AA)

        # 色彩方案: k=0 蓝(快), k 大 红(慢)
        k_colors = {0: (255, 180, 50), 1: (50, 220, 50), 2: (50, 220, 220), 5: (50, 80, 255)}
        total = sum(k_dist.values()) or 1
        x_cursor = bar_x
        for k_val in sorted(k_dist.keys()):
            frac = k_dist[k_val] / total
            seg_w = max(1, int(frac * bar_w))
            color = k_colors.get(k_val, (180, 180, 180))
            cv2.rectangle(frame, (x_cursor, bar_y),
                          (x_cursor + seg_w, bar_y + bar_h), color, -1)
            # 标签
            if seg_w > int(30 * scale):
                label = f"k={k_val}: {frac*100:.0f}%"
                cv2.putText(frame, label,
                            (x_cursor + int(3 * scale), bar_y + int(20 * scale)),
                            font, font_scale_body * 0.8, (0, 0, 0), thickness, cv2.LINE_AA)
            x_cursor += seg_w

    return frame


# ═══════════════════════════════════════════════════════════════
#  加载策略
# ═══════════════════════════════════════════════════════════════

def load_ada_bridger_policy(payload, cfg, device,
                            source_ckpt=None, refine_ckpt=None):
    """加载 Ada-BRIDGER checkpoint"""
    if source_ckpt:
        cfg.source_policy_checkpoint = source_ckpt
    if refine_ckpt:
        cfg.refinement_policy_checkpoint = refine_ckpt

    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace
    workspace = TrainAdaBridgerWorkspace(cfg)
    workspace.load_payload(payload)

    policy = workspace.policy
    policy.to(device)
    policy.eval()

    # 设置 normalizer
    src_norm = policy.source_policy.normalizer
    ref_norm = policy.refinement_policy.normalizer
    try:
        _ = src_norm['obs']
        normalizer = src_norm
    except (AttributeError, KeyError):
        normalizer = ref_norm
        policy.source_policy.normalizer = copy.deepcopy(normalizer)
    policy.set_normalizer(normalizer)
    policy.scheduler_deterministic = True
    return policy


def load_diffusion_policy(payload, cfg, device, steps_override=None):
    """
    加载标准扩散策略 (DDPM / DDIM) checkpoint。
    steps_override: 若指定，将推理步数切换为该值并使用 DDIMScheduler。
    """
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    cls = hydra.utils.get_class(cfg._target_)
    try:
        workspace = cls(cfg, output_dir='.')
    except TypeError:
        workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()

    # 覆盖推理步数 → 切换到 DDIMScheduler
    if steps_override is not None and hasattr(policy, 'num_inference_steps'):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        policy.num_inference_steps = steps_override
        policy.noise_scheduler = DDIMScheduler(
            num_train_timesteps=policy.noise_scheduler.config.num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon',
        )
        print(f"  Overriding to DDIMScheduler with {steps_override} inference steps")

    return policy


def load_policy(checkpoint_path, device='cuda:0',
                source_ckpt=None, refine_ckpt=None,
                steps_override=None):
    """
    统一入口：自动检测策略类型并加载。
    返回 (policy, cfg, policy_type)。
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    OmegaConf.set_struct(cfg, False)

    policy_type = detect_policy_type(cfg)
    print(f"  Detected policy type: {policy_type}")

    if policy_type == POLICY_ADA_BRIDGER:
        policy = load_ada_bridger_policy(
            payload, cfg, device,
            source_ckpt=source_ckpt, refine_ckpt=refine_ckpt,
        )
    else:
        policy = load_diffusion_policy(
            payload, cfg, device,
            steps_override=steps_override,
        )

    return policy, cfg, policy_type


# ═══════════════════════════════════════════════════════════════
#  获取高清帧
# ═══════════════════════════════════════════════════════════════

def get_hires_frame(env, render_size, task_type):
    """
    获取高分辨率 RGB 帧。
    - PushT: 从 pygame Surface (512×512) 抓取
    - robomimic: 通过 RobomimicLowdimWrapper 的 MuJoCo offscreen 渲染
    """
    if task_type == TASK_PUSHT:
        import pygame
        inner_env = env
        while hasattr(inner_env, 'env'):
            inner_env = inner_env.env
        inner_env.render(mode='rgb_array')
        if hasattr(inner_env, 'screen') and inner_env.screen is not None:
            frame = np.transpose(
                np.array(pygame.surfarray.pixels3d(inner_env.screen)),
                axes=(1, 0, 2)
            ).copy()
        else:
            frame = inner_env.render(mode='rgb_array')
    else:
        # robomimic: 找到 RobomimicLowdimWrapper 层调用 render
        # 它已经用 render_hw=(render_size, render_size) 初始化
        wrapper = env
        while hasattr(wrapper, 'env'):
            wrapper = wrapper.env
            if hasattr(wrapper, 'render_hw'):
                break
        frame = wrapper.render(mode='rgb_array')

    if frame.shape[0] != render_size or frame.shape[1] != render_size:
        frame = cv2.resize(frame, (render_size, render_size),
                           interpolation=cv2.INTER_LANCZOS4)
    return frame


# ═══════════════════════════════════════════════════════════════
#  单集录制
# ═══════════════════════════════════════════════════════════════

def record_episode(
    policy, cfg, env, seed,
    render_size=512, device='cuda:0',
    task_type=TASK_PUSHT, policy_type=POLICY_ADA_BRIDGER,
    rotation_transformer=None,
):
    """
    运行一个 episode 并返回带 HUD 的帧列表。
    同时支持 Ada-BRIDGER 和 DDPM/DDIM 策略。
    """
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    obs_dim = cfg.obs_dim
    abs_action = cfg.task.get('abs_action', False) if task_type == TASK_ROBOMIMIC else False
    max_steps = cfg.task.env_runner.get('max_steps', 700) if task_type == TASK_ROBOMIMIC else 300

    # Ada-BRIDGER 专用: refinement_steps 直接就是 DDIM 推理步数
    if policy_type == POLICY_ADA_BRIDGER:
        pass  # refinement_steps 值本身就是 DDIM 推理步数，无需映射

    # Diffusion 专用: scheduler 名称和步数
    scheduler_name = ''
    num_inference_steps = 0
    if policy_type == POLICY_DIFFUSION:
        num_inference_steps = getattr(policy, 'num_inference_steps', 100)
        sched_cls = type(policy.noise_scheduler).__name__
        if 'DDIM' in sched_cls:
            scheduler_name = 'DDIM'
        else:
            scheduler_name = 'DDPM'
        # num_inference_steps == num_train_timesteps 说明实际是全步 DDPM
        if num_inference_steps >= getattr(
            policy.noise_scheduler.config, 'num_train_timesteps', 100
        ):
            scheduler_name = 'DDPM'

    use_cuda_timing = (torch.device(device).type == 'cuda')

    env.seed(seed)
    obs = env.reset()
    policy.reset()

    frames = []
    k_history = []
    current_reward = 0.0
    decision_count = 0

    done = False
    while not done:
        # 抓取当前帧
        hires_frame = get_hires_frame(env, render_size, task_type)

        # 准备 obs_dict
        raw_obs = obs[np.newaxis].astype(np.float32)
        actual_dim = raw_obs.shape[-1]
        if actual_dim == obs_dim * 2:
            raw_obs = raw_obs[..., :obs_dim]
        obs_dict = {'obs': torch.from_numpy(raw_obs[:, :n_obs_steps]).to(device)}

        # ── 推理 + 计时 ──
        if policy_type == POLICY_ADA_BRIDGER:
            with torch.no_grad():
                result = policy.predict_action(obs_dict, return_intermediate=True)
            action = result['action']
            k_val = int(result['refinement_steps'].flatten()[0].item())
            k_history.append(k_val)

            stats = policy._inference_stats
            if stats:
                last = stats[-1]
                total_ms = last['total_time'] * 1000
                source_ms = last['source_time'] * 1000
                scheduler_ms = last['scheduler_time'] * 1000
                refine_ms = last['refine_time'] * 1000
                hz = 1000.0 / total_ms if total_ms > 0 else 0
            else:
                total_ms = source_ms = scheduler_ms = refine_ms = 0
                hz = 0

            hud_info = {
                'policy_type': POLICY_ADA_BRIDGER,
                'hz': hz,
                'k': k_val,
                'ddim_steps': k_val,  # refinement_steps 直接就是 DDIM 推理步数
                'source_ms': source_ms,
                'scheduler_ms': scheduler_ms,
                'refine_ms': refine_ms,
                'total_ms': total_ms,
                'k_distribution': dict(Counter(k_history)),
            }

        else:  # POLICY_DIFFUSION
            if use_cuda_timing:
                e_start = torch.cuda.Event(enable_timing=True)
                e_end = torch.cuda.Event(enable_timing=True)
                e_start.record()

            with torch.no_grad():
                result = policy.predict_action(obs_dict)

            if use_cuda_timing:
                e_end.record()
                torch.cuda.synchronize()
                total_ms = e_start.elapsed_time(e_end)
            else:
                total_ms = 0.0

            action = result['action']
            hz = 1000.0 / total_ms if total_ms > 0 else 0

            hud_info = {
                'policy_type': POLICY_DIFFUSION,
                'hz': hz,
                'scheduler_name': scheduler_name,
                'num_inference_steps': num_inference_steps,
                'denoise_ms': total_ms,
                'total_ms': total_ms,
            }

        decision_count += 1

        # 公共 HUD 字段
        step_count = len(env.reward) if hasattr(env, 'reward') else 0
        hud_info.update({
            'step': step_count,
            'max_steps': max_steps,
            'reward': current_reward,
        })
        frame_with_hud = draw_hud(hires_frame.copy(), hud_info, render_size)
        frames.append(frame_with_hud)

        # 执行动作
        np_action = action.detach().cpu().numpy()[0]
        env_action = np_action
        if abs_action and rotation_transformer is not None:
            env_action = _undo_transform_action(np_action, rotation_transformer)
        obs, reward, done, info = env.step(env_action)

        if isinstance(reward, (list, np.ndarray)):
            reward = float(np.max(reward))
        current_reward = max(current_reward, float(reward))

    # 最终帧
    final_frame = get_hires_frame(env, render_size, task_type)
    final_hud = {
        **hud_info,
        'step': max_steps,
        'reward': current_reward,
    }
    if policy_type == POLICY_ADA_BRIDGER:
        final_hud['k_distribution'] = dict(Counter(k_history))
    frames.append(draw_hud(final_frame.copy(), final_hud, render_size))

    episode_info = {
        'reward': current_reward,
        'decisions': decision_count,
        'avg_k': float(np.mean(k_history)) if k_history else 0,
        'k_history': k_history,
    }
    return frames, episode_info


# ═══════════════════════════════════════════════════════════════
#  视频写入
# ═══════════════════════════════════════════════════════════════

def write_video(frames, output_path, fps=10):
    """将 RGB 帧列表写为 mp4 视频"""
    if not frames:
        print("  [WARN] No frames to write")
        return

    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame in frames:
        # OpenCV 期望 BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"  Video saved: {output_path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Demo video with real-time inference HUD (Ada-BRIDGER / DDPM / DDIM)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (Ada-BRIDGER or Diffusion)")
    parser.add_argument("--output_dir", type=str, default="data/demo_video",
                        help="Output directory for videos")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of episodes to record")
    parser.add_argument("--render_size", type=int, default=512,
                        help="Video resolution (square)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video frame rate")
    parser.add_argument("--seed", type=int, default=10000,
                        help="Starting seed for episodes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--source_ckpt", type=str, default=None,
                        help="Override source policy checkpoint path (Ada-BRIDGER)")
    parser.add_argument("--refine_ckpt", type=str, default=None,
                        help="Override refinement policy checkpoint path (Ada-BRIDGER)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override diffusion inference steps (e.g. 8 for DDIM, 100 for DDPM)")
    parser.add_argument("--render_camera", type=str, default="agentview",
                        help="Camera name for robomimic rendering (default: agentview)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载策略（自动检测类型）
    policy, cfg, policy_type = load_policy(
        args.checkpoint, args.device,
        source_ckpt=args.source_ckpt,
        refine_ckpt=args.refine_ckpt,
        steps_override=args.steps,
    )

    # 自动检测任务类型
    task_type = detect_task_type(cfg)
    print(f"Task type: {task_type} (task_name={cfg.get('task_name', '?')})")

    # 准备 rotation_transformer（robomimic abs_action 需要）
    rotation_transformer = None
    if task_type == TASK_ROBOMIMIC and cfg.task.get('abs_action', False):
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    print(f"\nRecording {args.n_episodes} episodes at {args.render_size}x{args.render_size} @ {args.fps}fps")

    all_frames = []

    for ep_idx in range(args.n_episodes):
        seed = args.seed + ep_idx
        print(f"\n--- Episode {ep_idx + 1}/{args.n_episodes} (seed={seed}) ---")

        # 根据任务类型创建环境
        if task_type == TASK_PUSHT:
            env = create_pusht_env(cfg)
        else:
            env = create_robomimic_env(
                cfg,
                render_hw=(args.render_size, args.render_size),
                render_camera_name=args.render_camera,
            )

        frames, ep_info = record_episode(
            policy, cfg, env, seed,
            render_size=args.render_size,
            device=args.device,
            task_type=task_type,
            policy_type=policy_type,
            rotation_transformer=rotation_transformer,
        )
        env.close()

        summary = f"  Reward: {ep_info['reward']:.3f} | Decisions: {ep_info['decisions']}"
        if policy_type == POLICY_ADA_BRIDGER:
            summary += f" | Avg k: {ep_info['avg_k']:.2f}"
        print(summary)

        # 单 episode 视频
        ep_path = os.path.join(args.output_dir, f"episode_{ep_idx}_seed{seed}.mp4")
        write_video(frames, ep_path, args.fps)

        all_frames.extend(frames)

    # 合并视频
    if args.n_episodes > 1:
        combined_path = os.path.join(args.output_dir, "combined.mp4")
        write_video(all_frames, combined_path, args.fps)

    print(f"\nDone! Videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
