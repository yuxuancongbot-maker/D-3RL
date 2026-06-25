"""
D3RL 推理脚本 —— 加载 source policy + scheduler + diffusion refiner 在任意任务上推理

用法:
    # PushT
    python -m diffusion_policy.cli.infer_d3rl \
        --task pusht \
        --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
        --scheduler data/outputs/ada_scheduler_pusht/scheduler_best.pt

    # Robomimic tasks
    python -m diffusion_policy.cli.infer_d3rl \
        --task can \
        --source weights/lowdim/predictor/vae/vae_can_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/robomimic/cnn/can/epoch=0350-test_mean_score=1.000.ckpt \
        --scheduler data/outputs/ada_scheduler_can/scheduler_best.pt
"""

import argparse, copy, os, sys, time
from collections import Counter
from typing import Dict, List

import dill, hydra, numpy as np, torch
from omegaconf import OmegaConf

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler

# ────────────────────────────────────────────────────────────────
# Task registry: known task parameters for env creation
# ────────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "pusht": {
        "dataset_path": "data/pusht/pusht_cchi_v7_replay.zarr",
        "obs_dim": 2,
        "action_dim": 2,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 300,
    },
    "can": {
        "dataset_path": "data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 23,
        "action_dim": 10,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 400,
        "abs_action": True,
    },
    "lift": {
        "dataset_path": "data/robomimic/datasets/lift/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 19,
        "action_dim": 10,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 400,
        "abs_action": True,
    },
    "square": {
        "dataset_path": "data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 23,
        "action_dim": 10,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 400,
        "abs_action": True,
    },
    "transport": {
        "dataset_path": "data/robomimic/datasets/transport/ph/low_dim_abs.hdf5",
        "obs_keys": [
            "object",
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
        ],
        "obs_dim": 59,
        "action_dim": 20,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 700,
        "abs_action": True,
    },
    "tool_hang": {
        "dataset_path": "data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "obs_dim": 53,
        "action_dim": 10,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 700,
        "abs_action": True,
    },
}


# ────────────────────────────────────────────────────────────────
# Checkpoint loading
# ────────────────────────────────────────────────────────────────
def load_policy(ckpt_path: str, device: str = "cuda:0"):
    """加载 VAE source policy 或 diffusion refiner。"""
    print(f"  Loading: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    state_dict = payload["state_dicts"]["model"]

    looks_like_vae = any("net.encoder_net" in k for k in state_dict.keys())

    OmegaConf.set_struct(cfg.policy, False)
    if looks_like_vae:
        cfg.policy.backend = "vae"
        cfg.policy.model = {
            "_target_": "diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel",
            "action_dim": cfg.policy.action_dim,
            "action_horizon": cfg.policy.horizon,
            "obs_dim": cfg.policy.obs_dim,
            "obs_horizon": cfg.policy.n_obs_steps,
            "latent_dim": 32, "layer": 256,
            "use_ema": True, "pretrain": False, "ckpt_path": None,
        }

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)
    return policy, cfg


def load_scheduler(ckpt_path: str, device: str = "cuda:0"):
    """加载 Stage 3 预训练 scheduler。"""
    print(f"  Loading scheduler: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device)
    state = data["scheduler_state_dict"]
    scfg = data.get("config", {})

    # 兼容旧 checkpoint key
    if "step_options_tensor" in state and "refinement_steps_tensor" not in state:
        state["refinement_steps_tensor"] = state.pop("step_options_tensor")
    if "step_options" in scfg and "refinement_steps" not in scfg:
        scfg["refinement_steps"] = scfg["step_options"]

    rsteps = scfg.get("refinement_steps") or scfg.get("step_options") or [0, 1, 2, 5]
    scheduler = AdaScheduler(
        obs_dim=scfg.get("obs_dim", 2),
        action_dim=scfg.get("action_dim", 2),
        action_horizon=scfg.get("horizon", 16),
        n_obs_steps=scfg.get("n_obs_steps", 2),
        hidden_dim=256,
        num_layers=2,
        refinement_steps=rsteps,
    )
    scheduler.load_state_dict(state)
    scheduler.eval()
    scheduler.to(device)
    print(f"  Scheduler: obs_dim={scfg.get('obs_dim')}, action_dim={scfg.get('action_dim')}, "
          f"val_acc={data.get('val_acc', 'N/A')}")
    return scheduler


# ────────────────────────────────────────────────────────────────
# Env creation
# ────────────────────────────────────────────────────────────────
def create_env(task: str):
    """为指定任务创建环境。"""
    tcfg = TASK_CONFIGS[task]

    if task == "pusht":
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        env = PushTKeypointsEnv()
        env = MultiStepWrapper(
            env, n_obs_steps=tcfg["n_obs_steps"],
            n_action_steps=tcfg["n_action_steps"],
            max_episode_steps=tcfg["max_steps"],
        )
    else:
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer

        dataset_path = os.path.expanduser(tcfg["dataset_path"])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

        rotation_transformer = None
        if tcfg.get("abs_action"):
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")

        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": tcfg["obs_keys"]})
        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False,
        )
        wrapped_env = RobomimicLowdimWrapper(
            env=robomimic_env, obs_keys=tcfg["obs_keys"],
            init_state=None, render_hw=(128, 128), render_camera_name="agentview",
        )
        env = MultiStepWrapper(
            wrapped_env,
            n_obs_steps=tcfg["n_obs_steps"],
            n_action_steps=tcfg["n_action_steps"],
            max_episode_steps=tcfg["max_steps"],
        )
        env._rotation_transformer = rotation_transformer
        env._abs_action = tcfg.get("abs_action", False)

    return env, tcfg


# ────────────────────────────────────────────────────────────────
# Inference loop
# ────────────────────────────────────────────────────────────────
def run_episode(policy, env, tcfg: dict, device: str = "cuda:0"):
    """运行一个 episode，返回统计。"""
    policy.reset()
    obs = env.reset()
    done = False
    k_history: List[int] = []
    all_rewards: List[float] = []
    timings: List[float] = []

    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    obs_dim = tcfg["obs_dim"]
    n_obs_steps = tcfg["n_obs_steps"]
    n_action_steps = tcfg["n_action_steps"]
    is_robomimic = ("abs_action" in tcfg)

    if use_cuda:
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

    while not done:
        # 准备 obs: PushT 需要去掉 visibility mask，robomimic 直接用
        raw_obs = obs[np.newaxis].astype(np.float32)
        if not is_robomimic and raw_obs.shape[-1] == obs_dim * 2:
            raw_obs = raw_obs[..., :obs_dim]  # PushT: 去掉 mask
        obs_dict = {"obs": torch.from_numpy(raw_obs[:, :n_obs_steps]).to(device)}

        if use_cuda:
            e_start.record()
        else:
            t0 = time.perf_counter()

        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)

        if use_cuda:
            e_end.record()
            torch.cuda.synchronize()
            timings.append(e_start.elapsed_time(e_end))
        else:
            timings.append((time.perf_counter() - t0) * 1000)

        k = int(result["refinement_steps"].flatten()[0].item())
        k_history.append(k)

        np_action = result["action"].cpu().numpy()
        action_for_env = np_action[0, :n_action_steps]  # (n_action_steps, action_dim)

        # robomimic abs_action 需要反变换
        rot_tf = getattr(env, "_rotation_transformer", None)
        if rot_tf is not None and getattr(env, "_abs_action", False):
            action_for_env = _undo_transform_action(action_for_env, rot_tf)

        obs, reward, done, info = env.step(action_for_env)
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)

    final_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_ms = np.mean(timings) if timings else 0.0

    return {
        "final_reward": final_reward,
        "success": float(final_reward > 0.5),
        "avg_inference_ms": avg_ms,
        "hz": 1000 / avg_ms if avg_ms > 0 else 0,
        "k_distribution": dict(Counter(k_history)),
        "avg_refinement_steps": np.mean(k_history) if k_history else 0,
        "n_steps": len(timings),
    }


def _undo_transform_action(action, rotation_transformer):
    """将 rotation_6d 格式的动作转回 axis_angle（用于环境执行）。"""
    raw_shape = action.shape
    if raw_shape[-1] == 20:  # dual arm
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


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D3RL inference")
    parser.add_argument("--task", default="pusht",
                        choices=["pusht", "can", "lift", "square", "transport", "tool_hang"])
    parser.add_argument("--source", required=True, help="VAE source policy checkpoint")
    parser.add_argument("--refiner", required=True, help="Diffusion refiner checkpoint")
    parser.add_argument("--scheduler", default=None, help="Stage 3 scheduler checkpoint (可选)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--refinement-steps", type=int, default=None,
                        help="固定 DDIM 步数 (0=只用 VAE，跳过 scheduler)")
    args = parser.parse_args()

    device = args.device
    task = args.task
    tcfg = TASK_CONFIGS[task]

    # 1. 加载模型
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    print("1. Loading models...")
    source_policy, src_cfg = load_policy(args.source, device)
    refine_policy, ref_cfg = load_policy(args.refiner, device)

    # 2. Scheduler
    if args.scheduler:
        scheduler = load_scheduler(args.scheduler, device)
    else:
        k = args.refinement_steps if args.refinement_steps is not None else 5
        rsteps = [0, 1, 2, 5]

        class FixedScheduler(torch.nn.Module):
            def __init__(self, kv, rs):
                super().__init__()
                self.k, self.rsteps = kv, rs
                self.k_idx = rs.index(kv) if kv in rs else 0
                self.register_buffer("rs_t", torch.tensor(rs, dtype=torch.long))

            def select_action(self, obs, init_action, deterministic=True):
                B = obs.shape[0] if isinstance(obs, torch.Tensor) else obs[list(obs.keys())[0]].shape[0]
                dev = obs[list(obs.keys())[0]].device if isinstance(obs, dict) else obs.device
                idx = torch.full((B,), self.k_idx, dtype=torch.long, device=dev)
                steps = torch.full((B,), self.k, dtype=torch.long, device=dev)
                return steps, idx, torch.zeros(B, device=dev), torch.zeros(B, device=dev)

        scheduler = FixedScheduler(k, rsteps).to(device)
        print(f"  Using fixed scheduler: refinement_steps={k}")

    # 3. 组装 D3RL policy
    print("\n2. Building D3RL policy...")
    d3rl_policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=tcfg["horizon"],
        obs_dim=tcfg["obs_dim"],
        action_dim=tcfg["action_dim"],
        n_action_steps=tcfg["n_action_steps"],
        n_obs_steps=tcfg["n_obs_steps"],
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    try:
        _ = source_policy.normalizer["obs"]
        normalizer = source_policy.normalizer
        print("  Using normalizer from source policy")
    except (AttributeError, KeyError):
        normalizer = refine_policy.normalizer
        source_policy.normalizer = copy.deepcopy(normalizer)
        print("  Using normalizer from refiner (borrowed)")
    d3rl_policy.set_normalizer(normalizer)
    d3rl_policy.to(device)
    d3rl_policy.eval()

    # 4. 推理
    print(f"\n3. Running {args.n_episodes} episodes...")
    env, _ = create_env(task)
    all_results = []
    for ep in range(args.n_episodes):
        env.seed(42 + ep)
        r = run_episode(d3rl_policy, env, tcfg, device)
        all_results.append(r)
        print(f"  Ep {ep+1}: reward={r['final_reward']:.2f}  "
              f"success={'✓' if r['success'] else '✗'}  "
              f"avg_k={r['avg_refinement_steps']:.1f}  "
              f"{r['avg_inference_ms']:.1f}ms (~{r['hz']:.0f}Hz)  "
              f"k_dist={r['k_distribution']}")
    env.close()

    # 5. 汇总
    print("\n" + "=" * 60)
    print(f"Summary — {task}")
    print("=" * 60)
    rewards = [r["final_reward"] for r in all_results]
    successes = [r["success"] for r in all_results]
    ks = [r["avg_refinement_steps"] for r in all_results]
    ms_list = [r["avg_inference_ms"] for r in all_results]
    print(f"  Episodes:          {len(all_results)}")
    print(f"  Success rate:      {np.mean(successes):.2%}")
    print(f"  Mean reward:       {np.mean(rewards):.3f}")
    print(f"  Avg k (refine):    {np.mean(ks):.2f}")
    if ms_list:
        print(f"  Avg inference:     {np.mean(ms_list):.1f} ms  (~{1000/np.mean(ms_list):.0f} Hz)")
    total_k = Counter()
    for r in all_results:
        total_k.update(r["k_distribution"])
    total = sum(total_k.values())
    print(f"  k distribution:")
    for k_val in sorted(total_k):
        print(f"    k={k_val}: {total_k[k_val]/total*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
