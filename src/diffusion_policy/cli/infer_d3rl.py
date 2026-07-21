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
from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler, AdaSchedulerForImages
from diffusion_policy.common.pytorch_util import dict_apply

# ────────────────────────────────────────────────────────────────
# Task registry: known task parameters for env creation
# ────────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "pusht": {
        "dataset_path": "data/pusht/pusht_cchi_v7_replay.zarr",
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 300,
    },
    "can": {
        "dataset_path": "data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 400,
        "abs_action": True,
    },
    "lift": {
        "dataset_path": "data/robomimic/datasets/lift/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 400,
        "abs_action": True,
    },
    "square": {
        "dataset_path": "data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
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
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "max_steps": 700,
        "abs_action": True,
    },
    "tool_hang": {
        "dataset_path": "data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5",
        "obs_keys": ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
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
    policy_target = str(cfg.policy.get('_target_', ''))
    if looks_like_vae and 'action_predictor_image_vae_policy' not in policy_target:
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


def create_image_env(task: str, shape_meta: dict, dataset_path: str = None):
    """Create a single image-observation env using existing image wrappers."""
    if task in {"pusht", "pusht_image"}:
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        env = MultiStepWrapper(PushTImageEnv(render_size=96), n_obs_steps=2, n_action_steps=8,
                               max_episode_steps=300)
        return env, {"n_obs_steps": 2, "n_action_steps": 8, "max_steps": 300}

    base_task = task[:-6] if task.endswith("_image") else task
    if dataset_path is None:
        dataset_path = f"data/robomimic/datasets/{base_task}/ph/image.hdf5"
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    modality_mapping = {}
    for key, attr in shape_meta['obs'].items():
        modality_mapping.setdefault(attr.get('type', 'low_dim'), []).append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)
    env_meta = FileUtils.get_env_metadata_from_dataset(os.path.expanduser(dataset_path))
    # Detect abs_action / rotation_6d from shape_meta (action dim 20 → rotation_6d)
    action_dim = shape_meta['action']['shape'][0]
    rotation_transformer = None
    is_abs_action = False
    if action_dim == 20:  # dual-arm rotation_6d (abs)
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        is_abs_action = True
    elif action_dim == 10:  # single-arm rotation_6d (abs)
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
        is_abs_action = True
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=True)
    # Find first rgb key as render_obs_key (avoid hardcoding 'agentview_image')
    rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
    render_obs_key = rgb_keys[0] if rgb_keys else 'agentview_image'
    env = MultiStepWrapper(
        RobomimicImageWrapper(robomimic_env, shape_meta=shape_meta, render_obs_key=render_obs_key),
        n_obs_steps=2, n_action_steps=8,
        max_episode_steps=700 if base_task in {"transport", "tool_hang"} else 400)
    # Attach rotation transformer for run_episode action conversion (20d→14d)
    env._rotation_transformer = rotation_transformer
    env._abs_action = is_abs_action
    return env, {"n_obs_steps": 2, "n_action_steps": 8,
                 "max_steps": 700 if base_task in {"transport", "tool_hang"} else 400}


# ────────────────────────────────────────────────────────────────
# Inference loop
# ────────────────────────────────────────────────────────────────
def run_episode(policy, env, tcfg: dict, device: str = "cuda:0", obs_mode: str = "lowdim"):
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
        if obs_mode == "image":
            obs_dict = dict_apply(
                {k: v[np.newaxis, :n_obs_steps].astype(np.float32) for k, v in obs.items()},
                lambda x: torch.from_numpy(x).to(device),
            )
        else:
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
    avg_ms_all = np.mean(timings) if timings else 0.0
    timing_samples = timings[1:] if len(timings) > 1 else timings
    avg_ms = np.mean(timing_samples) if timing_samples else 0.0
    first_ms = timings[0] if timings else 0.0

    return {
        "final_reward": final_reward,
        "success": float(final_reward > 0.5),
        "avg_inference_ms": avg_ms,
        "avg_inference_ms_all": avg_ms_all,
        "first_inference_ms": first_ms,
        "hz": 1000 / avg_ms if avg_ms > 0 else 0,
        "k_distribution": dict(Counter(k_history)),
        "avg_refinement_steps": np.mean(k_history) if k_history else 0,
        "n_steps": len(timings),
        "n_timing_samples": len(timing_samples),
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
                        choices=["pusht", "can", "lift", "square", "transport", "tool_hang",
                                 "pusht_image", "can_image", "lift_image", "square_image",
                                 "transport_image", "tool_hang_image"])
    parser.add_argument("--source", required=True, help="VAE source policy checkpoint")
    parser.add_argument("--refiner", required=True, help="Diffusion refiner checkpoint")
    parser.add_argument("--scheduler", default=None, help="Stage 3 scheduler checkpoint (可选)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obs-mode", default="auto", choices=["auto", "lowdim", "image"])
    parser.add_argument("--image-dataset", default=None,
                        help="Robomimic image dataset path override for *_image tasks")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42,
                        help="First environment seed for evaluation episodes")
    parser.add_argument("--refinement-steps", type=int, default=None,
                        help="固定 DDIM 步数 (0=只用 VAE，跳过 scheduler)")
    parser.add_argument("--random-routing-steps", nargs="+", type=int, default=None,
                        help="State-independent random routing step choices, e.g. 0 5")
    parser.add_argument("--random-routing-probs", nargs="+", type=float, default=None,
                        help="Probabilities for --random-routing-steps, e.g. 0.625 0.375")
    parser.add_argument("--random-routing-seed", type=int, default=0,
                        help="Seed for state-independent random routing")
    args = parser.parse_args()

    device = args.device
    task = args.task
    obs_mode = args.obs_mode
    if obs_mode == "auto":
        obs_mode = "image" if task.endswith("_image") else "lowdim"
    base_task = task[:-6] if task.endswith("_image") else task
    tcfg = TASK_CONFIGS.get(base_task, {})

    # 1. 加载模型
    print("=" * 60)
    print(f"Task: {task}")
    print("=" * 60)
    print("1. Loading models...")
    source_policy, src_cfg = load_policy(args.source, device)
    refine_policy, ref_cfg = load_policy(args.refiner, device)

    # 2. Scheduler
    if args.random_routing_steps is not None:
        if args.scheduler is not None or args.refinement_steps is not None:
            raise ValueError("--random-routing-steps is mutually exclusive with --scheduler/--refinement-steps")
        if args.random_routing_probs is None:
            probs = np.ones(len(args.random_routing_steps), dtype=np.float64) / len(args.random_routing_steps)
        else:
            probs = np.asarray(args.random_routing_probs, dtype=np.float64)
            if len(probs) != len(args.random_routing_steps):
                raise ValueError("--random-routing-probs must have the same length as --random-routing-steps")
            probs = probs / probs.sum()
        rsteps = [0, 1, 2, 5]

        class RandomRoutingScheduler(torch.nn.Module):
            def __init__(self, step_choices, step_probs, all_steps, seed):
                super().__init__()
                self.step_choices = [int(x) for x in step_choices]
                self.step_probs = torch.tensor(step_probs, dtype=torch.float32)
                self.all_steps = [int(x) for x in all_steps]
                self.choice_to_idx = {step: self.all_steps.index(step) for step in self.step_choices}
                self.generator = torch.Generator(device="cpu")
                self.generator.manual_seed(int(seed))
                self.register_buffer("rs_t", torch.tensor(self.all_steps, dtype=torch.long))

            def select_action(self, obs, init_action, deterministic=True):
                B = obs.shape[0] if isinstance(obs, torch.Tensor) else obs[list(obs.keys())[0]].shape[0]
                dev = obs[list(obs.keys())[0]].device if isinstance(obs, dict) else obs.device
                choices = torch.multinomial(self.step_probs, B, replacement=True, generator=self.generator)
                chosen_steps = [self.step_choices[int(i)] for i in choices.cpu().tolist()]
                chosen_indices = [self.choice_to_idx[int(k)] for k in chosen_steps]
                steps = torch.tensor(chosen_steps, dtype=torch.long, device=dev)
                idx = torch.tensor(chosen_indices, dtype=torch.long, device=dev)
                return steps, idx, torch.zeros(B, device=dev), torch.zeros(B, device=dev)

        scheduler = RandomRoutingScheduler(args.random_routing_steps, probs, rsteps, args.random_routing_seed).to(device)
        print(f"  Using random routing scheduler: steps={args.random_routing_steps}, probs={probs.tolist()}, seed={args.random_routing_seed}")
    elif args.scheduler:
        if obs_mode == "image":
            data = torch.load(args.scheduler, map_location=device)
            scfg = data.get("config", {})
            if scfg.get("obs_mode") == "image_feature":
                # Feature-level image scheduler pretrained on source VAE obs_feat.
                # AdaBridgerPolicy._get_scheduler_obs will feed source_result['obs_feat']
                # when this obs_dim matches, so do not wrap a second image encoder here.
                scheduler = load_scheduler(args.scheduler, device)
            else:
                rsteps = scfg.get("refinement_steps") or scfg.get("step_options") or [0, 1, 2, 5]
                obs_feature_dim = scfg.get("obs_dim", None) or scfg.get("obs_feature_dim", None)
                if obs_feature_dim is None:
                    obs_feature_dim = refine_policy.obs_encoder.output_shape()[0]
                    print(f"  Inferred image scheduler obs_feature_dim={obs_feature_dim} from refiner encoder")
                scheduler = AdaSchedulerForImages(
                    obs_encoder=copy.deepcopy(refine_policy.obs_encoder),
                    obs_feature_dim=obs_feature_dim,
                    action_dim=scfg.get("action_dim", ref_cfg.policy.action_dim),
                    action_horizon=scfg.get("horizon", ref_cfg.policy.horizon),
                    n_obs_steps=scfg.get("n_obs_steps", ref_cfg.policy.n_obs_steps),
                    refinement_steps=rsteps,
                    freeze_encoder=True,
                ).to(device)
                scheduler.load_state_dict(data["scheduler_state_dict"], strict=False)
                scheduler.eval()
        else:
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

    # 从 source checkpoint 读取模型维度（不能用 TASK_CONFIGS 硬编码，PushT obs_dim=20 不是 2）
    policy_model = getattr(source_policy, 'model', source_policy)
    model_cfg = dict(
        horizon=getattr(source_policy, 'horizon', src_cfg.policy.horizon),
        obs_dim=src_cfg.policy.get('obs_dim', 0),
        action_dim=src_cfg.policy.get('action_dim', getattr(policy_model, 'action_dim')),
        n_action_steps=getattr(source_policy, 'n_action_steps', src_cfg.policy.n_action_steps),
        n_obs_steps=getattr(source_policy, 'n_obs_steps', src_cfg.policy.n_obs_steps),
    )
    # 向 run_episode 补充 env 参数
    tcfg_full = {**model_cfg, **tcfg}

    # 3. 组装 D3RL policy
    print(f"\n2. Building D3RL policy (obs_dim={model_cfg['obs_dim']}, action_dim={model_cfg['action_dim']})...")
    d3rl_policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=model_cfg["horizon"],
        obs_dim=model_cfg["obs_dim"],
        action_dim=model_cfg["action_dim"],
        n_action_steps=model_cfg["n_action_steps"],
        n_obs_steps=model_cfg["n_obs_steps"],
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
    if obs_mode == "image":
        shape_meta = ref_cfg.get('shape_meta', None)
        if shape_meta is None and 'task' in ref_cfg:
            shape_meta = ref_cfg.task.get('shape_meta', None)
        if shape_meta is None:
            raise ValueError("Image inference requires shape_meta in the refiner checkpoint config.")
        env, image_tcfg = create_image_env(task, shape_meta, dataset_path=args.image_dataset)
        tcfg_full.update(image_tcfg)
    else:
        env, _ = create_env(base_task)
    all_results = []
    for ep in range(args.n_episodes):
        env.seed(args.seed_start + ep)
        r = run_episode(d3rl_policy, env, tcfg_full, device, obs_mode=obs_mode)
        all_results.append(r)
        print(f"  Ep {ep+1}: reward={r['final_reward']:.2f}  "
              f"success={'✓' if r['success'] else '✗'}  "
              f"avg_k={r['avg_refinement_steps']:.1f}  "
              f"{r['avg_inference_ms']:.1f}ms (~{r['hz']:.0f}Hz)  "
              f"first={r['first_inference_ms']:.1f}ms  "
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
    ms_all_list = [r["avg_inference_ms_all"] for r in all_results]
    first_ms_list = [r["first_inference_ms"] for r in all_results]
    print(f"  Episodes:          {len(all_results)}")
    print(f"  Success rate:      {np.mean(successes):.2%}")
    print(f"  Mean reward:       {np.mean(rewards):.3f}")
    print(f"  Avg k (refine):    {np.mean(ks):.2f}")
    if ms_list:
        avg_ms = np.mean(ms_list)
        print(f"  Avg inference:     {avg_ms:.1f} ms  (~{1000/avg_ms:.0f} Hz)")
        print(f"  First inference:   {np.mean(first_ms_list):.1f} ms")
        print(f"  Avg inference all: {np.mean(ms_all_list):.1f} ms")

    internal_stats = d3rl_policy.get_inference_stats()
    if internal_stats.get('total_calls', 0) > 0:
        print("  Internal timing:")
        print(f"    source:          {internal_stats.get('avg_source_time', 0) * 1000:.2f} ms")
        print(f"    scheduler:       {internal_stats.get('avg_scheduler_time', 0) * 1000:.2f} ms")
        print(f"    refine:          {internal_stats.get('avg_refine_time', 0) * 1000:.2f} ms")
        print(f"    total:           {internal_stats.get('avg_total_time', 0) * 1000:.2f} ms")
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
