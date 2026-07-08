"""
诊断脚本：测试 δ (action discrepancy) 是否是"精炼有用"的可靠代理

回答三个问题：
  Q1: 精炼整体有没有用？ (k=0 vs k=5 成功率)
  Q2: δ 高的状态，精炼真的更可能有用吗？ (per-state counterfactual)
  Q3: δ 对"是否需要精炼"的分类准确率是多少？

方法：
  - 对每个 episode seed:
    (a) 跑纯 k=0 轨迹，记录每个 step 的 δ 和最终 reward
    (b) 在轨迹上选若干 switch point (覆盖高/低 δ)
    (c) 每个 switch point: 从同一 seed 重放到该步，切到 k=5 直到结束
        记录最终 reward → benefit = r_switched - r_k0
  - 分析 benefit 与 δ 的相关性

用法:
    python -m diffusion_policy.cli.diagnose_delta \
        --task pusht \
        --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
        --n-episodes 10 \
        --n-switches 5
"""

import argparse, copy, os, sys
from collections import Counter

import dill, hydra, numpy as np, torch
from omegaconf import OmegaConf

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy


# ─────────────────────────────────────────────────────────────────
# Task config (与 infer_d3rl 一致，精简版)
# ─────────────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "pusht": dict(
        max_steps=300, n_obs_steps=2, n_action_steps=8,
    ),
    "can": dict(
        max_steps=400, n_obs_steps=2, n_action_steps=8,
        abs_action=True,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        dataset_path="data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
    ),
    "lift": dict(
        max_steps=400, n_obs_steps=2, n_action_steps=8,
        abs_action=True,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        dataset_path="data/robomimic/datasets/lift/ph/low_dim_abs.hdf5",
    ),
    "square": dict(
        max_steps=400, n_obs_steps=2, n_action_steps=8,
        abs_action=True,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        dataset_path="data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
    ),
    "transport": dict(
        max_steps=700, n_obs_steps=2, n_action_steps=8,
        abs_action=True,
        obs_keys=["object","robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos",
                   "robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos"],
        dataset_path="data/robomimic/datasets/transport/ph/low_dim_abs.hdf5",
    ),
    "tool_hang": dict(
        max_steps=700, n_obs_steps=2, n_action_steps=8,
        abs_action=True,
        obs_keys=["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        dataset_path="data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5",
    ),
}

REFINEMENT_STEPS = [0, 1, 2, 5]
K_STRONG = 5  # strongest refinement, for computing δ


# ─────────────────────────────────────────────────────────────────
# Model loading (复用 infer_d3rl 逻辑)
# ─────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, device):
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
    # normalizer
    if "normalizer" in payload["state_dicts"]:
        policy.normalizer.load_state_dict(payload["state_dicts"]["normalizer"], strict=False)
    # 关键：返回 cfg，用于读 obs_dim/action_dim/horizon/n_obs_steps/n_action_steps
    return policy, cfg.policy


def _make_fixed_scheduler(k_val, device):
    """FixedKScheduler: 永远返回 k_val"""
    class _Fixed(torch.nn.Module):
        def __init__(self, k, rsteps):
            super().__init__()
            self.k = k
            self.k_idx = rsteps.index(k) if k in rsteps else 0
            self.register_buffer("rsteps_t", torch.tensor(rsteps, dtype=torch.long))
        def select_action(self, obs, init_action, deterministic=True):
            B = obs.shape[0] if isinstance(obs, torch.Tensor) else obs[list(obs)[0]].shape[0]
            dev = obs.device if isinstance(obs, torch.Tensor) else obs[list(obs)[0]].device
            idx = torch.full((B,), self.k_idx, dtype=torch.long, device=dev)
            steps = torch.full((B,), self.k, dtype=torch.long, device=dev)
            return steps, idx, torch.zeros(B, device=dev), torch.zeros(B, device=dev)
    return _Fixed(k_val, REFINEMENT_STEPS).to(device)


def build_d3rl_policy(source_policy, refine_policy, k_val, tcfg, device):
    """构建固定 k 值的 AdaBridgerPolicy"""
    scheduler = _make_fixed_scheduler(k_val, device)
    policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=tcfg["horizon"],
        obs_dim=tcfg["obs_dim"],
        action_dim=tcfg["action_dim"],
        n_action_steps=tcfg["n_action_steps"],
        n_obs_steps=tcfg["n_obs_steps"],
        refinement_steps=REFINEMENT_STEPS,
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    # normalizer
    try:
        _ = source_policy.normalizer["obs"]
        normalizer = source_policy.normalizer
    except (AttributeError, KeyError):
        normalizer = refine_policy.normalizer
        source_policy.normalizer = copy.deepcopy(normalizer)
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy


# ─────────────────────────────────────────────────────────────────
# Env creation
# ─────────────────────────────────────────────────────────────────
def create_env(task):
    tcfg = TASK_CONFIGS[task]
    if task == "pusht":
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        env = PushTKeypointsEnv()
        env = MultiStepWrapper(env, n_obs_steps=tcfg["n_obs_steps"],
                               n_action_steps=tcfg["n_action_steps"],
                               max_episode_steps=tcfg["max_steps"])
        return env, tcfg, False  # is_robomimic=False
    else:
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        from diffusion_policy.model.common.rotation_transformer import RotationTransformer
        dataset_path = os.path.expanduser(tcfg["dataset_path"])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        rot_tf = None
        if tcfg.get("abs_action"):
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rot_tf = RotationTransformer("axis_angle", "rotation_6d")
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": tcfg["obs_keys"]})
        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
        wrapped = RobomimicLowdimWrapper(env=robomimic_env, obs_keys=tcfg["obs_keys"],
                                         init_state=None, render_hw=(128,128),
                                         render_camera_name="agentview")
        env = MultiStepWrapper(wrapped, n_obs_steps=tcfg["n_obs_steps"],
                               n_action_steps=tcfg["n_action_steps"],
                               max_episode_steps=tcfg["max_steps"])
        return env, tcfg, True, rot_tf


def _undo_transform_action(action, rot_tf):
    if rot_tf is None:
        return action
    raw = action.shape
    if raw[-1] == 20:
        action = action.reshape(-1, 2, 10)
    d_rot = action.shape[-1] - 4
    pos = action[..., :3]; rot = action[..., 3:3+d_rot]; grip = action[..., [-1]]
    rot = rot_tf.inverse(rot)
    out = np.concatenate([pos, rot, grip], axis=-1)
    if raw[-1] == 20:
        out = out.reshape(*raw[:-1], 14)
    return out


# ─────────────────────────────────────────────────────────────────
# Core: run one episode with fixed k, return trajectory + deltas
# ─────────────────────────────────────────────────────────────────
def run_episode_fixed_k(policy_k0, policy_k5, env, tcfg, device,
                        seed, is_robomimic, rot_tf,
                        torch_seed=42):
    """跑一条 k=0 轨迹，同时每步用 policy_k5 的 refiner 计算 δ。

    Returns:
        deltas: list[float]      每个 env step 的 δ
        final_reward: float      最终 reward (max over episode)
        n_steps: int
    """
    # 确定性：固定 torch seed 使 VAE 采样可复现
    torch.manual_seed(torch_seed)

    env.seed(seed)
    obs = env.reset()
    policy_k0.reset()

    done = False
    step = 0
    deltas = []
    all_rewards = []
    obs_dim = tcfg["obs_dim"]
    n_obs = tcfg["n_obs_steps"]
    n_act = tcfg["n_action_steps"]

    while not done and step < tcfg["max_steps"]:
        # 准备 obs_dict
        raw = obs[np.newaxis].astype(np.float32)
        if not is_robomimic and raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_dict = {"obs": torch.from_numpy(raw[:, :n_obs]).to(device)}

        with torch.no_grad():
            # k=0: source only
            result0 = policy_k0.predict_action(obs_dict, return_intermediate=True)
            a_init = result0["init_action"]  # [1, horizon, action_dim]

            # 用 k=5 refiner 算 a_strong (不执行，只计算)
            a_strong = policy_k0._sdedit_refine(obs_dict, a_init, K_STRONG)

        # δ = ||a_init - a_strong||_1 / D_a
        delta = (a_init - a_strong).abs().sum(dim=-1).mean().item() / tcfg["action_dim"]
        deltas.append(delta)

        # 执行 k=0 action
        action = result0["action"][0].cpu().numpy()  # [n_action_steps, action_dim]
        action_for_env = action[:n_act]
        if is_robomimic and rot_tf is not None:
            action_for_env = _undo_transform_action(action_for_env, rot_tf)

        obs, reward, done, info = env.step(action_for_env)
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)
        step += 1

    final_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    return deltas, final_reward, step


def run_episode_switch_at(policy_k5, env, tcfg, device,
                          seed, switch_step, is_robomimic, rot_tf,
                          torch_seed=42):
    """从同一 seed 出发，先跑 k=0 到 switch_step，然后切到 k=5 直到结束。

    注意：这里用 k=0 policy 跑前 switch_step 步（需要 policy_k5 的 source 部分，
    但 AdaBridgerPolicy 的 source_policy 和 refiner 是冻结共享的，所以
    用 policy_k5 跑 k=0 前 switch_step 步，再切到 k=5 即可。）

    但 AdaBridgerPolicy 的 k 是由 scheduler 决定的，这里 scheduler 固定为 k=5。
    所以前 switch_step 步需要暂时用 k=0 的 scheduler。

    更简单的方法：直接复用两条 policy。
    """
    # 这里需要同时用 k=0 (前 N 步) 和 k=5 (之后)
    # 简化：用 k=5 policy 跑，但前 switch_step 步手动用 source 输出
    torch.manual_seed(torch_seed)

    env.seed(seed)
    obs = env.reset()
    # 使用 k=5 policy 但临时覆盖 scheduler
    policy_k5.reset()

    done = False
    step = 0
    all_rewards = []
    obs_dim = tcfg["obs_dim"]
    n_obs = tcfg["n_obs_steps"]
    n_act = tcfg["n_action_steps"]

    while not done and step < tcfg["max_steps"]:
        raw = obs[np.newaxis].astype(np.float32)
        if not is_robomimic and raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_dict = {"obs": torch.from_numpy(raw[:, :n_obs]).to(device)}

        with torch.no_grad():
            if step < switch_step:
                # k=0: 只用 source
                result = policy_k5.predict_action(obs_dict, return_intermediate=True)
                # 手动取 init_action 作为最终 action（不精炼）
                action = result["init_action"][:, :n_act, :]
            else:
                # k=5: 正常精炼
                result = policy_k5.predict_action(obs_dict, return_intermediate=True)
                action = result["action"]  # 已经是精炼后的

        np_action = action[0].cpu().numpy()
        if is_robomimic and rot_tf is not None:
            np_action = _undo_transform_action(np_action, rot_tf)

        obs, reward, done, info = env.step(np_action[:n_act])
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)
        step += 1

    final_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    return final_reward, step


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Diagnose: is δ a reliable proxy?")
    parser.add_argument("--task", default="pusht", choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--source", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--n-switches", type=int, default=5,
                        help="每条 episode 选几个 switch point")
    parser.add_argument("--eta", type=float, default=0.5,
                        help="δ 阈值比例 (median * η)")
    args = parser.parse_args()

    device = args.device
    task = args.task
    env_tcfg = TASK_CONFIGS[task]

    print("=" * 70)
    print(f"Delta Diagnostic — Task: {task}")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/4] Loading models...")
    source_policy, src_cfg = load_policy(args.source, device)
    refine_policy, _ = load_policy(args.refiner, device)

    # 从 source checkpoint 提取模型维度（不能用硬编码，PushT VAE 的 obs_dim=20）
    model_cfg = dict(
        obs_dim=src_cfg.obs_dim,
        action_dim=src_cfg.action_dim,
        horizon=src_cfg.horizon,
        n_obs_steps=src_cfg.n_obs_steps,
        n_action_steps=src_cfg.n_action_steps,
    )
    print(f"  Model dims from checkpoint: obs_dim={model_cfg['obs_dim']}, "
          f"action_dim={model_cfg['action_dim']}, "
          f"horizon={model_cfg['horizon']}")

    # 2. 构建 k=0 和 k=5 policy
    print("\n[2/4] Building fixed-k policies...")
    policy_k0 = build_d3rl_policy(source_policy, refine_policy, 0, model_cfg, device)
    policy_k5 = build_d3rl_policy(source_policy, refine_policy, K_STRONG, model_cfg, device)

    # 3. 创建环境
    print("\n[3/4] Creating environment...")
    env_result = create_env(task)
    if len(env_result) == 3:
        env, _, is_robomimic = env_result
        rot_tf = None
    else:
        env, _, is_robomimic, rot_tf = env_result

    # 合并 model_cfg + env_tcfg 作为完整配置
    tcfg = {**model_cfg, **env_tcfg}

    # 4. 跑诊断
    print(f"\n[4/4] Running diagnostic ({args.n_episodes} episodes, "
          f"{args.n_switches} switches each)...")

    # ── Test 1: k=0 vs k=5 整体 ──
    print("\n── Test 1: k=0 vs k=5 overall success ──")
    rewards_k0 = []
    rewards_k5 = []
    for ep in range(args.n_episodes):
        seed = 42 + ep
        _, r0, n0 = run_episode_fixed_k(
            policy_k0, policy_k5, env, tcfg, device, seed, is_robomimic, rot_tf)
        torch.manual_seed(42)
        env.seed(seed)
        # k=5 全程
        r5, n5 = run_episode_switch_at(
            policy_k5, env, tcfg, device, seed, 0, is_robomimic, rot_tf)
        rewards_k0.append(r0)
        rewards_k5.append(r5)
        print(f"  Ep {ep+1}: k=0 reward={r0:.3f}, k=5 reward={r5:.3f}, "
              f"benefit={r5-r0:+.3f}, steps(k0)={n0}")

    print(f"\n  k=0 mean reward: {np.mean(rewards_k0):.3f} ± {np.std(rewards_k0):.3f}")
    print(f"  k=5 mean reward: {np.mean(rewards_k5):.3f} ± {np.std(rewards_k5):.3f}")
    print(f"  Mean benefit:    {np.mean(rewards_k5)-np.mean(rewards_k0):+.3f}")
    k0_success = np.mean([r > 0.5 for r in rewards_k0])
    k5_success = np.mean([r > 0.5 for r in rewards_k5])
    print(f"  k=0 success rate: {k0_success:.1%}")
    print(f"  k=5 success rate: {k5_success:.1%}")

    # ── Test 2: Per-state δ vs benefit (switch-point counterfactual) ──
    print(f"\n── Test 2: Switch-point counterfactual (δ vs benefit) ──")
    all_deltas = []
    all_benefits = []
    all_labels = []  # δ-based label (1=needs refine)

    for ep in range(min(args.n_episodes, 5)):  # 用前 5 条做 switch-point
        seed = 42 + ep

        # (a) 跑 k=0 轨迹，收集 δ
        deltas, r0, n_steps = run_episode_fixed_k(
            policy_k0, policy_k5, env, tcfg, device, seed, is_robomimic, rot_tf)

        if n_steps < 10:
            continue

        # (b) 选 switch points: 均匀采样 + δ 极端值
        delta_arr = np.array(deltas)
        # 均匀采样
        uniform_idx = np.linspace(n_steps//4, 3*n_steps//4, args.n_switches, dtype=int)
        # δ 最高的几个
        top_idx = np.argsort(delta_arr)[-args.n_switches:]
        # δ 最低的几个
        bot_idx = np.argsort(delta_arr)[:args.n_switches]
        switch_points = sorted(set(list(uniform_idx) + list(top_idx) + list(bot_idx)))

        median_delta = np.median(delta_arr)
        delta_thresh = median_delta * args.eta

        for sp in switch_points:
            if sp >= n_steps - 5:
                continue
            # 从同 seed 重放，在 sp 处切到 k=5
            r_switch, _ = run_episode_switch_at(
                policy_k5, env, tcfg, device, seed, sp, is_robomimic, rot_tf)
            benefit = r_switch - r0
            delta_val = deltas[sp]
            label = 1 if delta_val >= delta_thresh else 0

            all_deltas.append(delta_val)
            all_benefits.append(benefit)
            all_labels.append(label)

            print(f"  Ep {ep+1} step {sp:3d}: δ={delta_val:.4f} "
                  f"(label={label}), r0={r0:.3f}, r_switch={r_switch:.3f}, "
                  f"benefit={benefit:+.3f}")

    # ── Analysis ──
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if len(all_deltas) < 5:
        print("  Not enough switch-point data. Increase --n-episodes.")
        return

    deltas = np.array(all_deltas)
    benefits = np.array(all_benefits)
    labels = np.array(all_labels)

    # Q1: 精炼整体有用吗？
    print("\nQ1: Does refinement help overall?")
    print(f"  k=0 success: {k0_success:.1%}, k=5 success: {k5_success:.1%}")
    if k5_success > k0_success + 0.05:
        print("  → YES, refinement helps. Proceed to Q2.")
    elif k5_success < k0_success - 0.05:
        print("  → NO, refinement HURTS. The refiner may be broken on rollout states.")
        print("    This would explain scheduler collapse to k=0.")
    else:
        print("  → INCONCLUSIVE, refinement neither helps nor hurts much.")

    # Q2: δ 与 benefit 相关吗？
    print("\nQ2: Is δ correlated with refinement benefit?")
    if np.std(deltas) > 1e-8 and np.std(benefits) > 1e-8:
        from scipy.stats import spearmanr, pearsonr
        rho_s, p_s = spearmanr(deltas, benefits)
        rho_p, p_p = pearsonr(deltas, benefits)
        print(f"  Spearman ρ = {rho_s:.3f} (p={p_s:.4f})")
        print(f"  Pearson  r = {rho_p:.3f} (p={p_p:.4f})")
        if rho_s > 0.3 and p_s < 0.05:
            print("  → YES, δ positively correlates with benefit. Proxy is reasonable.")
        elif rho_s < -0.3 and p_s < 0.05:
            print("  → NEGATIVE correlation! High δ → refinement HURTS.")
            print("    The refiner is broken on high-δ states. Stage 2 labels are wrong.")
        else:
            print("  → NO significant correlation. δ is NOT a reliable proxy.")
            print("    Stage 2 labels based on δ are unreliable → pipeline foundation is weak.")
    else:
        print("  Cannot compute (no variance in δ or benefit).")

    # Q3: δ 分类的准确率
    print("\nQ3: δ-based label accuracy (does high δ predict positive benefit?)")
    # True label: benefit > 0 → refinement helped
    true_needs_refine = benefits > 0.01  # small positive threshold
    pred_needs_refine = labels == 1

    tp = np.sum(pred_needs_refine & true_needs_refine)
    fp = np.sum(pred_needs_refine & ~true_needs_refine)
    fn = np.sum(~pred_needs_refine & true_needs_refine)
    tn = np.sum(~pred_needs_refine & ~true_needs_refine)

    accuracy = (tp + tn) / len(benefits) if len(benefits) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  δ-label = 1 (needs refine): predicted {np.sum(pred_needs_refine)}, "
          f"truly beneficial: {tp}, harmful (FP): {fp}")
    print(f"  δ-label = 0 (sufficient):    predicted {np.sum(~pred_needs_refine)}, "
          f"missed beneficial (FN): {fn}, correct (TN): {tn}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}  (of states δ says 'needs refine', how many actually benefit)")
    print(f"  Recall:    {recall:.1%}  (of states that actually benefit, how many δ catches)")

    # Summary verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    issues = []
    if k5_success <= k0_success:
        issues.append("refiner is net-negative on rollout states (frozen primitive broken on OOD)")
    if len(all_deltas) >= 5:
        rho_s_val = spearmanr(deltas, benefits)[0] if np.std(deltas) > 1e-8 else 0
        if abs(rho_s_val) < 0.3:
            issues.append("δ is not correlated with benefit (Stage 2 label foundation is weak)")
        elif rho_s_val < -0.3:
            issues.append("δ is anti-correlated with benefit (high δ → refinement hurts)")
    if precision < 0.6:
        issues.append(f"δ precision is low ({precision:.1%}): many false 'needs refine' labels")

    if not issues:
        print("  δ proxy looks reasonable. Pipeline foundation is OK.")
        print("  → Problem is likely in Stage 3/4 (scheduler training), not Stage 2.")
    else:
        print("  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n  → Consider replacing δ with true counterfactual labels in Stage 2.")

    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
