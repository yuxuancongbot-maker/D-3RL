"""
PushT D3RL 推理脚本 —— 加载 source policy + scheduler + diffusion refiner 进行推理

用法:
    python -m diffusion_policy.cli.infer_pusht \
        --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
        --scheduler data/outputs/ada_scheduler_pusht/scheduler_best.pt

也可以不传 scheduler（纯 diffusion 推理）:
    python -m diffusion_policy.cli.infer_pusht \
        --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt
"""

import argparse, copy, os, sys, time
from collections import Counter

import dill, hydra, numpy as np, torch
from omegaconf import OmegaConf

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler

# ────────────────────────────────────────────────────────────────────
# 加载单个 checkpoint
# ────────────────────────────────────────────────────────────────────
def load_policy(ckpt_path: str, device: str = "cuda:0"):
    """加载 VAE source policy 或 diffusion refiner policy。"""
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
    """加载 Stage 3 预训练的 scheduler。"""
    print(f"  Loading scheduler: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device)
    state = data["scheduler_state_dict"]
    scfg = data.get("config", {})

    # 兼容旧 config 的 step_options
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
    # 兼容旧 checkpoint：step_options_tensor → refinement_steps_tensor
    if "step_options_tensor" in state and "refinement_steps_tensor" not in state:
        state["refinement_steps_tensor"] = state.pop("step_options_tensor")
    scheduler.load_state_dict(state)
    scheduler.eval()
    scheduler.to(device)
    print(f"  Scheduler config: obs_dim={scfg.get('obs_dim')}, "
          f"action_dim={scfg.get('action_dim')}, val_acc={data.get('val_acc', 'N/A')}")
    return scheduler

# ────────────────────────────────────────────────────────────────────
# PushT 推理
# ────────────────────────────────────────────────────────────────────
def run_inference(policy, env, n_obs_steps=2, n_action_steps=8, obs_dim=2, device="cuda:0"):
    """在 PushT 上运行一个 episode，返回统计。"""
    from diffusion_policy.common.pytorch_util import dict_apply

    policy.reset()
    obs = env.reset()
    done = False
    timings = []
    all_rewards = []
    k_history = []

    use_cuda_timing = (device == "cuda:0" and torch.cuda.is_available())
    if use_cuda_timing:
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)

    while not done:
        raw_obs = obs[np.newaxis].astype(np.float32)
        if raw_obs.shape[-1] == obs_dim * 2:
            raw_obs = raw_obs[..., :obs_dim]

        obs_dict = {"obs": torch.from_numpy(raw_obs[:, :n_obs_steps]).to(device)}

        if use_cuda_timing:
            e_start.record()
        else:
            t0 = time.perf_counter()

        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)

        if use_cuda_timing:
            e_end.record()
            torch.cuda.synchronize()
            timings.append(e_start.elapsed_time(e_end))
        else:
            timings.append((time.perf_counter() - t0) * 1000)

        k = int(result["refinement_steps"].flatten()[0].item())
        k_history.append(k)

        np_action = result["action"].cpu().numpy()
        # MultiStepWrapper 期望 (n_action_steps, action_dim)，去掉 batch 维
        obs, reward, done, info = env.step(np_action[0, :n_action_steps])
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)

    final_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    avg_ms = np.mean(timings) if timings else 0.0
    step_dist = Counter(k_history)
    stats = policy.get_inference_stats() if hasattr(policy, "get_inference_stats") else {}

    return {
        "steps": len(timings),
        "final_reward": final_reward,
        "success": float(final_reward > 0.5),
        "avg_inference_ms": avg_ms,
        "hz": 1000 / avg_ms if avg_ms > 0 else 0,
        "k_distribution": dict(step_dist),
        "avg_refinement_steps": np.mean(k_history) if k_history else 0,
        "policy_stats": stats,
    }


def create_env(n_obs_steps=2, n_action_steps=8):
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    env = PushTKeypointsEnv()
    env = MultiStepWrapper(
        env, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, max_episode_steps=300,
    )
    return env


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D3RL PushT inference")
    parser.add_argument("--source", required=True, help="VAE source policy checkpoint")
    parser.add_argument("--refiner", required=True, help="Diffusion refiner checkpoint")
    parser.add_argument("--scheduler", default=None, help="Stage 3 scheduler checkpoint (可选)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--refinement-steps", type=int, default=None,
                        help="如果没有 scheduler，用这个固定 DDIM 步数 (0=只用 VAE)")
    args = parser.parse_args()

    device = args.device

    # 1. 加载三个组件
    print("=" * 60)
    print("1. Loading models...")
    source_policy, src_cfg = load_policy(args.source, device)
    refine_policy, ref_cfg = load_policy(args.refiner, device)

    # 从 checkpoint 推断参数
    obs_dim = src_cfg.policy.obs_dim
    action_dim = src_cfg.policy.action_dim
    horizon = src_cfg.policy.horizon
    n_obs_steps = src_cfg.policy.n_obs_steps
    n_action_steps = src_cfg.policy.n_action_steps

    # 2. 加载或构造 scheduler
    if args.scheduler:
        scheduler = load_scheduler(args.scheduler, device)
    else:
        # 没有 scheduler：根据 --refinement-steps 固定推理
        k = args.refinement_steps if args.refinement_steps is not None else 5
        refinement_steps = [0, 1, 2, 5]

        class FixedScheduler(torch.nn.Module):
            def __init__(self, k_val, rsteps):
                super().__init__()
                self.k = k_val
                self.rsteps = rsteps
                self.k_idx = rsteps.index(k_val) if k_val in rsteps else 0
                self.register_buffer("rsteps_t", torch.tensor(rsteps, dtype=torch.long))

            def select_action(self, obs, init_action, deterministic=True):
                B = obs.shape[0]
                device = obs.device
                idx = torch.full((B,), self.k_idx, dtype=torch.long, device=device)
                steps = torch.full((B,), self.k, dtype=torch.long, device=device)
                lp = torch.zeros(B, device=device)
                val = torch.zeros(B, device=device)
                return steps, idx, lp, val

        scheduler = FixedScheduler(k, refinement_steps).to(device)
        print(f"  Using fixed scheduler: k={k} (no RL scheduler provided)")

    # 3. 组装 D3RL policy
    print("\n2. Building D3RL policy...")
    d3rl_policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    # normalizer: 优先用 source 的，没有就从 refiner 借
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
    print("\n3. Running inference...")
    env = create_env(n_obs_steps, n_action_steps)
    all_results = []
    for ep in range(args.n_episodes):
        env.seed(42 + ep)
        r = run_inference(d3rl_policy, env, n_obs_steps, n_action_steps, obs_dim, device)
        all_results.append(r)
        print(f"  Episode {ep+1}: reward={r['final_reward']:.2f}  "
              f"success={'✓' if r['success'] else '✗'}  "
              f"avg_k={r['avg_refinement_steps']:.1f}  "
              f"{r['avg_inference_ms']:.1f}ms  (~{r['hz']:.0f}Hz)  "
              f"k_dist={r['k_distribution']}")
    env.close()

    # 5. 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    rewards = [r["final_reward"] for r in all_results]
    successes = [r["success"] for r in all_results]
    ks = [r["avg_refinement_steps"] for r in all_results]
    ms_list = [r["avg_inference_ms"] for r in all_results]
    print(f"  Episodes:          {len(all_results)}")
    print(f"  Success rate:      {np.mean(successes):.2%}")
    print(f"  Mean reward:       {np.mean(rewards):.3f}")
    print(f"  Avg k (refine):    {np.mean(ks):.2f}")
    print(f"  Avg inference:     {np.mean(ms_list):.1f} ms  (~{1000/np.mean(ms_list):.0f} Hz)" if ms_list else "")
    # 汇总 k 分布
    total_k = Counter()
    for r in all_results:
        total_k.update(r["k_distribution"])
    total = sum(total_k.values())
    print(f"  k distribution:")
    for k in sorted(total_k):
        print(f"    k={k}: {total_k[k]/total*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
