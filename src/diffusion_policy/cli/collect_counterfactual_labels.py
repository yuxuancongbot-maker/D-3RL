"""
Stage 2: True Counterfactual Label Collection

对每条 rollout 轨迹同时测 k=0 和 k=5 的结果，用真实 success 差打标签。
替代原来基于 δ (action discrepancy) 的弱标签。

用法:
    python -m diffusion_policy.cli.collect_counterfactual_labels \
        --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
        --task pusht --n-episodes 200 \
        --output data/counterfactual_labels_pusht.pt
"""

import argparse, copy, os, sys
from collections import defaultdict

import dill, hydra, numpy as np, torch, tqdm
from omegaconf import OmegaConf


# ──────────────────────────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────────────────────────
def make_env_pusht():
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    env = PushTKeypointsEnv()
    return MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, device):
    print(f"  Loading: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    state_dict = payload["state_dicts"]["model"]
    looks_vae = any("net.encoder_net" in k for k in state_dict.keys())
    OmegaConf.set_struct(cfg.policy, False)
    if looks_vae:
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
    policy.to(device)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    # normalizer
    if hasattr(policy, 'set_normalizer') and "normalizer" in payload["state_dicts"]:
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        nm = LinearNormalizer()
        nm.load_state_dict(payload["state_dicts"]["normalizer"], strict=False)
        nm.to(device)
        policy.set_normalizer(nm)
    return policy, cfg.policy


def _make_fixed_scheduler(k_val, refinement_steps, device):
    """创建一个固定 k 的 scheduler。"""
    class FixedScheduler(torch.nn.Module):
        def __init__(self, kv, rs):
            super().__init__()
            self.k = kv
            self.k_idx = rs.index(kv) if kv in rs else 0
            self.rs = rs
        def select_action(self, obs, init_action, deterministic=True):
            B = obs.shape[0]; dev = obs.device
            steps = torch.full((B,), self.k, dtype=torch.long, device=dev)
            idx = torch.full((B,), self.k_idx, dtype=torch.long, device=dev)
            return steps, idx, torch.zeros(B, device=dev), torch.zeros(B, device=dev)
    return FixedScheduler(k_val, refinement_steps).to(device)


def build_refiner_policy(source_policy, diffusion_policy, k_val, device):
    """构建一个固定 k 的 AdaBridgerPolicy 用于推理。

    source_policy (VAE) 提供 init_action，diffusion_policy 提供 refinement。
    """
    from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy

    policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=diffusion_policy,
        scheduler=_make_fixed_scheduler(k_val, [0, 1, 2, 5], device),
        horizon=diffusion_policy.horizon,
        obs_dim=diffusion_policy.obs_dim,
        action_dim=diffusion_policy.action_dim,
        n_action_steps=diffusion_policy.n_action_steps,
        n_obs_steps=diffusion_policy.n_obs_steps,
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
    )
    return policy


def build_source_policy(source_policy):
    """包装 VAE 为 AdaBridgerPolicy k=0 (只有 source，没有 refiner)。"""
    from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy

    return AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=None,
        scheduler=_make_fixed_scheduler(0, [0], source_policy.device),
        horizon=source_policy.horizon,
        obs_dim=source_policy.obs_dim,
        action_dim=source_policy.action_dim,
        n_action_steps=source_policy.n_action_steps,
        n_obs_steps=source_policy.n_obs_steps,
        refinement_steps=[0], max_refinement_steps=0,
    )


def build_combined_policy(source_policy, refinement_policy, device):
    """包装 VAE + refiner，用于 delta 计算（_sdedit_refine 需要两者都存在）。"""
    from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy

    return AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refinement_policy,
        scheduler=_make_fixed_scheduler(0, [0, 1, 2, 5], device),
        horizon=source_policy.horizon,
        obs_dim=source_policy.obs_dim,
        action_dim=source_policy.action_dim,
        n_action_steps=source_policy.n_action_steps,
        n_obs_steps=source_policy.n_obs_steps,
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
    )


# ──────────────────────────────────────────────────────────────────
# Run episode
# ──────────────────────────────────────────────────────────────────
def run_episode_k(policy, env, device, obs_dim, n_obs_steps, n_action_steps,
                  compute_delta=False, combined_policy=None, save_init_action=False):
    """Run one episode. Returns (states, final_reward, deltas, init_actions).

    当 compute_delta=True 时，combined_policy 必须提供（同时包含 source + refiner），
    用于计算 δ = ||a_init - a_strong|| / D_a。
    当 save_init_action=True 时，额外保存 VAE 的 init_action（展平）。"""
    obs = env.reset()
    policy.reset()
    if combined_policy is not None:
        combined_policy.reset()
    done = False
    all_obs = []
    all_deltas = []
    all_init_actions = []
    all_rewards = []

    while not done:
        raw = obs[np.newaxis].astype(np.float32)
        if raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_tensor = raw[:, :n_obs_steps]
        obs_dict = {"obs": torch.from_numpy(obs_tensor).to(device)}

        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)

        action = result["action"][0, :n_action_steps].cpu().numpy()

        # Save init_action if requested
        if save_init_action:
            a_init = result.get("init_action", result["action"][:, :policy.horizon])
            all_init_actions.append(a_init[0].cpu().numpy().flatten())  # [T*Da]

        # compute δ if needed — use combined_policy which has both source + refiner
        if compute_delta and combined_policy is not None:
            with torch.no_grad():
                a_init = result.get("init_action", result["action"][:, :policy.horizon])
                if a_init is not None:
                    a_strong = combined_policy._sdedit_refine(obs_dict, a_init, 5)
                    delta = (a_init - a_strong).abs().sum(dim=-1).mean().item() / obs_dim
                else:
                    delta = 0.0
            all_deltas.append(delta)

        all_obs.append(obs_tensor[0].copy())  # [T_obs, obs_dim]
        obs, reward, done, info = env.step(action)
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)

    final_reward = float(np.max(all_rewards)) if all_rewards else 0.0
    return all_obs, final_reward, all_deltas, all_init_actions


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--refiner", required=True)
    parser.add_argument("--task", default="pusht")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--output", default="data/counterfactual_labels_pusht.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    print("=" * 60)
    print("Counterfactual Label Collection")
    print(f"  n_episodes = {args.n_episodes}")
    print("=" * 60)

    # 1. Load models
    print("\n[1] Loading models...")
    source, src_cfg = load_policy(args.source, device)
    refiner, ref_cfg = load_policy(args.refiner, device)

    obs_dim = src_cfg.obs_dim
    n_obs_steps = src_cfg.n_obs_steps
    n_action_steps = src_cfg.n_action_steps

    # 2. Setup
    print("\n[2] Setting up...")
    _, ds_norm = _load_normalizer(device)  # 用 dataset normalizer
    if hasattr(source, 'set_normalizer'):
        source.set_normalizer(ds_norm)
    if hasattr(refiner, 'set_normalizer'):
        refiner.set_normalizer(ds_norm)

    env = make_env_pusht()

    # Build policies
    source_policy = build_source_policy(source)                          # k=0 only
    refiner_policy = build_refiner_policy(source, refiner, 5, device)    # VAE + refiner k=5
    combined_policy = build_combined_policy(source, refiner, device)     # for δ computation
    # Give policies the normalizer
    source_policy.set_normalizer(ds_norm)
    refiner_policy.set_normalizer(ds_norm)
    combined_policy.set_normalizer(ds_norm)

    # 3. Collect
    print(f"\n[3] Running {args.n_episodes} episodes...")
    dataset = []
    stats = {"k0_success": 0, "k5_success": 0, "k0_mean_reward": 0,
             "k5_mean_reward": 0, "refine_helps": 0, "delta_mean": 0,
             "delta_refine": 0, "delta_no_refine": 0}

    for ep in tqdm.tqdm(range(args.n_episodes), desc="Counterfactual"):
        seed = 42 + ep

        # (a) k=0 episode, with δ and init_action (use combined_policy for delta)
        env.seed(seed)
        torch.manual_seed(42)
        states_0, r0, deltas, init_actions = run_episode_k(
            source_policy, env, device, obs_dim,
            n_obs_steps, n_action_steps,
            compute_delta=True, combined_policy=combined_policy,
            save_init_action=True,
        )

        # (b) k=5 episode (same seed)
        env.seed(seed)
        torch.manual_seed(42)
        _, r5, _, _ = run_episode_k(
            refiner_policy, env, device, obs_dim,
            n_obs_steps, n_action_steps,
        )

        # Label: r5 > r0 → needs refinement
        label = 1 if r5 > r0 else 0

        for s, d, a_init in zip(
            states_0,
            deltas if deltas else [0.0] * len(states_0),
            init_actions if init_actions else [np.zeros(0)] * len(states_0),
        ):
            dataset.append({
                "obs": s if isinstance(s, np.ndarray) else s.numpy(),
                "delta": d,
                "init_action": a_init,
                "label": label,
                "r0": r0, "r5": r5, "seed": seed,
            })

        stats["k0_success"] += int(r0 > 0.5)
        stats["k5_success"] += int(r5 > 0.5)
        stats["k0_mean_reward"] += r0
        stats["k5_mean_reward"] += r5
        stats["refine_helps"] += label

    env.close()

    # 4. Stats
    n = args.n_episodes
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  k=0 success:    {stats['k0_success']/n:.1%}")
    print(f"  k=5 success:    {stats['k5_success']/n:.1%}")
    print(f"  k=0 mean reward:{stats['k0_mean_reward']/n:.3f}")
    print(f"  k=5 mean reward:{stats['k5_mean_reward']/n:.3f}")
    print(f"  Refine helps:   {stats['refine_helps']/n:.1%} of episodes")
    print(f"  Total states:   {len(dataset)}")

    # Compute δ accuracy vs true label
    all_d = [s["delta"] for s in dataset]
    all_l = [s["label"] for s in dataset]
    if len(all_d) > 0:
        median_d = np.median(all_d)
        d_pred = [1 if d > median_d else 0 for d in all_d]
        d_acc = np.mean([p == l for p, l in zip(d_pred, all_l)])
        print(f"  δ (median split) accuracy vs truth: {d_acc:.1%}")

    # 5. Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({"samples": dataset, "stats": stats, "config": {
        "source": args.source, "refiner": args.refiner,
        "n_episodes": args.n_episodes, "obs_dim": obs_dim,
        "n_obs_steps": n_obs_steps,
    }}, args.output)
    print(f"\nSaved {len(dataset)} labeled states to {args.output}")
    print("=" * 60)


def _load_normalizer(device):
    """Load dataset normalizer for PushT."""
    from omegaconf import OmegaConf as OC
    OC.register_new_resolver("eval", eval, replace=True)
    ds_cfg = OC.create({
        "_target_": "diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset",
        "zarr_path": "data/pusht/pusht_cchi_v7_replay.zarr",
        "horizon": 16, "pad_before": 1, "pad_after": 7,
        "seed": 42, "val_ratio": 0.02, "max_train_episodes": 90,
    })
    ds = hydra.utils.instantiate(ds_cfg)
    nm = ds.get_normalizer()
    nm.to(device)
    return ds, nm


if __name__ == "__main__":
    main()
