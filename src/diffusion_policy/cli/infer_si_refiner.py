"""
SI Refiner 推理脚本 —— 加载 VAE + SI refiner 在 PushT 上推理

用法:
    python -m diffusion_policy.cli.infer_si_refiner \
        --vae-ckpt weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --si-ckpt outputs/si_refiner_pusht/si_refiner_final.pt \
        --k 5 --n-episodes 10
"""

import argparse, copy, os, sys, time
from collections import Counter

import dill, hydra, numpy as np, torch
from omegaconf import OmegaConf

from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.model.bridger.si_model import StochasticInterpolants


# ──────────────────────────────────────────────────────────────────
def load_vae(ckpt_path, device):
    print(f"  VAE: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    state_dict = payload["state_dicts"]["model"]
    OmegaConf.set_struct(cfg.policy, False)
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
    return policy, cfg.policy


def load_si_refiner(ckpt_path, model_cfg, device, net_type="transformer", net_kwargs=None):
    """加载 SI refiner。"""
    print(f"  SI refiner: {ckpt_path}  (net_type={net_type})")
    net_kwargs = net_kwargs or {}
    if net_type == "transformer":
        net_kwargs.setdefault("n_emb", 768)
        net_kwargs.setdefault("n_layer", 8)
        net_kwargs.setdefault("n_head", 12)
        net_kwargs.setdefault("n_cond_layers", 2)
        net_kwargs.setdefault("horizon", model_cfg.horizon)

    si = StochasticInterpolants(
        action_dim=model_cfg.action_dim,
        obs_dim=model_cfg.obs_dim,
        obs_horizon=model_cfg.n_obs_steps,
        sde_type="vs",
        net_type=net_type,
        net_kwargs=net_kwargs,
    )
    si.load_checkpoint(ckpt_path, device=device)
    si.to(device)
    si.ema.to(device)
    si.eval()
    for p in si.parameters():
        p.requires_grad = False
    return si


# ──────────────────────────────────────────────────────────────────
def make_env():
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    env = PushTKeypointsEnv()
    return MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)


def run_episode(policy, env, obs_dim, n_obs_steps, n_action_steps, device):
    policy.reset()
    obs = env.reset()
    done = False
    k_hist = []
    all_rewards = []
    timings = []

    while not done:
        raw = obs[np.newaxis].astype(np.float32)
        if raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_dict = {"obs": torch.from_numpy(raw[:, :n_obs_steps]).to(device)}

        t0 = time.perf_counter()
        with torch.no_grad():
            r = policy.predict_action(obs_dict, return_intermediate=True)
        timings.append((time.perf_counter() - t0) * 1000)

        k = int(r["refinement_steps"].flatten()[0].item())
        k_hist.append(k)
        act = r["action"][0, :n_action_steps].cpu().numpy()
        obs, reward, done, info = env.step(act)
        if reward is not None:
            all_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)

    final = float(np.max(all_rewards)) if all_rewards else 0.0
    return {
        "reward": final, "success": final > 0.5,
        "avg_k": np.mean(k_hist) if k_hist else 0,
        "k_dist": dict(Counter(k_hist)),
        "avg_ms": np.mean(timings) if timings else 0,
    }


# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-ckpt", required=True)
    parser.add_argument("--si-ckpt", required=True)
    parser.add_argument("--k", type=int, default=5, help="num SI steps (1-10)")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--net-type", default="transformer", choices=["unet", "transformer"])
    parser.add_argument("--n-emb", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=12)
    args = parser.parse_args()

    device = args.device
    print("=" * 60)
    print(f"SI Refiner Inference  (k={args.k})")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading models...")
    vae, vae_cfg = load_vae(args.vae_ckpt, device)
    net_kwargs = dict(n_layer=args.n_layer, n_head=args.n_head, n_emb=args.n_emb)
    si_refiner = load_si_refiner(args.si_ckpt, vae_cfg, device, net_type=args.net_type,
                                  net_kwargs=net_kwargs)

    # 2. Normalizer: 从 PushT dataset 构建（与训练一致）
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    from omegaconf import OmegaConf as OC
    ds_cfg = OC.create({
        "_target_": "diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset",
        "zarr_path": "data/pusht/pusht_cchi_v7_replay.zarr",
        "horizon": vae_cfg.horizon,
        "pad_before": vae_cfg.n_obs_steps - 1,
        "pad_after": vae_cfg.n_action_steps - 1,
        "seed": 42, "val_ratio": 0.02, "max_train_episodes": 90,
    })
    ds = hydra.utils.instantiate(ds_cfg)
    normalizer = ds.get_normalizer()
    normalizer.to(device)
    vae.set_normalizer(normalizer)
    # SI refiner 需要 normalizer 属性（_sdedit_refine fast-path 使用）
    si_refiner.normalizer = normalizer
    model_cfg = dict(
        obs_dim=vae_cfg.obs_dim,
        action_dim=vae_cfg.action_dim,
        horizon=vae_cfg.horizon,
        n_obs_steps=vae_cfg.n_obs_steps,
        n_action_steps=vae_cfg.n_action_steps,
    )
    print(f"  Model: obs_dim={model_cfg['obs_dim']}, action_dim={model_cfg['action_dim']}, "
          f"horizon={model_cfg['horizon']}")

    # 3. Fixed scheduler
    class FixedScheduler(torch.nn.Module):
        def __init__(self, kv, rsteps):
            super().__init__()
            self.k, self.rsteps = kv, rs
            self.k_idx = rs.index(kv)
            self.register_buffer("rs_t", torch.tensor(rs, dtype=torch.long))
        def select_action(self, obs, init_action, deterministic=True):
            B = obs.shape[0]
            dev = obs.device
            idx = torch.full((B,), self.k_idx, dtype=torch.long, device=dev)
            st = torch.full((B,), self.k, dtype=torch.long, device=dev)
            return st, idx, torch.zeros(B, device=dev), torch.zeros(B, device=dev)

    rs = sorted(set([0, 1, 2, 5, args.k]))
    sched = FixedScheduler(args.k, rs).to(device)

    # 4. Build policy
    print("\n[2] Building AdaBridgerPolicy (SI refiner)...")
    policy = AdaBridgerPolicy(
        source_policy=vae,
        refinement_policy=si_refiner,  # ← has .sample() → SI fast-path
        scheduler=sched,
        horizon=model_cfg["horizon"],
        obs_dim=model_cfg["obs_dim"],
        action_dim=model_cfg["action_dim"],
        n_action_steps=model_cfg["n_action_steps"],
        n_obs_steps=model_cfg["n_obs_steps"],
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
    )
    policy.set_normalizer(vae.normalizer)
    policy.to(device)
    policy.eval()

    # 5. Infer
    print(f"\n[3] Running {args.n_episodes} episodes...")
    env = make_env()
    results = []
    for ep in range(args.n_episodes):
        env.seed(42 + ep)
        r = run_episode(policy, env, model_cfg["obs_dim"],
                        model_cfg["n_obs_steps"], model_cfg["n_action_steps"], device)
        results.append(r)
        print(f"  Ep {ep+1:2d}: reward={r['reward']:.3f}  {'✓' if r['success'] else '✗'}  "
              f"k={r['avg_k']:.1f}  {r['avg_ms']:.1f}ms  dist={r['k_dist']}")

    env.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    succ = [r["success"] for r in results]
    rews = [r["reward"] for r in results]
    mss = [r["avg_ms"] for r in results]
    print(f"  Success: {np.mean(succ):.1%}")
    print(f"  Mean reward: {np.mean(rews):.3f} ± {np.std(rews):.3f}")
    print(f"  Mean inference: {np.mean(mss):.1f} ms  (~{1000/np.mean(mss):.0f} Hz)")
    print("=" * 60)


if __name__ == "__main__":
    main()
