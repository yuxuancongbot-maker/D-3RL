"""
训练 SI (Stochastic Interpolants) Refiner on PushT

BRIDGER 复现: 冻结 VAE → 采样 prior action → SI 学习 prior→expert 速度场
推理: SDE 正向积分 (vs mode, Euler-Maruyama)

用法:
    python -m diffusion_policy.cli.train_si_refiner \
        --vae-ckpt weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
        --dataset-path data/pusht/pusht_cchi_v7_replay.zarr \
        --output-dir outputs/si_refiner_pusht \
        --num-itr 50000 --batch-size 256 --lr 1e-4
"""

from __future__ import annotations

import argparse, math, os, sys, time
from collections import defaultdict

import dill, hydra, numpy as np, torch, tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from diffusion_policy.model.bridger.si_model import StochasticInterpolants
from diffusion_policy.model.common.normalizer import LinearNormalizer


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────
def load_vae(ckpt_path: str, device: str = "cuda:0"):
    """Load frozen VAE policy. Returns (policy, normalizer, model_cfg)."""
    print(f"Loading VAE from: {ckpt_path}")
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    state_dict = payload["state_dicts"]["model"]

    # Ensure VAE backend
    OmegaConf.set_struct(cfg.policy, False)
    cfg.policy.backend = "vae"
    cfg.policy.model = {
        "_target_": "diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel",
        "action_dim": cfg.policy.action_dim,
        "action_horizon": cfg.policy.horizon,
        "obs_dim": cfg.policy.obs_dim,
        "obs_horizon": cfg.policy.n_obs_steps,
        "latent_dim": 32,
        "layer": 256,
        "use_ema": True,
        "pretrain": False,
        "ckpt_path": None,
    }
    policy = hydra.utils.instantiate(cfg.policy)
    policy.to(device)  # pre-move before load_state_dict to keep modules on device
    policy.load_state_dict(state_dict, strict=False)
    # load_state_dict 内部可能重建子网络 → 回到 CPU → 再移一次
    policy.to(device)
    policy.eval()

    # Normalizer
    normalizer = LinearNormalizer()
    if "normalizer" in payload["state_dicts"]:
        try:
            normalizer.load_state_dict(payload["state_dicts"]["normalizer"], strict=False)
        except Exception:
            print("  [WARN] normalizer load failed, using empty")
    normalizer.to(device)

    for p in policy.parameters():
        p.requires_grad = False

    model_cfg = dict(
        obs_dim=cfg.policy.obs_dim,
        action_dim=cfg.policy.action_dim,
        horizon=cfg.policy.horizon,
        n_obs_steps=cfg.policy.n_obs_steps,
        n_action_steps=cfg.policy.n_action_steps,
    )
    return policy, normalizer, model_cfg


def load_push_t_dataset(dataset_path: str, horizon: int, n_obs_steps: int,
                        n_action_steps: int, n_latency_steps: int = 0):
    """Load PushT lowdim dataset, returns (dataset, normalizer)."""
    print(f"Loading PushT dataset from: {dataset_path}")
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    from omegaconf import OmegaConf as OC
    cfg = OC.create({
        "dataset": {
            "_target_": "diffusion_policy.dataset.pusht_dataset.PushTLowdimDataset",
            "zarr_path": dataset_path,
            "horizon": horizon,
            "pad_before": n_obs_steps - 1 + n_latency_steps,
            "pad_after": n_action_steps - 1,
            "seed": 42,
            "val_ratio": 0.02,
            "max_train_episodes": 90,
        }
    })
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = dataset.get_normalizer()
    return dataset, normalizer


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device)

    # 1. Load VAE
    vae_policy, vae_norm, model_cfg = load_vae(args.vae_ckpt, device)
    print(f"  VAE: obs_dim={model_cfg['obs_dim']}, action_dim={model_cfg['action_dim']}, "
          f"horizon={model_cfg['horizon']}")

    # 2. Load dataset
    dataset, ds_norm = load_push_t_dataset(
        args.dataset_path,
        horizon=model_cfg["horizon"],
        n_obs_steps=model_cfg["n_obs_steps"],
        n_action_steps=model_cfg["n_action_steps"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)
    # Share dataset normalizer with VAE (VAE was trained on the same data)
    normalizer = ds_norm
    normalizer.to(device)
    vae_policy.set_normalizer(normalizer)

    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    print(f"  Normalizer loaded: keys={list(normalizer.params_dict.keys())}")

    # 3. Create SI model
    obs_horizon = model_cfg["n_obs_steps"]
    net_type = args.net_type
    if net_type == "transformer":
        net_kwargs = dict(
            horizon=model_cfg["horizon"],
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_emb=args.n_emb,
            n_cond_layers=args.n_cond_layers,
            p_drop_emb=args.p_drop,
            p_drop_attn=args.p_drop,
        )
    else:
        net_kwargs = dict(
            down_dims=args.down_dims,
            kernel_size=args.kernel_size,
            n_groups=args.n_groups,
        )

    si_model = StochasticInterpolants(
        action_dim=model_cfg["action_dim"],
        obs_dim=model_cfg["obs_dim"],
        obs_horizon=obs_horizon,
        interpolant_type=args.interpolant_type,
        gamma_type=args.gamma_type,
        epsilon_type=args.epsilon_type,
        sde_type="vs",
        beta_max=args.beta_max,
        net_type=net_type,
        net_kwargs=net_kwargs,
    ).to(device)
    si_model.ema.to(device)

    n_params = sum(p.numel() for p in si_model.parameters())
    print(f"  SI model: {n_params/1e6:.1f}M params")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(si_model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_itr, eta_min=args.lr * 0.01)

    os.makedirs(args.output_dir, exist_ok=True)

    # wandb
    use_wandb = args.use_wandb and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, dir=args.output_dir,
                   config=vars(args))
        wandb.watch(si_model.net, log="all", log_freq=500)

    # 5. Training loop
    print(f"\nTraining for {args.num_itr} iterations...")
    pbar = tqdm.tqdm(total=args.num_itr, desc="SI train")
    dataloader_iter = iter(dataloader)
    stats = defaultdict(float)
    best_loss = float("inf")
    print_every = max(args.num_itr // 50, 1)

    for it in range(args.num_itr):
        # Get next batch
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        nobs = batch["obs"].to(device)        # [B, T, Do]
        naction = batch["action"].to(device)   # [B, T, Da]

        # Normalize
        nobs_norm = normalizer["obs"].normalize(nobs)
        naction_norm = normalizer["action"].normalize(naction)

        # VAE prior (stochastic sample, in normalized space)
        B = nobs.shape[0]
        Do = model_cfg["obs_dim"]
        # 条件格式：UNet 用 flatten [B, Dc]，Transformer 用 [B, To, Do]
        if net_type == "transformer":
            cond_input = nobs_norm[:, :model_cfg["n_obs_steps"]]  # [B, To, Do]
            obs_flat_for_vae = cond_input.reshape(B, -1)  # VAE.sample 用 flatten
        else:
            cond_input = nobs_norm[:, :model_cfg["n_obs_steps"]].reshape(B, -1)  # [B, Dc]
            obs_flat_for_vae = cond_input

        with torch.no_grad():
            prior_action = vae_policy.model.sample(obs_flat_for_vae)

        optimizer.zero_grad()
        loss, loss_info = si_model.get_loss(
            x0=prior_action,
            x1=naction_norm,
            cond=cond_input,
            return_info=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(si_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        si_model.ema.update()

        # Stats
        stats["loss"] += loss.item()
        stats["v_loss"] += loss_info["v_loss"]
        stats["s_loss"] += loss_info["s_loss"]
        stats["b_loss"] += loss_info["b_loss"]

        if (it + 1) % print_every == 0 or it == 0:
            n = print_every if it > 0 else 1
            avg = {k: v / n for k, v in stats.items()}
            pbar.set_postfix(
                loss=f"{avg['loss']:.3f}", v=f"{avg['v_loss']:.3f}",
                s=f"{avg['s_loss']:.3f}", b=f"{avg['b_loss']:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )
            if use_wandb:
                wandb.log({"train/" + k: v for k, v in avg.items()} |
                          {"train/lr": scheduler.get_last_lr()[0]},
                          step=it)
            pbar.update(print_every)
            stats = defaultdict(float)

        # Save
        if (it + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"si_refiner_{it+1:06d}.pt")
            si_model.save_checkpoint(ckpt_path)
            avg_loss = loss.item()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output_dir, "si_refiner_best.pt")
                si_model.save_checkpoint(best_path)

    # Final save
    final_path = os.path.join(args.output_dir, "si_refiner_final.pt")
    si_model.save_checkpoint(final_path)
    pbar.close()
    if use_wandb:
        wandb.finish()
    print(f"\nDone! Models saved to {args.output_dir}")
    print(f"  Best loss: {best_loss:.4f}")


# ──────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train SI Refiner on PushT")
    parser.add_argument("--vae-ckpt", required=True, help="VAE source policy checkpoint")
    parser.add_argument("--dataset-path", default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--output-dir", default="outputs/si_refiner_pusht")
    parser.add_argument("--device", default="cuda:0")

    # SI config
    parser.add_argument("--interpolant-type", default="power3",
                        choices=["linear", "power3", "reverse_power3",
                                 "power4", "reverse_power4"])
    parser.add_argument("--gamma-type", default="(2t(t-1))^0.5",
                        choices=["(2t(t-1))^0.5", "2^0.5*t(t-1)", "(1-t)^2(2t)^0.5"])
    parser.add_argument("--epsilon-type", default="1-t")
    parser.add_argument("--beta-max", type=float, default=0.03)

    # Network
    parser.add_argument("--net-type", default="unet", choices=["unet", "transformer"])
    # UNet
    parser.add_argument("--down-dims", type=int, nargs="+", default=[256, 512, 512])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    # Transformer
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-emb", type=int, default=768)
    parser.add_argument("--n-cond-layers", type=int, default=2)
    parser.add_argument("--p-drop", type=float, default=0.1)

    # Training
    parser.add_argument("--num-itr", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", default="step", choices=["cosine", "step"])
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--lr-gamma", type=float, default=0.99)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="si-refiner")
    parser.add_argument("--wandb-name", default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
