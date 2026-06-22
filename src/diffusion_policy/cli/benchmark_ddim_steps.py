#!/usr/bin/env python
"""
DDIM 不同步数推理时间基准测试

测量 Diffusion Policy 在 1/2/3/5/10 DDIM steps 下的推理延迟，
以及 VAE (Source Policy) 的推理时间作为 baseline。

用法:
  python benchmark_ddim_steps.py \
      --refine_ckpt weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
      --source_ckpt weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
      --device cuda:0 --n_warmup 10 --n_runs 100
"""

import os
import sys
import pathlib
import argparse
import time

import numpy as np
import torch
import dill
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)


def benchmark_ddim(refine_policy, obs_dict, ddim_steps, n_warmup, n_runs, device):
    """
    测量指定 DDIM 步数下的 SDEdit 推理时间

    Returns:
        times_ms: list of float (每次推理的毫秒数)
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    noise_scheduler = refine_policy.noise_scheduler
    model = refine_policy.model
    normalizer = refine_policy.normalizer

    # 构建 DDIM scheduler
    if isinstance(noise_scheduler, DDIMScheduler):
        ddim_scheduler = noise_scheduler
    else:
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler.config.num_train_timesteps,
            beta_start=noise_scheduler.config.beta_start,
            beta_end=noise_scheduler.config.beta_end,
            beta_schedule=noise_scheduler.config.beta_schedule,
            clip_sample=noise_scheduler.config.clip_sample,
            set_alpha_to_one=noise_scheduler.config.get('set_alpha_to_one', True),
            prediction_type=noise_scheduler.config.prediction_type,
        )

    nobs = normalizer['obs'].normalize(obs_dict['obs'])
    B = nobs.shape[0]
    To = refine_policy.n_obs_steps
    T = refine_policy.horizon
    Da = refine_policy.action_dim
    Do = refine_policy.obs_dim

    use_global_cond = getattr(refine_policy, 'obs_as_global_cond', False)
    use_local_cond = getattr(refine_policy, 'obs_as_local_cond', False)

    # 准备条件（一次性，不计入推理时间）
    local_cond = None
    global_cond = None

    if use_global_cond:
        global_cond = nobs[:, :To].reshape(B, -1)
        shape = (B, T, Da)
    elif use_local_cond:
        local_cond = torch.zeros(B, T, Do, device=device, dtype=nobs.dtype)
        local_cond[:, :To] = nobs[:, :To]
        shape = (B, T, Da)
    else:
        shape = (B, T, Da + Do)

    # 准备 condition_data / condition_mask
    condition_data = torch.zeros(shape, device=device, dtype=nobs.dtype)
    condition_mask = torch.zeros(shape, device=device, dtype=torch.bool)
    if not use_global_cond and not use_local_cond:
        condition_data[:, :To, Da:] = nobs[:, :To]
        condition_mask[:, :To, Da:] = True

    ddim_scheduler.set_timesteps(ddim_steps)

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            trajectory = torch.randn(shape, device=device, dtype=nobs.dtype)
            # 加噪
            start_t = ddim_scheduler.timesteps[0].item()
            trajectory = ddim_scheduler.add_noise(
                condition_data, trajectory,
                torch.tensor([start_t], device=device).expand(B)
            )
            for t in ddim_scheduler.timesteps:
                trajectory[condition_mask] = condition_data[condition_mask]
                model_output = model(trajectory, t,
                                     local_cond=local_cond,
                                     global_cond=global_cond)
                trajectory = ddim_scheduler.step(model_output, t, trajectory).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]

    if use_cuda:
        torch.cuda.synchronize()

    # Benchmark
    times_ms = []
    for _ in range(n_runs):
        with torch.no_grad():
            noise = torch.randn(shape, device=device, dtype=nobs.dtype)

            if use_cuda:
                e_start = torch.cuda.Event(enable_timing=True)
                e_end = torch.cuda.Event(enable_timing=True)
                e_start.record()
            else:
                t0 = time.perf_counter()

            # 加噪
            start_t = ddim_scheduler.timesteps[0].item()
            trajectory = ddim_scheduler.add_noise(
                condition_data, noise,
                torch.tensor([start_t], device=device).expand(B)
            )
            # 去噪循环
            for t in ddim_scheduler.timesteps:
                trajectory[condition_mask] = condition_data[condition_mask]
                model_output = model(trajectory, t,
                                     local_cond=local_cond,
                                     global_cond=global_cond)
                trajectory = ddim_scheduler.step(model_output, t, trajectory).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]

            if use_cuda:
                e_end.record()
                torch.cuda.synchronize()
                times_ms.append(e_start.elapsed_time(e_end))
            else:
                times_ms.append((time.perf_counter() - t0) * 1000)

    return times_ms


def benchmark_source(source_policy, obs_dict, n_warmup, n_runs, device):
    """测量 VAE Source Policy 推理时间"""
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()

    for _ in range(n_warmup):
        with torch.no_grad():
            source_policy.predict_action(obs_dict)

    if use_cuda:
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        with torch.no_grad():
            if use_cuda:
                e_start = torch.cuda.Event(enable_timing=True)
                e_end = torch.cuda.Event(enable_timing=True)
                e_start.record()
            else:
                t0 = time.perf_counter()

            source_policy.predict_action(obs_dict)

            if use_cuda:
                e_end.record()
                torch.cuda.synchronize()
                times_ms.append(e_start.elapsed_time(e_end))
            else:
                times_ms.append((time.perf_counter() - t0) * 1000)

    return times_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark DDIM inference at different step counts")
    parser.add_argument("--refine_ckpt", type=str,
                        default="weights/lowdim/diffusion/pusht/cnn/latest.ckpt",
                        help="Diffusion policy checkpoint")
    parser.add_argument("--source_ckpt", type=str,
                        default="weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt",
                        help="VAE source policy checkpoint (optional, for baseline)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--ddim_steps", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                        help="DDIM step counts to benchmark")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Warmup iterations (not timed)")
    parser.add_argument("--n_runs", type=int, default=100,
                        help="Timed iterations per step count")
    parser.add_argument("--no_source", action="store_true",
                        help="Skip VAE source policy benchmark")
    args = parser.parse_args()

    device = args.device

    # ── 加载 Diffusion Policy ──
    print(f"Loading Diffusion Policy from {args.refine_ckpt} ...")
    payload = torch.load(open(args.refine_ckpt, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    OmegaConf.set_struct(cfg, False)

    # 通过 workspace class 正确加载（避免 hydra 递归实例化 EMA 等组件失败）
    workspace_cls_name = cfg._target_  # e.g. "diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace.TrainDiffusionUnetLowdimWorkspace"
    mod_path, cls_name = workspace_cls_name.rsplit('.', 1)
    import importlib
    mod = importlib.import_module(mod_path)
    WorkspaceCls = getattr(mod, cls_name)
    workspace = WorkspaceCls(cfg)
    workspace.load_payload(payload)

    # Diffusion workspace 的 model 就是 policy
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        refine_policy = workspace.ema_model
        print("  Using EMA model")
    elif hasattr(workspace, 'model'):
        refine_policy = workspace.model
    else:
        refine_policy = workspace.policy
    refine_policy.to(device)
    refine_policy.eval()
    print(f"  Model: {type(refine_policy).__name__}")
    print(f"  obs_dim={refine_policy.obs_dim}, action_dim={refine_policy.action_dim}, "
          f"horizon={refine_policy.horizon}, n_obs_steps={refine_policy.n_obs_steps}")

    # ── 构造 dummy 输入 ──
    B = args.batch_size
    obs = torch.randn(B, refine_policy.n_obs_steps, refine_policy.obs_dim, device=device)
    obs_dict = {'obs': obs}

    # ── 加载 Source Policy（可选） ──
    source_policy = None
    if not args.no_source and os.path.exists(args.source_ckpt):
        print(f"\nLoading Source Policy from {args.source_ckpt} ...")
        src_payload = torch.load(open(args.source_ckpt, 'rb'), pickle_module=dill)
        src_cfg = src_payload['cfg']
        OmegaConf.set_struct(src_cfg, False)
        src_ws_cls_name = src_cfg._target_
        src_mod_path, src_cls_name = src_ws_cls_name.rsplit('.', 1)
        src_mod = importlib.import_module(src_mod_path)
        SrcWorkspaceCls = getattr(src_mod, src_cls_name)
        src_workspace = SrcWorkspaceCls(src_cfg)
        src_workspace.load_payload(src_payload)
        source_policy = src_workspace.model if hasattr(src_workspace, 'model') else src_workspace.policy
        source_policy.to(device)
        source_policy.eval()
        print(f"  Model: {type(source_policy).__name__}")

    # ══════════════════════════════════════════
    #  Benchmark
    # ══════════════════════════════════════════
    print(f"\nBenchmark config: batch_size={B}, warmup={args.n_warmup}, runs={args.n_runs}, device={device}")
    print(f"{'=' * 65}")

    results = {}

    # Source Policy (VAE baseline)
    if source_policy is not None:
        print(f"\n  Benchmarking VAE Source Policy ...")
        src_obs = {'obs': obs}
        # source policy 需要自己的 normalizer 数据，用 dummy 输入可能不完全准确
        # 但 forward pass 耗时是一样的
        vae_times = benchmark_source(source_policy, src_obs, args.n_warmup, args.n_runs, device)
        results['VAE (k=0)'] = vae_times
        print(f"    VAE:  {np.mean(vae_times):7.2f} ms  ± {np.std(vae_times):.2f} ms  "
              f"({1000/np.mean(vae_times):.1f} Hz)")

    # DDIM at each step count
    for ddim_steps in args.ddim_steps:
        print(f"\n  Benchmarking DDIM {ddim_steps} step{'s' if ddim_steps > 1 else ''} ...")
        times = benchmark_ddim(
            refine_policy, obs_dict, ddim_steps,
            args.n_warmup, args.n_runs, device
        )
        label = f"DDIM {ddim_steps}step"
        results[label] = times
        print(f"    {label}: {np.mean(times):7.2f} ms  ± {np.std(times):.2f} ms  "
              f"({1000/np.mean(times):.1f} Hz)")

    # ══════════════════════════════════════════
    #  Summary Table
    # ══════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print(f"  DDIM Inference Benchmark Summary  (batch_size={B})")
    print(f"{'=' * 65}")
    print(f"  {'Method':<18} {'Mean(ms)':>10} {'Std(ms)':>10} {'Freq(Hz)':>10} {'Speedup':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # 以最慢的 DDIM (最大步数) 作为 speedup 基准
    max_step_key = f"DDIM {max(args.ddim_steps)}step"
    baseline_ms = np.mean(results.get(max_step_key, [1]))

    for label, times in results.items():
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        freq = 1000 / mean_ms if mean_ms > 0 else 0
        speedup = baseline_ms / mean_ms if mean_ms > 0 else 0
        print(f"  {label:<18} {mean_ms:>10.2f} {std_ms:>10.2f} {freq:>10.1f} {speedup:>9.1f}x")

    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
