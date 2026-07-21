#!/usr/bin/env python
"""Evaluate the PushT lowdim Weighted PPO control checkpoint.

This is a PushT-only, checkpoint-override version of eval_lowdim_pegrad_meanstd.py.
It avoids inline shell/Python quoting so tmux evaluation is reproducible.
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import dill
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)


def sync_if_cuda(device: str):
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_workspace(ckpt_path: pathlib.Path, device: str):
    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace

    payload = torch.load(str(ckpt_path), map_location=device, pickle_module=dill)
    ws = TrainAdaBridgerWorkspace(payload['cfg'])
    ws.policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
    ws.policy.to(device)
    ws.policy.eval()
    ws.policy.scheduler_deterministic = True
    try:
        normalizer = ws.policy.source_policy.normalizer
        _ = normalizer['obs']
    except Exception:
        normalizer = ws.policy.refinement_policy.normalizer
    ws.policy.set_normalizer(normalizer)
    return ws, payload['cfg']


def run_pusht_episode(policy, cfg, seed: int, device: str):
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    n_obs_steps = int(getattr(cfg, 'n_obs_steps', 2))
    n_action_steps = int(getattr(cfg, 'n_action_steps', 8))
    obs_dim = int(getattr(cfg, 'obs_dim', 20))

    env = PushTKeypointsEnv()
    env = MultiStepWrapper(
        env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=300,
    )
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    policy.reset()

    done = False
    rewards, episode_ks, episode_times = [], [], []
    while not done:
        raw = obs[np.newaxis].astype(np.float32)
        if raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_dict = {'obs': torch.from_numpy(raw[:, :n_obs_steps]).float().to(device)}

        sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)
        sync_if_cuda(device)
        episode_times.append((time.perf_counter() - t0) * 1000)
        episode_ks.append(int(result['refinement_steps'].flatten()[0].item()))

        action = result['action'][0, :n_action_steps].detach().cpu().numpy()
        obs, reward, done, info = env.step(action)
        if reward is not None:
            rewards.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))

    env.close()
    max_reward = float(np.max(rewards)) if rewards else 0.0
    return {
        'success': float(max_reward > 0.5),
        'reward': max_reward,
        'avg_k': float(np.mean(episode_ks)) if episode_ks else 0.0,
        'latency_ms': float(np.mean(episode_times)) if episode_times else 0.0,
        'n_calls': len(episode_ks),
        'k_counts': {str(k): int(episode_ks.count(k)) for k in sorted(set(episode_ks))},
    }


def merge_counts(episodes: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ep in episodes:
        for k, v in ep['k_counts'].items():
            out[k] = out.get(k, 0) + int(v)
    return out


def summarize_run(episodes: List[Dict]) -> Dict:
    return {
        'success': float(np.mean([x['success'] for x in episodes])),
        'mean_reward': float(np.mean([x['reward'] for x in episodes])),
        'avg_nfe': float(np.mean([x['avg_k'] for x in episodes])),
        'latency_ms': float(np.mean([x['latency_ms'] for x in episodes])),
        'total_calls': int(sum(x['n_calls'] for x in episodes)),
        'k_counts': merge_counts(episodes),
        'episodes': episodes,
    }


def fmt_mean_std(vals: List[float], scale: float = 1.0, suffix: str = '') -> str:
    vals = np.asarray(vals, dtype=np.float64) * scale
    return f'{vals.mean():.1f}±{vals.std():.1f}{suffix}'


def fmt_mean_std2(vals: List[float], suffix: str = '') -> str:
    vals = np.asarray(vals, dtype=np.float64)
    return f'{vals.mean():.2f}±{vals.std():.2f}{suffix}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs' / 'train' / 'checkpoints' / 'latest.ckpt'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--n-episodes', type=int, default=50)
    parser.add_argument('--seed-starts', nargs='+', type=int, default=[100000, 100050, 100100])
    parser.add_argument('--output-dir', default=str(ROOT / 'result' / 'weighted_ppo_pusht_eval'))
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()

    if len(args.seed_starts) < args.n_runs:
        raise ValueError('--seed-starts must provide at least --n-runs values')

    ckpt_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Checkpoint: {ckpt_path}', flush=True)
    print(f'Output dir: {out_dir.resolve()}', flush=True)
    print(f'Device: {args.device}', flush=True)
    print(f'Protocol: {args.n_runs} runs x {args.n_episodes} episodes; seed_starts={args.seed_starts[:args.n_runs]}', flush=True)

    ws, cfg = load_workspace(ckpt_path, args.device)

    run_summaries = []
    for run_idx, seed_start in enumerate(args.seed_starts[:args.n_runs]):
        print(f'Run {run_idx + 1}/{args.n_runs}: seeds {seed_start}-{seed_start + args.n_episodes - 1}', flush=True)
        episodes = []
        for ep_idx, seed in enumerate(range(seed_start, seed_start + args.n_episodes)):
            ep_result = run_pusht_episode(ws.policy, cfg, seed, args.device)
            episodes.append(ep_result)
            if (ep_idx + 1) % args.print_every == 0 or ep_idx + 1 == args.n_episodes:
                cur = summarize_run(episodes)
                acc = cur['success'] * 100.0
                latency_ms = cur['latency_ms']
                avg_nfe = cur['avg_nfe']
                print(f'  {ep_idx + 1:3d}/{args.n_episodes}: acc={acc:.1f}% lat={latency_ms:.1f}ms nfe={avg_nfe:.2f}', flush=True)
        summary = summarize_run(episodes)
        summary['seed_start'] = int(seed_start)
        summary['seed_end'] = int(seed_start + args.n_episodes - 1)
        run_summaries.append(summary)
        acc = summary['success'] * 100.0
        latency_ms = summary['latency_ms']
        avg_nfe = summary['avg_nfe']
        calls = summary['total_calls']
        print(f'Run {run_idx + 1} done: acc={acc:.1f}% lat={latency_ms:.1f}ms nfe={avg_nfe:.2f} calls={calls}', flush=True)

    acc_vals = [r['success'] for r in run_summaries]
    lat_vals = [r['latency_ms'] for r in run_summaries]
    nfe_vals = [r['avg_nfe'] for r in run_summaries]
    reward_vals = [r['mean_reward'] for r in run_summaries]

    result = {
        'task': 'PushT',
        'task_key': 'pusht',
        'method': 'Weighted PPO',
        'checkpoint': str(ckpt_path),
        'checkpoint_selection': 'final/latest checkpoint after full cost warmup and success-guard checkpoint save',
        'n_runs': len(run_summaries),
        'n_episodes_per_run': int(args.n_episodes),
        'seed_starts': [int(x) for x in args.seed_starts[:args.n_runs]],
        'summary': {
            'acc_percent': fmt_mean_std(acc_vals, scale=100.0, suffix='%'),
            'latency_ms': fmt_mean_std(lat_vals),
            'avg_nfe': fmt_mean_std2(nfe_vals),
            'mean_reward': fmt_mean_std2(reward_vals),
            'acc_mean': float(np.mean(acc_vals)),
            'acc_std': float(np.std(acc_vals)),
            'latency_mean': float(np.mean(lat_vals)),
            'latency_std': float(np.std(lat_vals)),
            'avg_nfe_mean': float(np.mean(nfe_vals)),
            'avg_nfe_std': float(np.std(nfe_vals)),
        },
        'runs': run_summaries,
    }

    json_path = out_dir / 'results.json'
    md_path = out_dir / 'results.md'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(md_path, 'w') as f:
        s = result['summary']
        f.write('# Weighted PPO PushT lowdim evaluation\n\n')
        f.write(f'- Checkpoint: `{ckpt_path}`\n')
        f.write('- Selection: final/latest checkpoint after full cost warmup and success-guard checkpoint save\n')
        f.write(f'- Protocol: {args.n_runs} runs × {args.n_episodes} episodes; seeds {args.seed_starts[:args.n_runs]}\n\n')
        f.write('| Method | Task | Success | Latency(ms) | Avg NFE |\n')
        f.write('|---|---|---:|---:|---:|\n')
        f.write(f'| Weighted PPO | PushT lowdim | {s["acc_percent"]} | {s["latency_ms"]} | {s["avg_nfe"]} |\n')

    s = result['summary']
    print('FINAL Weighted PPO PushT:', flush=True)
    print(f'  acc={s["acc_percent"]} lat={s["latency_ms"]}ms nfe={s["avg_nfe"]} reward={s["mean_reward"]}', flush=True)
    print(f'  saved {json_path}', flush=True)
    print(f'  saved {md_path}', flush=True)


if __name__ == '__main__':
    main()
