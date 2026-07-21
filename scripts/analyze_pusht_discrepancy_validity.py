#!/usr/bin/env python
"""PushT discrepancy-label validity analysis.

For matched PushT initial states, measure the action discrepancy between k=0
(source action) and k=5 (SDEdit-refined action), then run paired fixed-k full
rollouts from the same environment seed. Reports Spearman correlation between
initial discrepancy and k=5 vs k=0 outcome gain, plus low/mid/high discrepancy
bins.
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


class FixedScheduler(torch.nn.Module):
    def __init__(self, k: int, refinement_steps: List[int]):
        super().__init__()
        self.k = int(k)
        self.refinement_steps = list(refinement_steps)
        if self.k not in self.refinement_steps:
            self.refinement_steps.append(self.k)
        self.k_idx = self.refinement_steps.index(self.k)
        self.register_buffer(
            'refinement_steps_tensor',
            torch.tensor(self.refinement_steps, dtype=torch.long),
        )

    def select_action(self, obs, init_action, deterministic=True):
        if isinstance(init_action, torch.Tensor):
            batch_size = init_action.shape[0]
            device = init_action.device
        elif isinstance(obs, torch.Tensor):
            batch_size = obs.shape[0]
            device = obs.device
        else:
            first = next(iter(obs.values()))
            batch_size = first.shape[0]
            device = first.device
        steps = torch.full((batch_size,), self.k, dtype=torch.long, device=device)
        idx = torch.full((batch_size,), self.k_idx, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch_size, device=device)
        value = torch.zeros(batch_size, device=device)
        return steps, idx, log_prob, value


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


def make_env(cfg, seed: int):
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    n_obs_steps = int(getattr(cfg, 'n_obs_steps', 2))
    n_action_steps = int(getattr(cfg, 'n_action_steps', 8))
    env = PushTKeypointsEnv()
    env = MultiStepWrapper(
        env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=300,
    )
    env.seed(seed)
    obs = env.reset()
    return env, obs


def obs_to_dict(obs, cfg, device: str):
    n_obs_steps = int(getattr(cfg, 'n_obs_steps', 2))
    obs_dim = int(getattr(cfg, 'obs_dim', 20))
    raw = obs[np.newaxis].astype(np.float32)
    if raw.shape[-1] == obs_dim * 2:
        raw = raw[..., :obs_dim]
    return {'obs': torch.from_numpy(raw[:, :n_obs_steps]).float().to(device)}


def initial_action_discrepancy(policy, cfg, seed: int, device: str, strong_k: int):
    env, obs = make_env(cfg, seed)
    try:
        policy.reset()
        obs_dict = obs_to_dict(obs, cfg, device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        with torch.no_grad():
            source_result = policy.source_policy.predict_action(obs_dict)
            init_action = source_result['action_pred']
            refined_action = policy._sdedit_refine(obs_dict, init_action, strong_k)
        diff = (init_action - refined_action).abs().mean().item()
        return float(diff)
    finally:
        env.close()


def run_fixed_episode(policy, cfg, seed: int, device: str, fixed_k: int):
    old_scheduler = policy.scheduler
    old_det = policy.scheduler_deterministic
    policy.scheduler = FixedScheduler(fixed_k, list(policy.refinement_steps)).to(device)
    policy.scheduler_deterministic = True
    try:
        env, obs = make_env(cfg, seed)
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            policy.reset()
            done = False
            rewards, latencies = [], []
            n_calls = 0
            while not done:
                obs_dict = obs_to_dict(obs, cfg, device)
                sync_if_cuda(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    result = policy.predict_action(obs_dict, return_intermediate=True)
                sync_if_cuda(device)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                n_calls += 1
                action = result['action'][0, :int(getattr(cfg, 'n_action_steps', 8))].detach().cpu().numpy()
                obs, reward, done, info = env.step(action)
                if reward is not None:
                    rewards.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))
            max_reward = float(np.max(rewards)) if rewards else 0.0
            return {
                'reward': max_reward,
                'success': float(max_reward > 0.5),
                'latency_ms': float(np.mean(latencies)) if latencies else 0.0,
                'n_calls': int(n_calls),
            }
        finally:
            env.close()
    finally:
        policy.scheduler = old_scheduler
        policy.scheduler_deterministic = old_det


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_bin(name: str, rows: List[Dict]) -> Dict:
    if not rows:
        return {
            'name': name,
            'n': 0,
            'discrepancy_mean': None,
            'reward_gain_mean': None,
            'success_gain_mean': None,
            'k0_success_mean': None,
            'k5_success_mean': None,
            'k0_reward_mean': None,
            'k5_reward_mean': None,
        }
    return {
        'name': name,
        'n': int(len(rows)),
        'discrepancy_mean': float(np.mean([r['discrepancy'] for r in rows])),
        'reward_gain_mean': float(np.mean([r['reward_gain'] for r in rows])),
        'success_gain_mean': float(np.mean([r['success_gain'] for r in rows])),
        'k0_success_mean': float(np.mean([r['k0']['success'] for r in rows])),
        'k5_success_mean': float(np.mean([r['k5']['success'] for r in rows])),
        'k0_reward_mean': float(np.mean([r['k0']['reward'] for r in rows])),
        'k5_reward_mean': float(np.mean([r['k5']['reward'] for r in rows])),
    }


def make_bins(rows: List[Dict]) -> Dict[str, Dict]:
    ordered = sorted(rows, key=lambda r: r['discrepancy'])
    splits = np.array_split(ordered, 3)
    names = ['low', 'mid', 'high']
    return {name: summarize_bin(name, list(split)) for name, split in zip(names, splits)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs' / 'train' / 'checkpoints' / 'latest.ckpt'))
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--n-episodes', type=int, default=150)
    parser.add_argument('--seed-start', type=int, default=100000)
    parser.add_argument('--strong-k', type=int, default=5)
    parser.add_argument('--output-dir', default=str(ROOT / 'result' / 'pusht_discrepancy_validity'))
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('PushT discrepancy validity analysis', flush=True)
    print(f'  checkpoint = {ckpt_path}', flush=True)
    print(f'  device = {args.device}', flush=True)
    print(f'  seeds = {args.seed_start}-{args.seed_start + args.n_episodes - 1}', flush=True)
    print(f'  paired outcome = k={args.strong_k} minus k=0', flush=True)

    ws, cfg = load_workspace(ckpt_path, args.device)
    policy = ws.policy

    rows = []
    for idx, seed in enumerate(range(args.seed_start, args.seed_start + args.n_episodes)):
        discrepancy = initial_action_discrepancy(policy, cfg, seed, args.device, args.strong_k)
        k0 = run_fixed_episode(policy, cfg, seed, args.device, fixed_k=0)
        k5 = run_fixed_episode(policy, cfg, seed, args.device, fixed_k=args.strong_k)
        row = {
            'seed': int(seed),
            'discrepancy': float(discrepancy),
            'k0': k0,
            'k5': k5,
            'reward_gain': float(k5['reward'] - k0['reward']),
            'success_gain': float(k5['success'] - k0['success']),
        }
        rows.append(row)
        if (idx + 1) % args.print_every == 0 or idx + 1 == args.n_episodes:
            rho_reward = spearman([r['discrepancy'] for r in rows], [r['reward_gain'] for r in rows])
            rho_success = spearman([r['discrepancy'] for r in rows], [r['success_gain'] for r in rows])
            print(
                f'  {idx + 1:3d}/{args.n_episodes}: '
                f'rho_reward={rho_reward:.3f} rho_success={rho_success:.3f} '
                f'mean_gain={np.mean([r["reward_gain"] for r in rows]):.3f}',
                flush=True,
            )

    discrepancies = [r['discrepancy'] for r in rows]
    reward_gains = [r['reward_gain'] for r in rows]
    success_gains = [r['success_gain'] for r in rows]
    bins = make_bins(rows)
    summary = {
        'n': int(len(rows)),
        'spearman_discrepancy_reward_gain': spearman(discrepancies, reward_gains),
        'spearman_discrepancy_success_gain': spearman(discrepancies, success_gains),
        'discrepancy_mean': float(np.mean(discrepancies)),
        'discrepancy_std': float(np.std(discrepancies)),
        'reward_gain_mean': float(np.mean(reward_gains)),
        'reward_gain_std': float(np.std(reward_gains)),
        'success_gain_mean': float(np.mean(success_gains)),
        'success_gain_std': float(np.std(success_gains)),
        'k0_success_mean': float(np.mean([r['k0']['success'] for r in rows])),
        'k5_success_mean': float(np.mean([r['k5']['success'] for r in rows])),
        'k0_reward_mean': float(np.mean([r['k0']['reward'] for r in rows])),
        'k5_reward_mean': float(np.mean([r['k5']['reward'] for r in rows])),
        'bins': bins,
    }
    result = {
        'task': 'PushT lowdim',
        'analysis': 'strict initial-state paired rollout validity',
        'checkpoint': str(ckpt_path),
        'seed_start': int(args.seed_start),
        'n_episodes': int(args.n_episodes),
        'strong_k': int(args.strong_k),
        'summary': summary,
        'rows': rows,
    }

    json_path = out_dir / 'results.json'
    md_path = out_dir / 'results.md'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    with open(md_path, 'w') as f:
        f.write('# PushT discrepancy label validity\n\n')
        f.write(f'- Checkpoint: `{ckpt_path}`\n')
        f.write(f'- Protocol: matched initial states, seeds {args.seed_start}-{args.seed_start + args.n_episodes - 1}; gain = fixed k={args.strong_k} rollout minus fixed k=0 rollout.\n\n')
        f.write('| Metric | Value |\n')
        f.write('|---|---:|\n')
        f.write(f'| Spearman(discrepancy, reward gain) | {summary["spearman_discrepancy_reward_gain"]:.3f} |\n')
        f.write(f'| Spearman(discrepancy, success gain) | {summary["spearman_discrepancy_success_gain"]:.3f} |\n')
        f.write(f'| Mean reward gain | {summary["reward_gain_mean"]:.3f}±{summary["reward_gain_std"]:.3f} |\n')
        f.write(f'| Mean success gain | {summary["success_gain_mean"]:.3f}±{summary["success_gain_std"]:.3f} |\n')
        f.write(f'| k=0 success | {summary["k0_success_mean"]*100:.1f}% |\n')
        f.write(f'| k={args.strong_k} success | {summary["k5_success_mean"]*100:.1f}% |\n\n')
        f.write('| Discrepancy bin | n | discrepancy | reward gain | success gain | k=0 succ | k=5 succ |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for name in ['low', 'mid', 'high']:
            b = bins[name]
            f.write(
                f'| {name} | {b["n"]} | {b["discrepancy_mean"]:.4f} | '
                f'{b["reward_gain_mean"]:.3f} | {b["success_gain_mean"]:.3f} | '
                f'{b["k0_success_mean"]*100:.1f}% | {b["k5_success_mean"]*100:.1f}% |\n'
            )

    print('FINAL PushT discrepancy validity:', flush=True)
    print(f'  rho_reward={summary["spearman_discrepancy_reward_gain"]:.3f}', flush=True)
    print(f'  rho_success={summary["spearman_discrepancy_success_gain"]:.3f}', flush=True)
    print(f'  reward_gain={summary["reward_gain_mean"]:.3f}±{summary["reward_gain_std"]:.3f}', flush=True)
    print(f'  k0_success={summary["k0_success_mean"]*100:.1f}% k{args.strong_k}_success={summary["k5_success_mean"]*100:.1f}%', flush=True)
    print(f'  saved {json_path}', flush=True)
    print(f'  saved {md_path}', flush=True)


if __name__ == '__main__':
    main()
