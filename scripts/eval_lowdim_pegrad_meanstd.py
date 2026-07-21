#!/usr/bin/env python
"""Evaluate selected lowdim Ada-BRIDGER/PEGrad checkpoints over seed blocks.

Default rows match the lowdim PEGrad result table used in the paper notes:
PushT, Can, Lift, Square, Transport, and Tool-Hang. Each run uses a distinct
contiguous seed block and reports mean±std across runs for success, latency,
and average refinement steps (NFE/k).
"""

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)


TASKS = {
    # task_key: display_name, checkpoint, epoch_label, env_type, dataset, max_steps
    'pusht': {
        'name': 'PushT',
        'ckpt': 'outputs/pusht/checkpoints/epoch=0399-test_mean_score=0.681.ckpt',
        'epoch': '399',
        'env_type': 'pusht',
        'dataset': None,
        'max_steps': 300,
    },
    'can': {
        'name': 'Can',
        'ckpt': 'outputs/can/checkpoints/epoch=0299-test_mean_score=0.980.ckpt',
        'epoch': 'latest',
        'env_type': 'robo',
        'dataset': 'can',
        'max_steps': 400,
    },
    'lift': {
        'name': 'Lift',
        'ckpt': 'outputs/lift/checkpoints/epoch=0249-test_mean_score=1.000.ckpt',
        'epoch': '49',
        'env_type': 'robo',
        'dataset': 'lift',
        'max_steps': 400,
    },
    'square': {
        'name': 'Square',
        'ckpt': 'outputs/square/checkpoints/latest.ckpt',
        'epoch': 'latest',
        'env_type': 'robo',
        'dataset': 'square',
        'max_steps': 400,
    },
    # The pasted historical row reports Test Mean Score 0.720, matching epoch=0199.
    'transport': {
        'name': 'Transport',
        'ckpt': 'outputs/train_transport/checkpoints/epoch=0199-test_mean_score=0.720.ckpt',
        'epoch': 'latest',
        'env_type': 'robo',
        'dataset': 'transport',
        'max_steps': 700,
    },
    'tool_hang': {
        'name': 'ToolHang',
        'ckpt': 'outputs/train_toolhang/checkpoints/latest.ckpt',
        'epoch': 'latest',
        'env_type': 'robo',
        'dataset': 'tool_hang',
        'max_steps': 700,
    },
}


ROBO_OBS_KEYS = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
TRANSPORT_OBS_KEYS = [
    'object',
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
    'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos',
]


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


def make_robo_runner(task_key: str, out_dir: pathlib.Path, max_steps: int):
    dataset_path = ROOT / 'data' / 'robomimic' / 'datasets' / task_key / 'ph' / 'low_dim_abs.hdf5'
    obs_keys = TRANSPORT_OBS_KEYS if task_key == 'transport' else ROBO_OBS_KEYS
    rcfg = OmegaConf.create({
        '_target_': 'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner',
        'output_dir': str(out_dir),
        'dataset_path': str(dataset_path),
        'obs_keys': obs_keys,
        'n_train': 0,
        'n_train_vis': 0,
        'train_start_idx': 0,
        'n_test': 1,
        'n_test_vis': 0,
        'test_start_seed': 100000,
        'max_steps': max_steps,
        'n_obs_steps': 2,
        'n_action_steps': 8,
        'n_latency_steps': 0,
        'render_hw': [128, 128],
        'fps': 10,
        'crf': 22,
        'past_action': False,
        'abs_action': True,
        'n_envs': 1,
    })
    return hydra.utils.instantiate(rcfg)


def run_robo_episode(policy, runner, env, cfg, seed: int, device: str):
    n_obs_steps = int(getattr(cfg, 'n_obs_steps', 2))
    n_action_steps = int(getattr(cfg, 'n_action_steps', 8))

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    policy.reset()

    done = False
    rewards, episode_ks, episode_times = [], [], []
    while not done:
        obs_dict = {'obs': torch.from_numpy(obs).float().unsqueeze(0).to(device)}
        sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)
        sync_if_cuda(device)
        episode_times.append((time.perf_counter() - t0) * 1000)
        episode_ks.append(int(result['refinement_steps'].flatten()[0].item()))

        action = result['action'][0, :n_action_steps].detach().cpu().numpy()
        action_env = runner.undo_transform_action(action)
        obs, reward, done, info = env.step(action_env)
        if isinstance(done, np.ndarray):
            done = bool(np.all(done))
        if reward is not None:
            rewards.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))

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


def eval_task(task_key: str, args) -> Dict:
    spec = TASKS[task_key]
    ckpt_path = ROOT / spec['ckpt']
    if not ckpt_path.exists():
        raise FileNotFoundError(f'{task_key}: checkpoint not found: {ckpt_path}')

    print(f'\n=== {spec["name"]} | epoch={spec["epoch"]} | {ckpt_path} ===', flush=True)
    ws, cfg = load_workspace(ckpt_path, args.device)

    out_task = pathlib.Path(args.output_dir) / task_key
    out_task.mkdir(parents=True, exist_ok=True)

    runner = None
    env = None
    if spec['env_type'] == 'robo':
        runner = make_robo_runner(spec['dataset'], out_task / 'env', int(spec['max_steps']))
        env = runner.env_fns[0]()

    run_summaries = []
    try:
        for run_idx, seed_start in enumerate(args.seed_starts[:args.n_runs]):
            print(f'  Run {run_idx + 1}/{args.n_runs}: seeds {seed_start}-{seed_start + args.n_episodes - 1}', flush=True)
            episodes = []
            for ep_idx, seed in enumerate(range(seed_start, seed_start + args.n_episodes)):
                if spec['env_type'] == 'pusht':
                    ep_result = run_pusht_episode(ws.policy, cfg, seed, args.device)
                else:
                    ep_result = run_robo_episode(ws.policy, runner, env, cfg, seed, args.device)
                episodes.append(ep_result)
                if (ep_idx + 1) % args.print_every == 0 or ep_idx + 1 == args.n_episodes:
                    cur = summarize_run(episodes)
                    print(
                        f'    {ep_idx + 1:3d}/{args.n_episodes}: '
                        f'acc={cur["success"] * 100:.1f}% '
                        f'lat={cur["latency_ms"]:.1f}ms '
                        f'nfe={cur["avg_nfe"]:.2f}',
                        flush=True,
                    )
            summary = summarize_run(episodes)
            summary['seed_start'] = int(seed_start)
            summary['seed_end'] = int(seed_start + args.n_episodes - 1)
            run_summaries.append(summary)
            print(
                f'  {spec["name"]} run{run_idx + 1}: '
                f'acc={summary["success"] * 100:.1f}% '
                f'lat={summary["latency_ms"]:.1f}ms '
                f'nfe={summary["avg_nfe"]:.2f} '
                f'calls={summary["total_calls"]}',
                flush=True,
            )
    finally:
        if env is not None:
            env.close()

    acc_vals = [r['success'] for r in run_summaries]
    lat_vals = [r['latency_ms'] for r in run_summaries]
    nfe_vals = [r['avg_nfe'] for r in run_summaries]
    reward_vals = [r['mean_reward'] for r in run_summaries]

    result = {
        'task': spec['name'],
        'task_key': task_key,
        'epoch': spec['epoch'],
        'checkpoint': str(ckpt_path),
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

    json_path = out_task / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(
        f'  FINAL {spec["name"]}: acc={result["summary"]["acc_percent"]} '
        f'lat={result["summary"]["latency_ms"]}ms '
        f'nfe={result["summary"]["avg_nfe"]}',
        flush=True,
    )
    print(f'  saved {json_path}', flush=True)
    return result


def write_combined(results: Dict[str, Dict], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'combined_results.json'
    md_path = output_dir / 'combined_results.md'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open(md_path, 'w') as f:
        f.write('# Lowdim PEGrad mean/std evaluation\n\n')
        f.write('| Task | Epoch | Acc | Latency(ms) | Avg NFE |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for key, r in results.items():
            s = r['summary']
            f.write(f'| {r["task"]} | {r["epoch"]} | {s["acc_percent"]} | {s["latency_ms"]} | {s["avg_nfe"]} |\n')

    print('\n' + '=' * 80)
    print('Combined table')
    print('Task       Epoch    Acc          Latency(ms)    Avg NFE')
    for key, r in results.items():
        s = r['summary']
        print(f'{r["task"]:<10} {r["epoch"]:<8} {s["acc_percent"]:<12} {s["latency_ms"]:<14} {s["avg_nfe"]}')
    print(f'\nsaved {json_path}')
    print(f'saved {md_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--n-episodes', type=int, default=50)
    parser.add_argument('--seed-starts', nargs='+', type=int, default=[100000, 100050, 100100])
    parser.add_argument('--output-dir', default=str(ROOT / 'result' / 'lowdim_pegrad_meanstd'))
    parser.add_argument('--print-every', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.seed_starts) < args.n_runs:
        raise ValueError('--seed-starts must provide at least --n-runs values')
    results = {}
    for task_key in args.tasks:
        results[task_key] = eval_task(task_key, args)
    write_combined(results, pathlib.Path(args.output_dir))


if __name__ == '__main__':
    main()
