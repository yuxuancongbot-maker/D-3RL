#!/usr/bin/env python
"""
采集 Ada-BRIDGER 调度预算轨迹并自动出图。

用法示例：
python collect_budget_trace.py \
  --checkpoint data/outputs/ada_bridger_rl_can_v7/checkpoints/latest.ckpt \
  --output_dir data/outputs/budget_trace_can \
  --n_episodes 50 \
  --device cuda:0
"""

import os
import sys
import csv
import json
import argparse
import pathlib
from collections import Counter, defaultdict

import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)

from eval_ada_bridger import load_ada_bridger_policy
from diffusion_policy.common.pytorch_util import dict_apply


def _prepare_policy_normalizer(policy):
    src_normalizer = policy.source_policy.normalizer
    ref_normalizer = policy.refinement_policy.normalizer
    try:
        _ = src_normalizer['obs']
        normalizer = src_normalizer
    except (AttributeError, KeyError):
        normalizer = ref_normalizer
        policy.source_policy.normalizer = ref_normalizer
    policy.set_normalizer(normalizer)


def collect_budget_trace(
    checkpoint: str,
    output_dir: str,
    n_episodes: int,
    device: str,
    success_threshold: float,
    deterministic: bool,
    episode_index: int,
):
    os.makedirs(output_dir, exist_ok=True)

    policy, cfg = load_ada_bridger_policy(checkpoint_path=checkpoint, device=device)
    _prepare_policy_normalizer(policy)
    policy.scheduler_deterministic = deterministic
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
    )

    env = env_runner.env
    n_envs = len(env_runner.env_fns)

    trace_rows = []
    episode_meta = []

    for rollout_idx in range(n_episodes):
        policy.reset()

        init_fn_dill = env_runner.env_init_fn_dills[rollout_idx % len(env_runner.env_init_fn_dills)]
        env.call_each('run_dill_function', args_list=[(init_fn_dill,)] * n_envs)

        obs = env.reset()
        done_arr = np.zeros(n_envs, dtype=bool)
        episode_length = np.zeros(n_envs, dtype=np.int32)
        decision_count = np.zeros(n_envs, dtype=np.int32)
        episode_max_reward = [None for _ in range(n_envs)]

        while (not np.all(done_arr)) and (np.max(episode_length) < env_runner.max_steps):
            raw_obs = obs[:, :env_runner.n_obs_steps].astype(np.float32)
            expected_obs_dim = cfg.obs_dim
            if raw_obs.shape[-1] == expected_obs_dim * 2:
                raw_obs = raw_obs[..., :expected_obs_dim]

            np_obs_dict = {'obs': raw_obs}
            obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

            with torch.no_grad():
                result = policy.predict_action(obs_dict, return_intermediate=True)

            refinement_steps = result['refinement_steps']
            for env_idx in range(n_envs):
                if done_arr[env_idx]:
                    continue
                k = int(refinement_steps[env_idx].item()) if refinement_steps.dim() > 0 else int(refinement_steps.item())
                trace_rows.append({
                    'episode_id': rollout_idx,
                    'env_idx': env_idx,
                    't': int(decision_count[env_idx]),
                    'k_t': k,
                })
                decision_count[env_idx] += 1

            np_action = result['action'].detach().cpu().numpy()
            action_for_env = np_action[:, env_runner.n_latency_steps:]
            if getattr(env_runner, 'abs_action', False):
                action_for_env = env_runner.undo_transform_action(action_for_env)
            if not np.isfinite(action_for_env).all():
                action_for_env = np.nan_to_num(action_for_env, nan=0.0, posinf=1.0, neginf=-1.0)

            obs, reward, done_arr, _ = env.step(action_for_env)
            episode_length += (~done_arr).astype(np.int32) * action_for_env.shape[1]

            reward_arr = np.asarray(reward)
            if reward_arr.size > 0:
                for env_idx in range(n_envs):
                    reward_env = reward_arr[env_idx]
                    reward_val = float(np.max(reward_env)) if np.ndim(reward_env) > 0 else float(reward_env)
                    if episode_max_reward[env_idx] is None:
                        episode_max_reward[env_idx] = reward_val
                    else:
                        episode_max_reward[env_idx] = max(episode_max_reward[env_idx], reward_val)

        final_rewards = env.call('get_attr', 'reward')
        for env_idx in range(n_envs):
            if final_rewards and (final_rewards[env_idx] is not None) and (len(final_rewards[env_idx]) > 0):
                max_reward = float(np.max(final_rewards[env_idx]))
            elif episode_max_reward[env_idx] is not None:
                max_reward = float(episode_max_reward[env_idx])
            else:
                max_reward = 0.0
            success = float(max_reward > success_threshold)
            episode_meta.append({
                'episode_id': rollout_idx,
                'env_idx': env_idx,
                'success': success,
                'max_reward': max_reward,
            })

    meta_map = {(m['episode_id'], m['env_idx']): m for m in episode_meta}
    for row in trace_rows:
        meta = meta_map[(row['episode_id'], row['env_idx'])]
        row['success'] = int(meta['success'])
        row['max_reward'] = meta['max_reward']

    csv_path = os.path.join(output_dir, 'budget_trace.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['episode_id', 'env_idx', 't', 'k_t', 'success', 'max_reward']
        )
        writer.writeheader()
        writer.writerows(trace_rows)

    grouped = defaultdict(list)
    for row in trace_rows:
        key = (row['episode_id'], row['env_idx'])
        grouped[key].append((row['t'], row['k_t']))

    successful_keys = [
        (m['episode_id'], m['env_idx'])
        for m in episode_meta if m['success'] > 0.5
    ]

    if successful_keys:
        selected_key = successful_keys[min(max(episode_index, 0), len(successful_keys) - 1)]
    else:
        best = max(episode_meta, key=lambda x: x['max_reward'])
        selected_key = (best['episode_id'], best['env_idx'])

    seq = sorted(grouped[selected_key], key=lambda x: x[0])
    ts = [x[0] for x in seq]
    ks = [x[1] for x in seq]

    plt.figure(figsize=(8, 6))  # 4:3，适合论文单栏
    plt.step(ts, ks, where='post', linewidth=2.8)
    plt.scatter(ts, ks, c=ks, s=50)
    plt.xlabel('time step', fontsize=20)
    plt.ylabel('step decision', fontsize=20)
    plt.title('budget over time', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'budget_over_time.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()

    all_k = [r['k_t'] for r in trace_rows]
    counter = Counter(all_k)
    total = sum(counter.values()) if counter else 1
    dist = {int(k): float(v) / float(total) for k, v in sorted(counter.items(), key=lambda x: x[0])}

    summary = {
        'checkpoint': checkpoint,
        'n_episodes': n_episodes,
        'deterministic': deterministic,
        'success_threshold': success_threshold,
        'selected_episode': {
            'episode_id': selected_key[0],
            'env_idx': selected_key[1],
        },
        'num_successful_episode_envs': len(successful_keys),
        'step_distribution': dist,
        'output_files': {
            'csv': csv_path,
            'figure': fig_path,
        },
    }

    summary_path = os.path.join(output_dir, 'budget_trace_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Budget Trace Done ===')
    print(f'CSV: {csv_path}')
    print(f'Figure: {fig_path}')
    print(f'Summary: {summary_path}')
    print(f'Selected episode/env: {selected_key}')
    print(f'Successful episode-env count: {len(successful_keys)}')


def plot_budget_from_csv(
    csv_path: str,
    output_dir: str,
    episode_index: int,
):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'episode_id': int(row['episode_id']),
                'env_idx': int(row['env_idx']),
                't': int(row['t']),
                'k_t': int(row['k_t']),
                'success': int(row.get('success', 0)),
                'max_reward': float(row.get('max_reward', 0.0)),
            })

    if len(rows) == 0:
        raise ValueError(f'CSV 没有可用数据: {csv_path}')

    grouped = defaultdict(list)
    meta = {}
    for row in rows:
        key = (row['episode_id'], row['env_idx'])
        grouped[key].append((row['t'], row['k_t']))
        if key not in meta:
            meta[key] = {'success': row['success'], 'max_reward': row['max_reward']}

    successful_keys = [k for k, v in meta.items() if v['success'] > 0]
    if successful_keys:
        selected_key = successful_keys[min(max(episode_index, 0), len(successful_keys) - 1)]
    else:
        selected_key = max(meta.items(), key=lambda x: x[1]['max_reward'])[0]

    seq = sorted(grouped[selected_key], key=lambda x: x[0])
    ts = [x[0] for x in seq]
    ks = [x[1] for x in seq]

    plt.figure(figsize=(8, 6))
    plt.step(ts, ks, where='post', linewidth=2.8)
    plt.scatter(ts, ks, c=ks, s=50)
    plt.xlabel('time step', fontsize=20)
    plt.ylabel('step decision', fontsize=20)
    plt.title('budget over time', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'budget_over_time.png')
    plt.savefig(fig_path, dpi=200)
    plt.close()

    all_k = [r['k_t'] for r in rows]
    counter = Counter(all_k)
    total = sum(counter.values()) if counter else 1
    dist = {int(k): float(v) / float(total) for k, v in sorted(counter.items(), key=lambda x: x[0])}

    summary = {
        'mode': 'csv_only',
        'csv_path': csv_path,
        'selected_episode': {
            'episode_id': selected_key[0],
            'env_idx': selected_key[1],
        },
        'num_successful_episode_envs': len(successful_keys),
        'step_distribution': dist,
        'output_files': {
            'figure': fig_path,
        },
    }
    summary_path = os.path.join(output_dir, 'budget_trace_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Budget Plot From CSV Done ===')
    print(f'CSV: {csv_path}')
    print(f'Figure: {fig_path}')
    print(f'Summary: {summary_path}')
    print(f'Selected episode/env: {selected_key}')
    print(f'Successful episode-env count: {len(successful_keys)}')


def parse_args():
    parser = argparse.ArgumentParser(description='Collect budget trace and plot k_t over time.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default=None,
                        help='若提供，则直接从已有 budget_trace.csv 生成图片，不重新跑环境')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--success_threshold', type=float, default=0.5)
    parser.add_argument('--episode_index', type=int, default=0,
                        help='选择第几个成功 episode-env 用于绘图（0-based）')
    parser.add_argument('--stochastic', action='store_true',
                        help='启用随机调度（默认确定性）')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.csv_path is not None:
        plot_budget_from_csv(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            episode_index=args.episode_index,
        )
    else:
        collect_budget_trace(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            n_episodes=args.n_episodes,
            device=args.device,
            success_threshold=args.success_threshold,
            deterministic=(not args.stochastic),
            episode_index=args.episode_index,
        )
