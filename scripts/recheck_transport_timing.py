#!/usr/bin/env python
"""Recheck Transport lowdim timing under one evaluation path.

Compares the historical lightweight PPO scheduler artifact and the PPO+PEGrad
workspace checkpoint with the same environment construction, hardware, warmup,
evaluation seeds, external CUDA-event timing, and per-episode internal timing
aggregation.
"""

import argparse
import copy
import json
import pathlib
import sys
import time
from collections import Counter
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('eval', eval, replace=True)

TRANSPORT_OBS_KEYS = [
    'object',
    'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
    'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos',
]


def make_transport_runner(out_dir: pathlib.Path):
    dataset_path = ROOT / 'data' / 'robomimic' / 'datasets' / 'transport' / 'ph' / 'low_dim_abs.hdf5'
    rcfg = OmegaConf.create({
        '_target_': 'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner',
        'output_dir': str(out_dir),
        'dataset_path': str(dataset_path),
        'obs_keys': TRANSPORT_OBS_KEYS,
        'n_train': 0,
        'n_train_vis': 0,
        'train_start_idx': 0,
        'n_test': 1,
        'n_test_vis': 0,
        'test_start_seed': 100000,
        'max_steps': 700,
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


def load_workspace_policy(ckpt_path: pathlib.Path, device: str):
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
    return ws.policy, payload['cfg']


def load_scheduler_policy(source_ckpt: str, refiner_ckpt: str, scheduler_ckpt: str, device: str):
    from diffusion_policy.cli.infer_d3rl import load_policy, load_scheduler
    from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy

    source_policy, src_cfg = load_policy(source_ckpt, device)
    refiner_policy, ref_cfg = load_policy(refiner_ckpt, device)
    scheduler = load_scheduler(scheduler_ckpt, device)
    policy_model = getattr(source_policy, 'model', source_policy)
    policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refiner_policy,
        scheduler=scheduler,
        horizon=getattr(source_policy, 'horizon', src_cfg.policy.horizon),
        obs_dim=src_cfg.policy.get('obs_dim', 59),
        action_dim=src_cfg.policy.get('action_dim', getattr(policy_model, 'action_dim', 20)),
        n_action_steps=getattr(source_policy, 'n_action_steps', src_cfg.policy.n_action_steps),
        n_obs_steps=getattr(source_policy, 'n_obs_steps', src_cfg.policy.n_obs_steps),
        refinement_steps=[0, 1, 2, 5],
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    try:
        _ = source_policy.normalizer['obs']
        normalizer = source_policy.normalizer
    except Exception:
        normalizer = refiner_policy.normalizer
        source_policy.normalizer = copy.deepcopy(normalizer)
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy, src_cfg


def sync_if_cuda(device: str):
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_episode(policy, runner, env, cfg, seed: int, device: str) -> Dict:
    n_obs_steps = int(getattr(cfg, 'n_obs_steps', 2))
    n_action_steps = int(getattr(cfg, 'n_action_steps', 8))

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    policy.reset()

    done = False
    rewards, ks, external_times = [], [], []
    while not done:
        obs_dict = {'obs': torch.from_numpy(obs).float().unsqueeze(0).to(device)}
        sync_if_cuda(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)
        sync_if_cuda(device)
        external_times.append((time.perf_counter() - t0) * 1000.0)
        ks.append(int(result['refinement_steps'].flatten()[0].item()))

        action = result['action'][0, :n_action_steps].detach().cpu().numpy()
        action_env = runner.undo_transform_action(action)
        obs, reward, done, info = env.step(action_env)
        if isinstance(done, np.ndarray):
            done = bool(np.all(done))
        if reward is not None:
            rewards.append(np.max(reward) if np.ndim(reward) > 0 else float(reward))

    internal = policy.get_inference_stats()
    max_reward = float(np.max(rewards)) if rewards else 0.0
    return {
        'seed': int(seed),
        'success': float(max_reward > 0.5),
        'reward': max_reward,
        'avg_nfe': float(np.mean(ks)) if ks else 0.0,
        'latency_external_ms': float(np.mean(external_times[1:] if len(external_times) > 1 else external_times)) if external_times else 0.0,
        'latency_external_all_ms': float(np.mean(external_times)) if external_times else 0.0,
        'n_calls': int(len(ks)),
        'k_counts': {str(k): int(ks.count(k)) for k in sorted(set(ks))},
        'internal': {
            'total_calls': int(internal.get('total_calls', 0)),
            'avg_total_ms': float(internal.get('avg_total_time', 0.0) * 1000.0),
            'avg_source_ms': float(internal.get('avg_source_time', 0.0) * 1000.0),
            'avg_scheduler_ms': float(internal.get('avg_scheduler_time', 0.0) * 1000.0),
            'avg_refine_ms': float(internal.get('avg_refine_time', 0.0) * 1000.0),
        },
    }


def merge_counts(episodes: List[Dict]) -> Dict[str, int]:
    out = Counter()
    for ep in episodes:
        out.update({k: int(v) for k, v in ep['k_counts'].items()})
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def summarize(episodes: List[Dict]) -> Dict:
    internal_weights = np.asarray([max(ep['internal']['total_calls'], 1) for ep in episodes], dtype=np.float64)
    def wavg(key):
        vals = np.asarray([ep['internal'][key] for ep in episodes], dtype=np.float64)
        return float(np.average(vals, weights=internal_weights))
    return {
        'success': float(np.mean([ep['success'] for ep in episodes])),
        'mean_reward': float(np.mean([ep['reward'] for ep in episodes])),
        'avg_nfe': float(np.mean([ep['avg_nfe'] for ep in episodes])),
        'latency_external_ms': float(np.mean([ep['latency_external_ms'] for ep in episodes])),
        'latency_external_all_ms': float(np.mean([ep['latency_external_all_ms'] for ep in episodes])),
        'internal_total_ms': wavg('avg_total_ms'),
        'internal_source_ms': wavg('avg_source_ms'),
        'internal_scheduler_ms': wavg('avg_scheduler_ms'),
        'internal_refine_ms': wavg('avg_refine_ms'),
        'total_calls': int(sum(ep['n_calls'] for ep in episodes)),
        'k_counts': merge_counts(episodes),
    }


def eval_method(name: str, policy, cfg, args) -> Dict:
    runner = make_transport_runner(pathlib.Path(args.output_dir) / name / 'env')
    env = runner.env_fns[0]()
    try:
        print(f'\n=== {name} warmup ===', flush=True)
        for i in range(args.warmup_episodes):
            _ = run_episode(policy, runner, env, cfg, args.warmup_seed_start + i, args.device)
        print(f'=== {name} eval: seeds {args.seed_start}-{args.seed_start + args.n_episodes - 1} ===', flush=True)
        episodes = []
        for i, seed in enumerate(range(args.seed_start, args.seed_start + args.n_episodes)):
            ep = run_episode(policy, runner, env, cfg, seed, args.device)
            episodes.append(ep)
            if (i + 1) % args.print_every == 0 or i + 1 == args.n_episodes:
                s = summarize(episodes)
                print(
                    f'  {i + 1:3d}/{args.n_episodes}: '
                    f'acc={s["success"]*100:.1f}% nfe={s["avg_nfe"]:.2f} '
                    f'ext={s["latency_external_ms"]:.1f}ms int={s["internal_total_ms"]:.1f}ms',
                    flush=True,
                )
    finally:
        env.close()
    return {'name': name, 'summary': summarize(episodes), 'episodes': episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=100000)
    parser.add_argument('--warmup-episodes', type=int, default=3)
    parser.add_argument('--warmup-seed-start', type=int, default=99000)
    parser.add_argument('--print-every', type=int, default=5)
    parser.add_argument('--output-dir', default=str(ROOT / 'result' / 'transport_timing_recheck'))
    parser.add_argument('--source', default='weights/lowdim/predictor/vae/vae_transport_lowdim/checkpoints/latest.ckpt')
    parser.add_argument('--refiner', default='weights/lowdim/diffusion/robomimic/cnn/transport/latest.ckpt')
    parser.add_argument('--ppo-scheduler', default='outputs/train_transport/scheduler_epoch149_ppo_best.pt')
    parser.add_argument('--pegrad-ckpt', default='outputs/train_transport/checkpoints/epoch=0199-test_mean_score=0.720.ckpt')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print('Transport timing recheck', flush=True)
    print(f'  device={args.device}', flush=True)
    print(f'  eval seeds={args.seed_start}-{args.seed_start + args.n_episodes - 1}', flush=True)
    print(f'  warmup episodes={args.warmup_episodes}', flush=True)

    print('\nLoading PPO scheduler-only policy...', flush=True)
    ppo_policy, ppo_cfg = load_scheduler_policy(args.source, args.refiner, args.ppo_scheduler, args.device)
    print('\nLoading PPO+PEGrad workspace policy...', flush=True)
    pegrad_policy, pegrad_cfg = load_workspace_policy((ROOT / args.pegrad_ckpt).resolve(), args.device)

    results = {
        'protocol': {
            'device': args.device,
            'n_episodes': int(args.n_episodes),
            'seed_start': int(args.seed_start),
            'warmup_episodes': int(args.warmup_episodes),
            'warmup_seed_start': int(args.warmup_seed_start),
            'timing': 'external wall-clock with cuda synchronize, first decision per episode excluded; internal timing aggregated per episode from policy stats',
        },
        'methods': {},
    }
    for name, policy, cfg in [
        ('ppo_scheduler149', ppo_policy, ppo_cfg),
        ('pegrad_ckpt199', pegrad_policy, pegrad_cfg),
    ]:
        r = eval_method(name, policy, cfg, args)
        results['methods'][name] = r

    json_path = out_dir / 'results.json'
    md_path = out_dir / 'results.md'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    with open(md_path, 'w') as f:
        f.write('# Transport timing recheck\n\n')
        f.write(f'- Device: `{args.device}`\n')
        f.write(f'- Eval seeds: {args.seed_start}-{args.seed_start + args.n_episodes - 1}\n')
        f.write(f'- Warmup episodes: {args.warmup_episodes}\n')
        f.write('- Timing: same script/env/hardware; external wall-clock excludes first decision per episode; internal timing aggregated per episode.\n\n')
        f.write('| Method | Success | Avg NFE | External ms | Internal total ms | source | scheduler | refine | k distribution |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|---|\n')
        for name, r in results['methods'].items():
            s = r['summary']
            kdist = ', '.join(f'k={k}:{v/s["total_calls"]*100:.1f}%' for k, v in s['k_counts'].items())
            f.write(
                f'| {name} | {s["success"]*100:.1f}% | {s["avg_nfe"]:.2f} | '
                f'{s["latency_external_ms"]:.1f} | {s["internal_total_ms"]:.1f} | '
                f'{s["internal_source_ms"]:.1f} | {s["internal_scheduler_ms"]:.1f} | {s["internal_refine_ms"]:.1f} | {kdist} |\n'
            )
    print('\nFINAL', flush=True)
    for name, r in results['methods'].items():
        s = r['summary']
        print(
            f'  {name}: acc={s["success"]*100:.1f}% nfe={s["avg_nfe"]:.2f} '
            f'ext={s["latency_external_ms"]:.1f}ms int={s["internal_total_ms"]:.1f}ms '
            f'(source={s["internal_source_ms"]:.1f}, sched={s["internal_scheduler_ms"]:.1f}, refine={s["internal_refine_ms"]:.1f})',
            flush=True,
        )
    print(f'  saved {json_path}', flush=True)
    print(f'  saved {md_path}', flush=True)


if __name__ == '__main__':
    main()
