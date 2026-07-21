"""Collect image-feature oracle labels for Ada-BRIDGER scheduler pretraining.

Default mode mirrors the original lowdim scheduler pretrain pipeline: run a
strong fixed-k image D3RL trajectory, compare the source init action with the
strong refined action at every state, and label each state by action
 discrepancy. Low discrepancy maps to k=0; high discrepancy maps to max k.

The previous episode-level smallest-success collector is still available via
``--label-mode fixed_success`` for compatibility.
"""

import argparse
import copy
import os
import signal
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.cli.infer_d3rl import load_policy, create_image_env, _undo_transform_action


class EpisodeTimeout(Exception):
    pass


class FixedScheduler(torch.nn.Module):
    def __init__(self, k: int, refinement_steps: List[int]):
        super().__init__()
        self.k = int(k)
        self.refinement_steps = list(refinement_steps)
        if self.k not in self.refinement_steps:
            self.refinement_steps.append(self.k)
        self.k_idx = self.refinement_steps.index(self.k)

    def select_action(self, obs, init_action, deterministic=True):
        if isinstance(obs, torch.Tensor):
            batch_size = obs.shape[0]
            device = obs.device
        else:
            first = next(iter(obs.values()))
            batch_size = first.shape[0]
            device = first.device
        steps = torch.full((batch_size,), self.k, dtype=torch.long, device=device)
        idx = torch.full((batch_size,), self.k_idx, dtype=torch.long, device=device)
        return steps, idx, torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)


def build_policy(source_policy, refiner_policy, scheduler, cfg, device):
    policy_model = getattr(source_policy, 'model', source_policy)
    policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refiner_policy,
        scheduler=scheduler,
        horizon=getattr(source_policy, 'horizon', cfg.policy.horizon),
        obs_dim=cfg.policy.get('obs_dim', 0),
        action_dim=cfg.policy.get('action_dim', getattr(policy_model, 'action_dim')),
        n_action_steps=getattr(source_policy, 'n_action_steps', cfg.policy.n_action_steps),
        n_obs_steps=getattr(source_policy, 'n_obs_steps', cfg.policy.n_obs_steps),
        refinement_steps=scheduler.refinement_steps,
        max_refinement_steps=max(scheduler.refinement_steps),
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    try:
        _ = source_policy.normalizer['obs']
        normalizer = source_policy.normalizer
    except (AttributeError, KeyError):
        normalizer = refiner_policy.normalizer
        source_policy.normalizer = copy.deepcopy(normalizer)
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy


def _obs_to_tensor_dict(obs, n_obs_steps, device):
    return dict_apply(
        {k: v[np.newaxis, :n_obs_steps].astype(np.float32) for k, v in obs.items()},
        lambda x: torch.from_numpy(x).to(device),
    )


def _action_for_env(result, env, n_action_steps):
    np_action = result['action'].detach().cpu().numpy()
    action_for_env = np_action[0, :n_action_steps]
    rot_tf = getattr(env, '_rotation_transformer', None)
    if rot_tf is not None and getattr(env, '_abs_action', False):
        action_for_env = _undo_transform_action(action_for_env, rot_tf)
    return np.nan_to_num(action_for_env, nan=0.0, posinf=1.0, neginf=-1.0)


def run_episode(policy, env, device, n_obs_steps, n_action_steps, collect_samples=False,
                success_threshold=0.5):
    """Compatibility fixed-success episode runner."""
    policy.reset()
    obs = env.reset()
    done = False
    rewards = []
    samples = []
    k_history = []

    while not done:
        obs_dict = _obs_to_tensor_dict(obs, n_obs_steps, device)
        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)

        k = int(result['refinement_steps'].flatten()[0].item())
        k_history.append(k)

        if collect_samples:
            if 'obs_feat' not in result:
                raise KeyError(
                    "Policy did not return obs_feat. Image scheduler labels require "
                    "ActionPredictorImageVAEPolicy obs features."
                )
            samples.append({
                'obs': result['obs_feat'][0, :n_obs_steps].detach().cpu(),
                'init_action': result['init_action'][0].detach().cpu(),
            })

        action_for_env = _action_for_env(result, env, n_action_steps)
        try:
            obs, reward, done, info = env.step(action_for_env)
        except EpisodeTimeout:
            raise
        except Exception as e:
            print(f"  [WARN] env.step failed for k={k}: {e}")
            done = True
            reward = 0.0
        if reward is not None:
            reward_value = np.max(reward) if np.ndim(reward) > 0 else reward
            rewards.append(reward_value)
            if float(reward_value) > success_threshold:
                done = True

    final_reward = float(np.max(rewards)) if rewards else 0.0
    return {
        'reward': final_reward,
        'success': float(final_reward > success_threshold),
        'samples': samples,
        'k_distribution': dict(Counter(k_history)),
    }


def run_action_diff_episode(policy, env, device, n_obs_steps, n_action_steps,
                            strong_k: int, success_threshold: float = 0.5,
                            save_refined_action: bool = False,
                            max_decisions: int = None):
    """Run one strong-k episode and collect per-state action discrepancy samples."""
    policy.reset()
    obs = env.reset()
    done = False
    rewards = []
    samples = []
    k_history = []
    step_idx = 0

    while not done:
        obs_dict = _obs_to_tensor_dict(obs, n_obs_steps, device)
        with torch.no_grad():
            result = policy.predict_action(obs_dict, return_intermediate=True)

        for key in ('obs_feat', 'init_action', 'refined_action'):
            if key not in result:
                raise KeyError(
                    f"Policy did not return {key}. Image action-diff labels require "
                    "return_intermediate=True with image source features."
                )

        init_action = result['init_action'][0].detach()
        refined_action = result['refined_action'][0].detach()
        action_diff = (init_action - refined_action).abs().mean().item()
        k = int(result['refinement_steps'].flatten()[0].item())
        k_history.append(k)

        sample = {
            'obs': result['obs_feat'][0, :n_obs_steps].detach().cpu(),
            'init_action': init_action.cpu(),
            'action_diff': float(action_diff),
            'step_in_episode': int(step_idx),
            'strong_refinement_steps': int(strong_k),
        }
        if save_refined_action:
            sample['refined_action'] = refined_action.cpu()
        samples.append(sample)

        action_for_env = _action_for_env(result, env, n_action_steps)
        try:
            obs, reward, done, info = env.step(action_for_env)
        except EpisodeTimeout:
            raise
        except Exception as e:
            print(f"  [WARN] env.step failed for strong k={strong_k}: {e}")
            done = True
            reward = 0.0
        if reward is not None:
            reward_value = np.max(reward) if np.ndim(reward) > 0 else reward
            rewards.append(reward_value)
            if float(reward_value) > success_threshold:
                done = True
        step_idx += 1
        if max_decisions is not None and step_idx >= max_decisions:
            done = True

    final_reward = float(np.max(rewards)) if rewards else 0.0
    success = float(final_reward > success_threshold)
    for sample in samples:
        sample['episode_reward'] = final_reward
        sample['episode_success'] = success
    return {
        'reward': final_reward,
        'success': success,
        'samples': samples,
        'k_distribution': dict(Counter(k_history)),
        'episode_length': len(samples),
    }


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) > 0 else 0.0


def assign_action_diff_labels(samples, refinement_steps, threshold, threshold_mode,
                              threshold_eta, zero_step, max_label_step):
    if zero_step not in refinement_steps:
        raise ValueError(f"zero_refinement_step={zero_step} not in refinement_steps={refinement_steps}")
    if max_label_step not in refinement_steps:
        raise ValueError(
            f"max_label_refinement_step={max_label_step} not in refinement_steps={refinement_steps}")

    diffs = np.array([s['action_diff'] for s in samples], dtype=np.float64)
    if diffs.size == 0:
        raise RuntimeError('No action-diff samples collected.')
    median_diff = float(np.median(diffs))
    if threshold_mode == 'fixed':
        effective_threshold = float(threshold)
    elif threshold_mode == 'dynamic':
        effective_threshold = float(max(threshold, median_diff * threshold_eta))
    else:
        raise ValueError(f'Unknown threshold_mode: {threshold_mode}')

    zero_idx = refinement_steps.index(zero_step)
    max_idx = refinement_steps.index(max_label_step)
    labels = []
    label_ks = []
    for sample in samples:
        if sample['action_diff'] < effective_threshold:
            label = zero_idx
            label_k = zero_step
        else:
            label = max_idx
            label_k = max_label_step
        sample['label'] = int(label)
        sample['label_k'] = int(label_k)
        labels.append(int(label))
        label_ks.append(int(label_k))

    stats = {
        'action_diff_mean': float(np.mean(diffs)),
        'action_diff_std': float(np.std(diffs)),
        'action_diff_min': float(np.min(diffs)),
        'action_diff_max': float(np.max(diffs)),
        'action_diff_median': median_diff,
        'action_diff_p10': _percentile(diffs, 10),
        'action_diff_p25': _percentile(diffs, 25),
        'action_diff_p75': _percentile(diffs, 75),
        'action_diff_p90': _percentile(diffs, 90),
        'threshold': float(threshold),
        'threshold_mode': threshold_mode,
        'threshold_eta': float(threshold_eta),
        'effective_threshold': float(effective_threshold),
        'label_counts': dict(Counter(labels)),
        'label_counts_by_k': dict(Counter(label_ks)),
    }
    return samples, stats


def parse_args():
    parser = argparse.ArgumentParser(description='Collect image scheduler oracle labels')
    parser.add_argument('--task', required=True,
                        choices=['pusht_image', 'can_image', 'lift_image', 'square_image',
                                 'transport_image', 'tool_hang_image'])
    parser.add_argument('--source', required=True, help='Image VAE source checkpoint')
    parser.add_argument('--refiner', required=True, help='Image diffusion refiner checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--image-dataset', default=None,
                        help='Dataset path override for Robomimic image tasks')
    parser.add_argument('--refinement-steps', nargs='+', type=int, default=[0, 1, 2, 5])
    parser.add_argument('--label-mode', choices=['action_diff', 'fixed_success'], default='action_diff')
    parser.add_argument('--strong-refinement-steps', type=int, default=None,
                        help='Strong fixed-k rollout for action_diff labels; defaults to max refinement step.')
    parser.add_argument('--zero-refinement-step', type=int, default=0)
    parser.add_argument('--max-label-refinement-step', type=int, default=None,
                        help='High-discrepancy label k; defaults to max refinement step.')
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--threshold-mode', choices=['dynamic', 'fixed'], default='dynamic')
    parser.add_argument('--threshold-eta', type=float, default=0.5)
    parser.add_argument('--success-threshold', type=float, default=0.5)
    parser.add_argument('--save-refined-action', action='store_true')
    parser.add_argument('--max-decisions-per-episode', type=int, default=None,
                        help='Optional action-diff smoke/debug cap on scheduler decisions per episode.')
    parser.add_argument('--n-episodes', type=int, default=20)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--candidate-timeout-sec', type=int, default=180,
                        help='Per fixed-k/action-diff episode timeout; 0 disables the alarm.')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def _create_output(obs_samples, init_action_samples, labels, meta, args, refinement_steps,
                   stats, extra_tensors=None):
    obs_tensor = torch.stack(obs_samples)
    init_action_tensor = torch.stack(init_action_samples)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    output = {
        'obs': obs_tensor,
        'init_action': init_action_tensor,
        'label': label_tensor,
        'meta': meta,
        'stats': stats,
        'config': {
            'task': args.task,
            'source': args.source,
            'refiner': args.refiner,
            'image_dataset': args.image_dataset,
            'n_episodes': args.n_episodes,
            'seed_start': args.seed_start,
            'refinement_steps': refinement_steps,
            'obs_mode': 'image_feature',
            'obs_dim': int(obs_tensor.shape[-1]),
            'action_dim': int(init_action_tensor.shape[-1]),
            'horizon': int(init_action_tensor.shape[1]),
            'n_obs_steps': int(obs_tensor.shape[1]),
            'n_action_steps': int(getattr(args, '_n_action_steps', 0)),
        },
    }
    if extra_tensors:
        output.update(extra_tensors)
    return output


def main():
    args = parse_args()
    device = args.device
    refinement_steps = list(args.refinement_steps)
    if len(refinement_steps) == 0:
        raise ValueError('At least one refinement step is required.')
    strong_k = max(refinement_steps) if args.strong_refinement_steps is None else int(args.strong_refinement_steps)
    max_label_step = max(refinement_steps) if args.max_label_refinement_step is None else int(args.max_label_refinement_step)

    print('=' * 60)
    print('Image Scheduler Label Collection')
    print(f'  task = {args.task}')
    print(f'  label_mode = {args.label_mode}')
    print(f'  refinement_steps = {refinement_steps}')
    if args.label_mode == 'action_diff':
        print(f'  strong_refinement_steps = {strong_k}')
    print(f'  n_episodes = {args.n_episodes}')
    print('=' * 60)

    print('\n[1] Loading policies...')
    source_policy, src_cfg = load_policy(args.source, device)
    refiner_policy, ref_cfg = load_policy(args.refiner, device)

    shape_meta = ref_cfg.get('shape_meta', None)
    if shape_meta is None and 'task' in ref_cfg:
        shape_meta = ref_cfg.task.get('shape_meta', None)
    if shape_meta is None:
        raise ValueError('Image label collection requires shape_meta in refiner checkpoint config.')

    n_obs_steps = getattr(source_policy, 'n_obs_steps', src_cfg.policy.n_obs_steps)
    n_action_steps = getattr(source_policy, 'n_action_steps', src_cfg.policy.n_action_steps)
    setattr(args, '_n_action_steps', n_action_steps)

    def _alarm_handler(signum, frame):
        raise EpisodeTimeout(f'candidate timed out after {args.candidate_timeout_sec}s')

    print('\n[2] Collecting labels...')
    obs_samples = []
    init_action_samples = []
    labels = []
    meta = []
    extra_tensors = {}

    if args.label_mode == 'fixed_success':
        policies = {}
        for k in refinement_steps:
            scheduler = FixedScheduler(k, refinement_steps).to(device)
            policies[k] = build_policy(source_policy, refiner_policy, scheduler, src_cfg, device)

        label_counts = Counter()
        success_counts = Counter()
        for ep in tqdm(range(args.n_episodes), desc='Episodes'):
            seed = args.seed_start + ep
            by_k: Dict[int, dict] = {}
            selected_k = None
            for k in refinement_steps:
                env = None
                try:
                    env, _ = create_image_env(args.task, shape_meta, dataset_path=args.image_dataset)
                    env.seed(seed)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if args.candidate_timeout_sec > 0:
                        signal.signal(signal.SIGALRM, _alarm_handler)
                        signal.alarm(args.candidate_timeout_sec)
                    by_k[k] = run_episode(
                        policies[k], env, device, n_obs_steps, n_action_steps,
                        collect_samples=True,
                        success_threshold=args.success_threshold,
                    )
                except EpisodeTimeout as e:
                    print(f"  [WARN] seed={seed} k={k} timed out: {e}")
                    by_k[k] = {'reward': 0.0, 'success': 0.0, 'samples': [], 'k_distribution': {}}
                except Exception as e:
                    print(f"  [WARN] seed={seed} k={k} failed: {e}")
                    by_k[k] = {'reward': 0.0, 'success': 0.0, 'samples': [], 'k_distribution': {}}
                finally:
                    if args.candidate_timeout_sec > 0:
                        signal.alarm(0)
                    if env is not None:
                        env.close()
                success_counts[k] += int(by_k[k]['success'] > 0.5)
                if by_k[k]['success'] > 0.5:
                    selected_k = k
                    break

            if selected_k is None:
                selected_k = refinement_steps[-1]

            label_idx = refinement_steps.index(selected_k)
            sample_k = selected_k
            if not by_k[sample_k]['samples']:
                sample_k = next((k for k in reversed(refinement_steps) if by_k[k]['samples']), selected_k)
            label_counts[selected_k] += len(by_k[sample_k]['samples'])
            reward_by_k = {int(k): float(v['reward']) for k, v in by_k.items()}
            for sample in by_k[sample_k]['samples']:
                obs_samples.append(sample['obs'])
                init_action_samples.append(sample['init_action'])
                labels.append(label_idx)
                meta.append({
                    'task': args.task,
                    'seed': seed,
                    'k': int(selected_k),
                    'sample_k': int(sample_k),
                    'reward_by_k': reward_by_k,
                    'label_mode': 'fixed_success',
                })

            print(f"  ep={ep} seed={seed} selected_k={selected_k} rewards={reward_by_k}")

        stats = {
            'success_counts': dict(success_counts),
            'label_counts': dict(label_counts),
            'label_counts_by_k': dict(label_counts),
            'n_samples': int(len(labels)),
            'label_mode': 'fixed_success',
        }
    else:
        policy_refinement_steps = list(refinement_steps)
        if strong_k not in policy_refinement_steps:
            policy_refinement_steps.append(strong_k)
        scheduler = FixedScheduler(strong_k, policy_refinement_steps).to(device)
        policy = build_policy(source_policy, refiner_policy, scheduler, src_cfg, device)

        raw_samples = []
        episode_lengths = []
        success_episodes = 0
        k_counts = Counter()
        for ep in tqdm(range(args.n_episodes), desc='Episodes'):
            seed = args.seed_start + ep
            env = None
            try:
                env, _ = create_image_env(args.task, shape_meta, dataset_path=args.image_dataset)
                env.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                if args.candidate_timeout_sec > 0:
                    signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.alarm(args.candidate_timeout_sec)
                result = run_action_diff_episode(
                    policy, env, device, n_obs_steps, n_action_steps,
                    strong_k=strong_k,
                    success_threshold=args.success_threshold,
                    save_refined_action=args.save_refined_action,
                    max_decisions=args.max_decisions_per_episode,
                )
            except EpisodeTimeout as e:
                print(f"  [WARN] seed={seed} strong_k={strong_k} timed out: {e}")
                result = {'reward': 0.0, 'success': 0.0, 'samples': [], 'k_distribution': {}, 'episode_length': 0}
            except Exception as e:
                print(f"  [WARN] seed={seed} strong_k={strong_k} failed: {e}")
                result = {'reward': 0.0, 'success': 0.0, 'samples': [], 'k_distribution': {}, 'episode_length': 0}
            finally:
                if args.candidate_timeout_sec > 0:
                    signal.alarm(0)
                if env is not None:
                    env.close()

            episode_lengths.append(int(result['episode_length']))
            success_episodes += int(result['success'] > 0.5)
            k_counts.update(result.get('k_distribution', {}))
            for sample in result['samples']:
                sample['task'] = args.task
                sample['seed'] = seed
                sample['episode_index'] = ep
                raw_samples.append(sample)
            print(
                f"  ep={ep} seed={seed} strong_k={strong_k} "
                f"reward={float(result['reward']):.3f} samples={len(result['samples'])}"
            )

        raw_samples, diff_stats = assign_action_diff_labels(
            raw_samples,
            refinement_steps=refinement_steps,
            threshold=args.threshold,
            threshold_mode=args.threshold_mode,
            threshold_eta=args.threshold_eta,
            zero_step=args.zero_refinement_step,
            max_label_step=max_label_step,
        )

        action_diffs = []
        label_ks = []
        refined_samples = []
        for sample in raw_samples:
            obs_samples.append(sample['obs'])
            init_action_samples.append(sample['init_action'])
            labels.append(sample['label'])
            action_diffs.append(sample['action_diff'])
            label_ks.append(sample['label_k'])
            if args.save_refined_action:
                refined_samples.append(sample['refined_action'])
            meta.append({
                'task': sample['task'],
                'seed': int(sample['seed']),
                'episode_index': int(sample['episode_index']),
                'step_in_episode': int(sample['step_in_episode']),
                'strong_refinement_steps': int(strong_k),
                'reward': float(sample['episode_reward']),
                'episode_success': float(sample['episode_success']),
                'action_diff': float(sample['action_diff']),
                'label': int(sample['label']),
                'label_k': int(sample['label_k']),
                'label_mode': 'action_diff',
            })

        extra_tensors['action_diff'] = torch.tensor(action_diffs, dtype=torch.float32)
        extra_tensors['label_k'] = torch.tensor(label_ks, dtype=torch.long)
        if args.save_refined_action and refined_samples:
            extra_tensors['refined_action'] = torch.stack(refined_samples)

        stats = {
            **diff_stats,
            'n_samples': int(len(labels)),
            'total_episodes': int(args.n_episodes),
            'success_episodes': int(success_episodes),
            'success_rate': float(success_episodes / max(args.n_episodes, 1)),
            'episode_lengths': episode_lengths,
            'strong_k_distribution': {int(k): int(v) for k, v in k_counts.items()},
            'strong_refinement_steps': int(strong_k),
            'zero_refinement_step': int(args.zero_refinement_step),
            'max_label_refinement_step': int(max_label_step),
            'label_mode': 'action_diff',
        }

    if not obs_samples:
        raise RuntimeError('No samples collected.')

    output = _create_output(
        obs_samples=obs_samples,
        init_action_samples=init_action_samples,
        labels=labels,
        meta=meta,
        args=args,
        refinement_steps=refinement_steps,
        stats=stats,
        extra_tensors=extra_tensors,
    )
    output['config'].update({
        'label_mode': args.label_mode,
        'label_strategy': 'action_diff_strong_trajectory' if args.label_mode == 'action_diff' else 'fixed_success_smallest_k',
        'strong_refinement_steps': int(strong_k),
        'zero_refinement_step': int(args.zero_refinement_step),
        'max_label_refinement_step': int(max_label_step),
        'threshold': float(args.threshold),
        'threshold_mode': args.threshold_mode,
        'threshold_eta': float(args.threshold_eta),
        'effective_threshold': stats.get('effective_threshold', None),
        'success_threshold': float(args.success_threshold),
    })

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(output, args.output)

    label_tensor = output['label']
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f"  samples: {label_tensor.numel()}")
    print(f"  obs: {tuple(output['obs'].shape)}")
    print(f"  init_action: {tuple(output['init_action'].shape)}")
    print(f"  label counts: {stats.get('label_counts', {})}")
    print(f"  label counts by k: {stats.get('label_counts_by_k', {})}")
    if args.label_mode == 'action_diff':
        print(f"  action_diff median: {stats['action_diff_median']:.6f}")
        print(f"  effective_threshold: {stats['effective_threshold']:.6f}")
    print(f"  saved: {args.output}")


if __name__ == '__main__':
    main()
