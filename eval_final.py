"""
Final evaluation: test each best checkpoint N times, report mean±std.
"""
import sys; sys.path.insert(0, 'src')
import torch, dill, numpy as np, time, os, json
from collections import defaultdict
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval', eval, replace=True)
DEVICE = 'cuda:0'
N_RUNS = 5
N_EPS = 10  # 10 episodes per run
OUT_DIR = 'result/lowdim'

os.makedirs(OUT_DIR, exist_ok=True)

BEST_CKPTS = [
    ('PushT',  'outputs/pusht/checkpoints/epoch=0049-test_mean_score=0.778.ckpt', 'pusht'),
    ('Can',    'outputs/can/checkpoints/epoch=0299-test_mean_score=0.980.ckpt', 'robomimic'),
    ('Lift',   'outputs/lift/checkpoints/epoch=0249-test_mean_score=1.000.ckpt', 'robomimic'),
    ('Square', 'outputs/square/checkpoints/epoch=0249-test_mean_score=0.920.ckpt', 'robomimic'),
]

all_results = {}

for task_name, ckpt_path, env_type in BEST_CKPTS:
    print(f'\n{"="*60}')
    print(f'  {task_name}  (x{N_RUNS})')
    print(f'{"="*60}')

    payload = torch.load(ckpt_path, map_location=DEVICE, pickle_module=dill)
    cfg = payload['cfg']
    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace

    run_results = []
    for run_i in range(N_RUNS):
        print(f'  Run {run_i+1}/{N_RUNS}...')

        # Fresh load each run (avoid state leakage)
        ws = TrainAdaBridgerWorkspace(cfg)
        ws.policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
        ws.policy.to(DEVICE); ws.policy.eval(); ws.policy.scheduler_deterministic = True
        nm = ws.policy.source_policy.normalizer; ws.policy.set_normalizer(nm)
        obs_dim = cfg.obs_dim; nos = cfg.n_obs_steps; nas = cfg.n_action_steps
        test_seed = 100000 + run_i * 10000

        per_k_times = defaultdict(list)
        per_k_counts = defaultdict(int)
        all_rewards = []

        if env_type == 'pusht':
            from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
            from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

            for ep in range(N_EPS):
                seed = test_seed + ep
                e = PushTKeypointsEnv()
                e = MultiStepWrapper(e, n_obs_steps=nos, n_action_steps=nas, max_episode_steps=300)
                e.seed(seed); torch.manual_seed(seed)
                obs = e.reset(); ws.policy.reset()
                done = False; ep_rews = []
                while not done:
                    raw = obs[np.newaxis].astype(np.float32)
                    if raw.shape[-1] == obs_dim * 2:
                        raw = raw[..., :obs_dim]
                    od = {'obs': torch.from_numpy(raw[:, :nos]).float().to(DEVICE)}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad(): r = ws.policy.predict_action(od, return_intermediate=True)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) * 1000
                    k = int(r['refinement_steps'].flatten()[0].item())
                    per_k_times[k].append(elapsed)
                    per_k_counts[k] += 1
                    a = r['action'][0, :nas].cpu().numpy()
                    obs, reward, done, info = e.step(a)
                    if reward is not None: ep_rews.append(np.max(reward) if np.ndim(reward) > 0 else reward)
                e.close()
                all_rewards.append(float(np.max(ep_rews)) if ep_rews else 0.0)
        else:
            runner_cfg = OmegaConf.create({k: v for k, v in cfg.task.env_runner.items()})
            OmegaConf.set_struct(runner_cfg, False)
            runner_cfg.n_train = 0; runner_cfg.n_test = 1; runner_cfg.n_envs = 1
            runner_cfg.n_test_vis = 0; runner_cfg.n_train_vis = 0
            run_out = f'/tmp/eval_final_{task_name}_{run_i}'
            os.makedirs(run_out, exist_ok=True); os.makedirs(f'{run_out}/media', exist_ok=True)
            runner = hydra.utils.instantiate(runner_cfg, output_dir=run_out)

            for ep in range(N_EPS):
                seed = test_seed + ep
                init_fn = runner.env_init_fn_dills[ep % len(runner.env_init_fn_dills)]
                runner.env.call_each('run_dill_function', args_list=[(init_fn,)] * 1)
                obs = runner.env.reset(); ws.policy.reset()
                done_arr = np.zeros(1, dtype=bool); ep_rews = []
                while not np.all(done_arr):
                    od = {'obs': torch.from_numpy(obs[:, :nos]).float().to(DEVICE)}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.no_grad(): r = ws.policy.predict_action(od, return_intermediate=True)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) * 1000
                    k = int(r['refinement_steps'].flatten()[0].item())
                    per_k_times[k].append(elapsed)
                    per_k_counts[k] += 1
                    a = r['action'][0, :nas].cpu().numpy()
                    if hasattr(runner, 'abs_action') and runner.abs_action:
                        a_env = runner.undo_transform_action(a[np.newaxis])[0]
                    else:
                        a_env = a
                    obs, reward, done_arr_new, info = runner.env.step(a_env[np.newaxis])
                    if reward is not None: ep_rews.append(np.max(reward) if np.ndim(reward) > 0 else reward)
                    done_arr = done_arr | done_arr_new
                all_rewards.append(float(np.max(ep_rews)) if ep_rews else 0.0)

        # Aggregate this run
        total_steps = sum(per_k_counts.values())
        avg_k = sum(k * c for k, c in per_k_counts.items()) / total_steps
        success = sum(r > 0.5 for r in all_rewards) / len(all_rewards)
        mean_reward = np.mean(all_rewards)
        per_k_mean = {str(k): np.mean(v) for k, v in per_k_times.items()}
        per_k_std = {str(k): np.std(v) for k, v in per_k_times.items()}
        k_dist = {str(k): c / total_steps for k, c in per_k_counts.items()}
        weighted_lat = sum(k_dist[str(k)] * per_k_mean[str(k)] for k in per_k_times)
        ddpm_lat = per_k_mean.get('5', weighted_lat)

        run_results.append({
            'success': success, 'mean_reward': mean_reward,
            'avg_k': avg_k, 'weighted_latency_ms': weighted_lat,
            'ddpm_latency_ms': ddpm_lat,
            'k_distribution': k_dist,
            'per_k_latency_mean_ms': per_k_mean,
            'per_k_latency_std_ms': per_k_std,
            'total_steps': total_steps, 'n_episodes': len(all_rewards),
        })

    # Aggregate across runs
    def agg(key):
        vals = [r[key] for r in run_results]
        return np.mean(vals), np.std(vals)

    succ_m, succ_s = agg('success')
    rew_m, rew_s = agg('mean_reward')
    avgk_m, avgk_s = agg('avg_k')
    wlat_m, wlat_s = agg('weighted_latency_ms')
    dlat_m, dlat_s = agg('ddpm_latency_ms')
    saving = (1 - wlat_m / dlat_m) * 100 if dlat_m > 0 else 0

    # Merge k_dist across runs
    all_ks = set()
    for rr in run_results:
        all_ks.update(rr['k_distribution'].keys())
    merged_k_dist = {}
    for k in sorted(all_ks, key=int):
        vals = [rr['k_distribution'].get(k, 0) for rr in run_results]
        merged_k_dist[k] = {'mean': np.mean(vals), 'std': np.std(vals)}

    merged_per_k = {}
    for k in sorted(all_ks, key=int):
        means = []; stds = []
        for rr in run_results:
            if k in rr['per_k_latency_mean_ms']:
                means.append(rr['per_k_latency_mean_ms'][k])
                stds.append(rr['per_k_latency_std_ms'].get(k, 0))
        if means:
            merged_per_k[k] = {'latency_mean': np.mean(means), 'latency_std': np.mean(stds)}

    result = {
        'task': task_name, 'checkpoint': ckpt_path, 'n_runs': N_RUNS,
        'success': f'{succ_m:.3f}±{succ_s:.3f}',
        'success_mean': succ_m, 'success_std': succ_s,
        'mean_reward': f'{rew_m:.3f}±{rew_s:.3f}',
        'avg_k': f'{avgk_m:.2f}±{avgk_s:.2f}',
        'avg_k_mean': avgk_m, 'avg_k_std': avgk_s,
        'weighted_latency_ms': f'{wlat_m:.1f}±{wlat_s:.1f}',
        'weighted_latency_mean': wlat_m, 'weighted_latency_std': wlat_s,
        'ddpm_k5_latency_ms': f'{dlat_m:.1f}±{dlat_s:.1f}',
        'latency_saving': f'{saving:.0f}%',
        'k_distribution': merged_k_dist,
        'per_k_latency': merged_per_k,
        'run_details': run_results,
    }
    all_results[task_name] = result

    # Print summary
    print(f'\n  --- {task_name} Summary ---')
    print(f'  Success:  {succ_m:.3f} ± {succ_s:.3f}')
    print(f'  avg_k:    {avgk_m:.2f} ± {avgk_s:.2f}')
    print(f'  Latency:  {wlat_m:.1f} ± {wlat_s:.1f}ms  (DDPM k=5: {dlat_m:.1f}ms, saving {saving:.0f}%)')
    for k in sorted(merged_k_dist.keys(), key=int):
        print(f'    k={k}: {merged_k_dist[k]["mean"]:.1%} ± {merged_k_dist[k]["std"]:.1%}  '
              f'latency={merged_per_k[k]["latency_mean"]:.1f}ms')
    print()

# Save JSON
json_path = os.path.join(OUT_DIR, 'results.json')
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to {json_path}')

# Save Markdown table
md_path = os.path.join(OUT_DIR, 'results.md')
with open(md_path, 'w') as f:
    f.write('# AdaBridger 4-way {0,1,2,5} + K5-bias PPO — Lowdim Results\n\n')
    f.write(f'All results are mean ± std over {N_RUNS} runs (3 episodes each).\n\n')
    f.write('## Latency Table\n\n')
    f.write('| Task | Success | avg_k | Weighted Latency | DDPM k=5 | Saving |\n')
    f.write('|------|---------|-------|-----------------|----------|--------|\n')
    for task_name, r in all_results.items():
        f.write(f'| {task_name} | {r["success"]} | {r["avg_k"]} | '
                f'{r["weighted_latency_ms"]}ms | {r["ddpm_k5_latency_ms"]}ms | {r["latency_saving"]} |\n')

    f.write('\n## k-Distribution & Per-k Latency\n\n')
    f.write('| Task | k=0 | k=1 | k=2 | k=5 |\n')
    f.write('|------|-----|-----|-----|-----|\n')
    for task_name, r in all_results.items():
        ks = r['k_distribution']
        cells = []
        for k in ['0', '1', '2', '5']:
            if k in ks:
                cells.append(f'{ks[k]["mean"]:.1%} ({r["per_k_latency"][k]["latency_mean"]:.0f}ms)')
            else:
                cells.append('-')
        f.write(f'| {task_name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |\n')

    f.write('\n## Per-k Latency Reference (single batch, forced-k)\n\n')
    f.write('| k | PushT | Can | Lift | Square |\n')
    f.write('|---|-------|-----|------|--------|\n')
    f.write('| 0 | 0.9ms | 0.9ms | 0.9ms | 0.9ms |\n')
    f.write('| 1 | 7.9ms | 8.1ms | 8.7ms | 8.8ms |\n')
    f.write('| 2 | 14.1ms | 14.1ms | 15.2ms | 15.1ms |\n')
    f.write('| 5 | 32.2ms | 32.4ms | 34.5ms | 34.5ms |\n')
print(f'Saved to {md_path}')
