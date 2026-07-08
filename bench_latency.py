"""Standalone single-env latency benchmark for AdaBridger checkpoints."""
import sys; sys.path.insert(0, 'src')
import torch, dill, numpy as np, time, os
from collections import defaultdict
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval', eval, replace=True)
DEVICE = 'cuda:0'

TASKS = [
    ('PushT', 'outputs/pusht/checkpoints/epoch=0049-test_mean_score=0.778.ckpt', 'pusht'),
    ('Can', 'outputs/can/checkpoints/epoch=0049-test_mean_score=0.980.ckpt', 'robomimic'),
    ('Lift', 'outputs/lift/checkpoints/epoch=0049-test_mean_score=1.000.ckpt', 'robomimic'),
    ('Square', 'outputs/square/checkpoints/epoch=0049-test_mean_score=0.920.ckpt', 'robomimic'),
]

for task_name, ckpt_path, env_type in TASKS:
    print(f'\n{"="*50}')
    print(f'  {task_name}')
    print(f'{"="*50}')

    # Load policy
    payload = torch.load(ckpt_path, map_location=DEVICE, pickle_module=dill)
    cfg = payload['cfg']
    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace
    ws = TrainAdaBridgerWorkspace(cfg)
    ws.policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
    ws.policy.to(DEVICE); ws.policy.eval()
    ws.policy.scheduler_deterministic = True
    nm = ws.policy.source_policy.normalizer; ws.policy.set_normalizer(nm)

    obs_dim = cfg.obs_dim; nos = cfg.n_obs_steps; nas = cfg.n_action_steps

    # ── Forced-k timing (single batch, 100 iterations) ──
    # Get a real observation batch
    if env_type == 'pusht':
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        e = PushTKeypointsEnv()
        e = MultiStepWrapper(e, n_obs_steps=nos, n_action_steps=nas, max_episode_steps=300)
        e.seed(0); torch.manual_seed(0)
        obs = e.reset(); e.close()
        raw = obs[np.newaxis].astype(np.float32)
        if raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_batch = torch.from_numpy(raw[:, :nos]).float().to(DEVICE)
    else:
        ds_cfg = cfg.task.dataset
        ds = hydra.utils.instantiate(ds_cfg)
        s = ds[0]
        obs_batch = s['obs'].unsqueeze(0).to(DEVICE).float()[:, :nos]
        # robomimic dataset gives [B, nos, combined_obs_dim], normalize
        nm_local = ds.get_normalizer()
        nm_local.to(DEVICE)
        obs_batch = nm_local['obs'].normalize(obs_batch)

    od = {'obs': obs_batch}

    # Force each k and time
    class ForceK(torch.nn.Module):
        def __init__(self, kv): super().__init__(); self.k = kv
        def select_action(self, o, i, deterministic=True):
            B = o.shape[0]; dev = o.device
            return (torch.full((B,), self.k, dtype=torch.long, device=dev),
                    torch.full((B,), 0, dtype=torch.long, device=dev),
                    torch.zeros(B, device=dev), torch.zeros(B, device=dev))

    original_scheduler = ws.policy.scheduler  # Save for later
    k_options = [0, 1, 2, 5]
    forced_times = {}
    for k in k_options:
        ws.policy.scheduler = ForceK(k).to(DEVICE)
        # Warmup
        for _ in range(5):
            with torch.no_grad(): ws.policy.predict_action(od)
        torch.cuda.synchronize()
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            with torch.no_grad(): ws.policy.predict_action(od)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        forced_times[k] = np.mean(times)
    ws.policy.scheduler = original_scheduler  # Restore real scheduler

    print(f'  Forced-k latency (single batch, 200 runs):')
    for k in k_options:
        print(f'    k={k}: {forced_times[k]:.1f}ms')

    # ── Real scheduler timing (single env, timed per step) ──
    per_k_times = defaultdict(list)
    n_episodes = 3

    for ep in range(n_episodes):
        seed = 100000 + ep
        if env_type == 'pusht':
            e = PushTKeypointsEnv()
            e = MultiStepWrapper(e, n_obs_steps=nos, n_action_steps=nas, max_episode_steps=300)
            e.seed(seed); torch.manual_seed(seed)
            obs = e.reset()
            ws.policy.reset()
            done = False
            while not done:
                raw = obs[np.newaxis].astype(np.float32)
                if raw.shape[-1] == obs_dim * 2:
                    raw = raw[..., :obs_dim]
                od_e = {'obs': torch.from_numpy(raw[:, :nos]).float().to(DEVICE)}
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(): r = ws.policy.predict_action(od_e, return_intermediate=True)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                k = int(r['refinement_steps'].flatten()[0].item())
                per_k_times[k].append(elapsed)
                a = r['action'][0, :nas].cpu().numpy()
                obs, reward, done, info = e.step(a)
            e.close()
        else:
            # Robomimic: use async env with n_envs=1
            runner_cfg = OmegaConf.create({k: v for k, v in cfg.task.env_runner.items()})
            OmegaConf.set_struct(runner_cfg, False)
            runner_cfg.n_train = 0; runner_cfg.n_test = 1; runner_cfg.n_envs = 1
            runner_cfg.n_test_vis = 0; runner_cfg.n_train_vis = 0
            out_dir = f'/tmp/bench_{task_name.lower()}_{ep}'
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(f'{out_dir}/media', exist_ok=True)
            runner = hydra.utils.instantiate(runner_cfg, output_dir=out_dir)

            init_fn = runner.env_init_fn_dills[0]
            runner.env.call_each('run_dill_function', args_list=[(init_fn,)] * 1)
            obs = runner.env.reset()
            ws.policy.reset()
            done_arr = np.zeros(1, dtype=bool)
            while not np.all(done_arr):
                od_e = {'obs': torch.from_numpy(obs[:, :nos]).float().to(DEVICE)}
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(): r = ws.policy.predict_action(od_e, return_intermediate=True)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000
                k = int(r['refinement_steps'].flatten()[0].item())
                per_k_times[k].append(elapsed)
                a = r['action'][0, :nas].cpu().numpy()
                # Robomimic abs: undo absolute action transform
                if hasattr(runner, 'abs_action') and runner.abs_action:
                    a_env = runner.undo_transform_action(a[np.newaxis])[0]
                else:
                    a_env = a
                obs, reward, done_arr_new, info = runner.env.step(a_env[np.newaxis])
                done_arr = done_arr | done_arr_new

    # ── Results ──
    all_k = [t for ts in per_k_times.values() for t in ts]
    k_dist = {k: len(v) / len(all_k) for k, v in per_k_times.items()}
    avg_k = sum(k * len(v) for k, v in per_k_times.items()) / len(all_k)
    weighted = sum(k_dist[k] * np.mean(per_k_times[k]) for k in per_k_times)

    print(f'  Real scheduler ({n_episodes} episodes, {len(all_k)} steps):')
    print(f'    avg_k = {avg_k:.2f}')
    for k in sorted(per_k_times.keys()):
        print(f'    k={k}: {np.mean(per_k_times[k]):.1f}ms ± {np.std(per_k_times[k]):.1f}ms  '
              f'({len(per_k_times[k])} steps, {k_dist[k]:.1%})')
    print(f'    Weighted avg: {weighted:.1f}ms')

    if 5 in per_k_times:
        ddpm_time = np.mean(per_k_times[5])
        saving = (1 - weighted / ddpm_time) * 100
        print(f'    vs DDPM k=5: {ddpm_time:.1f}ms → saving {saving:.0f}%')
