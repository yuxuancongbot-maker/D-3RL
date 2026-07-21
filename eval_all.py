"""Full eval: all lowdim tasks, best CKPT, 3 runs x 50 seeds. Latency + avg_k."""
import sys; sys.path.insert(0, 'src')
import torch, dill, numpy as np, time, json
from collections import Counter
from omegaconf import OmegaConf; import hydra
OmegaConf.register_new_resolver('eval', eval, replace=True)
DEV = 'cuda:0'

TASKS = {
    'pusht':       ('outputs/pusht/checkpoints/epoch=0399-test_mean_score=0.681.ckpt', 20, 2),
    'lift':        ('outputs/lift/checkpoints/epoch=0049-test_mean_score=1.000.ckpt', 19, 10),
    'can':         ('outputs/can/checkpoints/epoch=0049-test_mean_score=0.980.ckpt', 23, 10),
    'square':      ('outputs/square/checkpoints/epoch=0049-test_mean_score=0.920.ckpt', 23, 10),
    'transport':   ('outputs/train_transport/checkpoints/epoch=0149-test_mean_score=0.820.ckpt', 59, 20),
    'tool_hang':   ('outputs/train_toolhang/checkpoints/epoch=0049-test_mean_score=0.460.ckpt', 53, 10),
}
TASK_CONFIGS = {
    'lift':'lift_lowdim_abs','can':'can_lowdim_abs','square':'square_lowdim_abs',
    'transport':'transport_lowdim_abs','tool_hang':'tool_hang_lowdim_abs',
}

def load_policy(ckpt):
    payload = torch.load(ckpt, map_location=DEV, pickle_module=dill)
    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace
    ws = TrainAdaBridgerWorkspace(payload['cfg'])
    ws.policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
    ws.policy.to(DEV); ws.policy.eval(); ws.policy.scheduler_deterministic = True
    ws.policy.set_normalizer(ws.policy.source_policy.normalizer)
    return ws.policy

def make_env(name, seed, obs_dim):
    if name == 'pusht':
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        env = PushTKeypointsEnv()
        env = MultiStepWrapper(env, n_obs_steps=2, n_action_steps=8, max_episode_steps=300)
        env.seed(seed); torch.manual_seed(seed)
        return env, env.reset()
    else:
        tcfg = OmegaConf.create({
            '_target_': 'diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner',
            'output_dir': '/tmp/eval_rm',
            'dataset_path': f'data/robomimic/datasets/{name}/ph/low_dim_abs.hdf5',
            'obs_keys': ['object','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'],
            'n_train':0,'n_train_vis':0,'train_start_idx':0,
            'n_test':1,'n_test_vis':0,'test_start_seed':seed,
            'max_steps':400 if name in ('lift','can','square') else 700,
            'n_obs_steps':2,'n_action_steps':8,'n_latency_steps':0,
            'render_hw':[128,128],'fps':10,'crf':22,
            'past_action':False,'abs_action':True,'n_envs':1,
        })
        runner = hydra.utils.instantiate(tcfg)
        env = runner.env
        init_fn = dill.loads(runner.env_init_fn_dills[0])
        env.call_each('run_dill_function', args_list=[(init_fn,)] * 1)
        obs = env.reset()
        return env, obs

def run_ep(pol, env, obs, obs_dim, is_push):
    steps, times, rews = [], [], []
    done = False
    while not done:
        if is_push:
            raw = obs[np.newaxis].astype(np.float32)
            if raw.shape[-1] == obs_dim*2: raw = raw[...,:obs_dim]
            od = {'obs': torch.from_numpy(raw[:,:2]).to(DEV)}
        else:
            od = {k: torch.from_numpy(v).float().unsqueeze(0).to(DEV) for k,v in obs.items()}
        t0 = time.perf_counter()
        with torch.no_grad(): r = pol.predict_action(od, return_intermediate=True)
        t = (time.perf_counter() - t0) * 1000
        k = int(r['refinement_steps'].flatten()[0].item())
        steps.append(k); times.append(t)
        a = r['action'][0,:8].cpu().numpy()
        obs, reward, done, info = env.step(a)
        if isinstance(done, np.ndarray): done = np.all(done)
        if reward is not None: rews.append(float(np.max(reward)) if np.ndim(reward)>0 else float(reward))
    mr = float(np.max(rews)) if rews else 0.0
    return mr > 0.5, mr, np.mean(steps), np.mean(times)

results = {}
for name, (ckpt, obs_dim, act_dim) in TASKS.items():
    print(f'\n=== {name} ({ckpt}) ===')
    pol = load_policy(ckpt)
    is_push = (name == 'pusht')
    task_runs = []
    for run_idx in range(3):
        succs, aks, lats = [], [], []
        for seed in range(100000, 100050):
            env, obs = make_env(name, seed, obs_dim)
            pol.reset()
            s, r, ak, lat = run_ep(pol, env, obs, obs_dim, is_push)
            succs.append(s); aks.append(ak); lats.append(lat)
            env.close()
        task_runs.append({'succ':np.mean(succs), 'ak':np.mean(aks), 'lat':np.mean(lats)})
        tr = task_runs[-1]
        print("  Run %d: succ=%.1f%% avg_k=%.2f lat=%.1fms" % (run_idx+1, tr["succ"]*100, tr["ak"], tr["lat"]))
    ss = [r['succ'] for r in task_runs]; kk = [r['ak'] for r in task_runs]; ll = [r['lat'] for r in task_runs]
    results[name] = {
        'success': f'{np.mean(ss):.1%} ± {np.std(ss):.1%}',
        'avg_k': f'{np.mean(kk):.2f} ± {np.std(kk):.2f}',
        'latency_ms': f'{np.mean(ll):.1f} ± {np.std(ll):.1f}',
    }

print('\n' + '='*70)
for n, r in results.items():
        print("%-12s succ=%-20s avg_k=%-16s lat=%-16s" % (n, r["success"], r["avg_k"], r["latency_ms"]))

with open('experiments/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved experiments/eval_results.json')
