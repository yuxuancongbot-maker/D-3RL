"""
Quick eval of trained AdaBridger scheduler on PushT test set.
"""
import torch, dill, numpy as np, time, sys
from collections import Counter
from omegaconf import OmegaConf

device = 'cuda:0'
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
    'outputs/train/checkpoints/epoch=0249-test_mean_score=0.710.ckpt'

print(f'Loading checkpoint: {ckpt_path}')
payload = torch.load(ckpt_path, map_location=device, pickle_module=dill)
cfg = payload['cfg']

# ── Rebuild workspace ──
import hydra
from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace

# Fix checkpoint config keys for format string compatibility
if 'checkpoint' in cfg and 'topk' in cfg.checkpoint:
    cfg.checkpoint.topk.monitor_key = 'test_mean_score'

ws = TrainAdaBridgerWorkspace(cfg)
ws.policy.load_state_dict(payload['state_dicts']['policy'], strict=False)
ws.policy.to(device)
ws.policy.eval()
ws.policy.scheduler_deterministic = True  # deterministic argmax

# ── Set normalizer ──
from diffusion_policy.model.common.normalizer import LinearNormalizer
normalizer = ws.policy.source_policy.normalizer
ws.policy.set_normalizer(normalizer)

print(f'Policy loaded. scheduler_deterministic={ws.policy.scheduler_deterministic}')

# ── Run test episodes ──
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

n_test = 50
test_start_seed = 100000
horizon = cfg.horizon
n_obs_steps = cfg.n_obs_steps
n_action_steps = cfg.n_action_steps
obs_dim = cfg.obs_dim
max_steps = 300

results = []
all_step_choices = []
all_timings = []

for ep in range(n_test):
    seed = test_start_seed + ep
    env = PushTKeypointsEnv()
    env = MultiStepWrapper(env, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
                           max_episode_steps=max_steps)
    env.seed(seed)
    torch.manual_seed(seed)
    obs = env.reset()
    ws.policy.reset()

    done = False
    ep_rewards = []
    ep_steps = []
    ep_times = []

    while not done:
        raw = obs[np.newaxis].astype(np.float32)
        if raw.shape[-1] == obs_dim * 2:
            raw = raw[..., :obs_dim]
        obs_tensor = raw[:, :n_obs_steps]
        obs_dict = {'obs': torch.from_numpy(obs_tensor).to(device)}

        t0 = time.perf_counter()
        with torch.no_grad():
            result = ws.policy.predict_action(obs_dict, return_intermediate=True)
        elapsed = (time.perf_counter() - t0) * 1000
        ep_times.append(elapsed)

        k = int(result['refinement_steps'].flatten()[0].item())
        ep_steps.append(k)

        action = result['action'][0, :n_action_steps].cpu().numpy()
        obs, reward, done, info = env.step(action)
        if reward is not None:
            ep_rewards.append(np.max(reward) if np.ndim(reward) > 0 else reward)

    env.close()
    final_r = float(np.max(ep_rewards)) if ep_rewards else 0.0
    success = final_r > 0.5
    results.append({'seed': seed, 'reward': final_r, 'success': success,
                    'avg_k': np.mean(ep_steps), 'k_dist': dict(Counter(ep_steps)),
                    'avg_ms': np.mean(ep_times), 'len': len(ep_steps)})
    all_step_choices.extend(ep_steps)
    all_timings.extend(ep_times)

# ── Print results ──
succ = [r['success'] for r in results]
rews = [r['reward'] for r in results]
ks = [r['avg_k'] for r in results]
ms = [r['avg_ms'] for r in results]

print()
print('=' * 60)
print('TEST SET RESULTS (50 episodes, seeds 100000-100049)')
print('=' * 60)
print(f'  Success rate:    {np.mean(succ):.1%} ({sum(succ)}/50)')
print(f'  Mean reward:     {np.mean(rews):.3f} ± {np.std(rews):.3f}')
print(f'  Mean k:          {np.mean(ks):.2f}  (k=0 → VAE, k=5 → full refine)')
print(f'  Mean latency:    {np.mean(ms):.1f} ms')
print(f'  Median latency:  {np.median(ms):.1f} ms')

# Step distribution
total = len(all_step_choices)
step_dist = Counter(all_step_choices)
print(f'\n  Refinement step distribution:')
for k in sorted(step_dist.keys()):
    print(f'    k={k}: {step_dist[k]/total:.1%} ({step_dist[k]}/{total})')

# Per-episode detail
print(f'\n  Per-episode:')
for r in results:
    avg_k = r['avg_k']
    avg_ms = r['avg_ms']
    kd = r['k_dist']
    print(f'    seed={r["seed"]:6d}  reward={r["reward"]:.3f}  '
          f'{"OK" if r["success"] else "FAIL"}  avg_k={avg_k:.2f}  {avg_ms:.1f}ms  dist={kd}')

print()
print('Baseline comparison:')
print(f'  VAE only (k=0):     45.0% success,  1.4ms')
print(f'  DDPM refiner (k=5): 71.5% success, 35.0ms')
print(f'  AdaBridger (best):  {np.mean(succ):.1%} success, {np.mean(ms):.1f}ms')
print('=' * 60)
