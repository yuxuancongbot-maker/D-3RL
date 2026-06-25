# PushT Inference Guide

加载 source policy（VAE）和 diffusion refiner 在 PushT 任务上推理的完整流程。

## 前置条件

```bash
cd /path/to/d3rl_diffusion_policy
pip install -e ".[dev,sim]"
```

需要的 checkpoint 文件：

| 组件 | 文件 | 说明 |
|------|------|------|
| Source Policy (System 1) | `weights/source_vae.ckpt` | 条件 VAE，快速生成初始动作 |
| Diffusion Refiner (System 2) | `weights/diffusion.ckpt` | 扩散策略，精炼动作 |
| Scheduler (Stage 3 输出) | `weights/scheduler.ckpt` | 可选，决定是否需要精炼 |

## 方式 1：纯 Source Policy 推理（无精炼）

只用 VAE 产出动作，跳过 scheduler 和 diffusion refiner。适合测试 source policy 的 baseline 性能。

```python
import torch
import hydra
import dill
import numpy as np
from omegaconf import OmegaConf

# ── 加载 Source Policy (VAE) ──────────────────────────────────────
ckpt_path = "weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt"
payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']

# 确保 backend 是 VAE
state_dict = payload['state_dicts']['model']
looks_like_vae = any('net.encoder_net' in k for k in state_dict.keys())

OmegaConf.set_struct(cfg.policy, False)
if looks_like_vae:
    cfg.policy.backend = 'vae'
    cfg.policy.model = {
        '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
        'action_dim': cfg.policy.action_dim,
        'action_horizon': cfg.policy.horizon,
        'obs_dim': cfg.policy.obs_dim,
        'obs_horizon': cfg.policy.n_obs_steps,
        'latent_dim': 32,
        'layer': 256,
        'use_ema': True,
        'pretrain': False,
        'ckpt_path': None,
    }

source_policy = hydra.utils.instantiate(cfg.policy)
source_policy.load_state_dict(state_dict, strict=False)
source_policy.eval()
source_policy.to("cuda:0")

# 加载 normalizer
if 'normalizer' in payload['state_dicts']:
    source_policy.normalizer.load_state_dict(
        payload['state_dicts']['normalizer'], strict=False
    )

print(f"Source policy loaded: {type(source_policy).__name__}")
print(f"  obs_dim={cfg.policy.obs_dim}, action_dim={cfg.policy.action_dim}")
print(f"  horizon={cfg.policy.horizon}, n_obs_steps={cfg.policy.n_obs_steps}")
```

### 创建 PushT 环境

```python
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# 创建环境
n_obs_steps = cfg.policy.n_obs_steps     # 通常 2
n_action_steps = cfg.policy.n_action_steps  # 通常 8
obs_dim = cfg.policy.obs_dim             # 通常 2 (PushT keypoints)
action_dim = cfg.policy.action_dim       # 通常 2

env = PushTKeypointsEnv()
env = MultiStepWrapper(
    env,
    n_obs_steps=n_obs_steps,
    n_action_steps=n_action_steps,
    max_episode_steps=300,
)

obs = env.reset()
print(f"obs shape: {obs.shape}")  # [1, 4]  — PushT 拼接了 keypoints + visibility mask
```

### 推理循环（纯 VAE，k=0）

```python
source_policy.reset()
done = False
step_count = 0
all_rewards = []

while not done:
    # 准备输入：PushT env 的 obs 末尾是 visibility mask，需去掉
    raw_obs = obs[np.newaxis].astype(np.float32)
    if raw_obs.shape[-1] == obs_dim * 2:
        raw_obs = raw_obs[..., :obs_dim]  # 只取前半：真正的 keypoints

    obs_dict = {
        'obs': torch.from_numpy(raw_obs[:, :n_obs_steps]).to("cuda:0")
    }

    with torch.no_grad():
        result = source_policy.predict_action(obs_dict)

    action = result['action_pred']  # [1, horizon, action_dim]
    np_action = action.cpu().numpy()

    # 执行动作（取前 n_action_steps 步）
    action_for_env = np_action[:, :n_action_steps]
    obs, reward, done, info = env.step(action_for_env)

    step_count += 1
    if reward is not None:
        all_rewards.append(reward)

print(f"Episode finished: {step_count} steps, "
      f"final reward={all_rewards[-1] if all_rewards else 0}")
```

---

## 方式 2：Source Policy + Diffusion Refiner 推理

加载 VAE source policy 和 diffusion refiner，手动控制 DDIM 精炼步数。

### 加载两个模型

```python
import torch, dill, hydra, numpy as np
from omegaconf import OmegaConf
from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler


def load_policy(ckpt_path, device="cuda:0"):
    """加载单个策略（VAE 或 Diffusion）"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    state_dict = payload['state_dicts']['model']

    looks_like_vae = any('net.encoder_net' in k for k in state_dict.keys())

    OmegaConf.set_struct(cfg.policy, False)
    if looks_like_vae:
        cfg.policy.backend = 'vae'
        cfg.policy.model = {
            '_target_': 'diffusion_policy.model.action_predictor.vae_action_predictor.VAEModel',
            'action_dim': cfg.policy.action_dim,
            'action_horizon': cfg.policy.horizon,
            'obs_dim': cfg.policy.obs_dim,
            'obs_horizon': cfg.policy.n_obs_steps,
            'latent_dim': 32, 'layer': 256,
            'use_ema': True, 'pretrain': False, 'ckpt_path': None,
        }

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)
    return policy, cfg


# ── 加载 ───────────────────────────────────────────────────────────
source_policy, src_cfg = load_policy(
    "weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt"
)
refine_policy, ref_cfg = load_policy(
    "weights/lowdim/diffusion/pusht/cnn/latest.ckpt"
)

# 共享 normalizer（source 没有 normalizer 时从 refine 借）
from diffusion_policy.model.common.normalizer import LinearNormalizer
try:
    _ = source_policy.normalizer['obs']
    normalizer = source_policy.normalizer
except (AttributeError, KeyError):
    normalizer = refine_policy.normalizer

print(f"Source: {type(source_policy).__name__}")
print(f"Refiner: {type(refine_policy).__name__}")
```

### 创建 D3RL Policy（手动 scheduler 控制 k）

```python
# 创建一个始终返回固定 refinement_steps 的 scheduler
class FixedKScheduler(torch.nn.Module):
    def __init__(self, k, refinement_steps):
        super().__init__()
        self.k = k
        self.refinement_steps = refinement_steps
        self.k_idx = refinement_steps.index(k) if k in refinement_steps else 0
        self.register_buffer('refinement_steps_tensor',
                            torch.tensor(refinement_steps, dtype=torch.long))

    def select_action(self, obs, init_action, deterministic=True):
        B = obs.shape[0]
        device = obs.device
        k_idx = torch.full((B,), self.k_idx, dtype=torch.long, device=device)
        steps = torch.full((B,), self.k, dtype=torch.long, device=device)
        log_prob = torch.zeros(B, device=device)
        value = torch.zeros(B, device=device)
        return steps, k_idx, log_prob, value

# 创建 D3RL policy，scheduler 固定 k=5（最强精炼）
refinement_steps = [0, 1, 2, 5]
for test_k in [0, 1, 2, 5]:
    scheduler = FixedKScheduler(test_k, refinement_steps).to("cuda:0")

    d3rl_policy = AdaBridgerPolicy(
        source_policy=source_policy,
        refinement_policy=refine_policy,
        scheduler=scheduler,
        horizon=16,
        obs_dim=2,
        action_dim=2,
        n_action_steps=8,
        n_obs_steps=2,
        refinement_steps=refinement_steps,
        max_refinement_steps=5,
        scheduler_deterministic=True,
        freeze_backbone=True,
    )
    d3rl_policy.set_normalizer(normalizer)
    d3rl_policy.to("cuda:0")
    d3rl_policy.eval()

    # ── 推理 ───────────────────────────────────────────────────────
    d3rl_policy.reset()
    obs = env.reset()
    done = False
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    timings = []

    while not done:
        raw_obs = obs[np.newaxis].astype(np.float32)
        if raw_obs.shape[-1] == obs_dim * 2:
            raw_obs = raw_obs[..., :obs_dim]

        obs_dict = {'obs': torch.from_numpy(raw_obs[:, :2]).to("cuda:0")}

        t_start.record()
        with torch.no_grad():
            result = d3rl_policy.predict_action(obs_dict)
        t_end.record()
        torch.cuda.synchronize()
        timings.append(t_start.elapsed_time(t_end))

        np_action = result['action'].cpu().numpy()
        obs, reward, done, info = env.step(np_action[:, :8])

    avg_ms = np.mean(timings)
    print(f"k={test_k}: avg inference {avg_ms:.1f} ms "
          f"(~{1000/avg_ms:.0f} Hz), {len(timings)} steps")
```

---

## 方式 3：CLI 命令

### 纯 Source Policy 推理（最快）

```bash
# 直接用 VAE checkpoint 评估
dp-eval --checkpoint weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
    --output-dir outputs/eval_vae_only --device cuda:0
```

### 纯 Diffusion Refiner（全步 DDIM 推理）

```bash
dp-eval --checkpoint weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
    --output-dir outputs/eval_diffusion_only --device cuda:0
```

### Ada-BRIDGER 完整推理（Source + Scheduler + Refiner）

```bash
# 先加载 Ada-BRIDGER checkpoint（包含 scheduler），
# 再覆盖 source 和 refine 路径
dp-eval-ada-bridger \
    --checkpoint outputs/ada_bridger/checkpoints/latest.ckpt \
    --source_checkpoint weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
    --refine_checkpoint weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
    --n_episodes 50 --device cuda:0 --output-dir outputs/eval_ada
```

### 手动指定精炼步数 Benchmark

```bash
dp-benchmark-ddim \
    --checkpoint weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
    --ddim_steps 1 2 5 10 \
    --n-warmup 10 --n-runs 100
```

---

## 方式 4：直接 Run 完整 Workspace

如果有一个包含所有模型路径的 YAML config：

```bash
dp-train \
    experiment=ada_bridger_pusht_anti_collapse \
    task=pusht_lowdim \
    source_policy_checkpoint=weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
    refinement_policy_checkpoint=weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
    scheduler.pretrain_checkpoint=data/checkpoints/scheduler_pusht.pt
```

---

## 推理路径总结

```
                  ┌─────────────────────────────┐
                  │   PushTKeypointsEnv          │
                  │   obs: [keypoints, mask]     │
                  └──────────────┬──────────────┘
                                 │ obs[:, :obs_dim]  ← 去掉 mask
                                 ▼
                  ┌─────────────────────────────┐
                  │   Source Policy (VAE)        │
                  │   a_init = μ(O_t) + σ·ε      │
                  │   shape: [1, 16, 2]          │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │   Scheduler πθ(O_t, a_init) │
                  │   → refinement_steps ∈       │
                  │     {0, 1, 2, 5}            │
                  └──────────────┬──────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            refinement_steps=0       refinement_steps>0
            ┌──────────┐             ┌──────────────────┐
            │ 直接执行  │             │ DDIM Warm-Start  │
            │ a_init   │             │ Refine           │
            └────┬─────┘             │ a_init → a_final │
                 │                   └────────┬─────────┘
                 │                            │
                 └────────────┬───────────────┘
                              ▼
                  ┌─────────────────────────────┐
                  │  a_final[:, :8, :]           │
                  │  执行 n_action_steps=8 步    │
                  └─────────────────────────────┘
```

## 注意事项

1. **obs 截断**：PushT env 返回的 obs shape 是 `[1, obs_dim*2]`（keypoints + visibility mask），推理时只取前半 `[..., :obs_dim]`。
2. **Normalizer 共享**：VAE source policy 的 normalizer 可能与 diffusion refiner 不共享。如果 source 没有 normalizer，从 refiner 借，因为两者使用同一套归一化统计量。
3. **Freeze backbone**：D3RL 推理时 source 和 refiner 都应冻结（`requires_grad=False`），只有 scheduler 可训练。
4. **Deterministic scheduler**：推理时 `scheduler_deterministic=True`（argmax），训练时 `False`（采样探索）。
