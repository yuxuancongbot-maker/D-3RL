# D3RL Four-Stage Training Pipeline

## Stage 1 — Base Model Pretraining

Train or load two frozen primitives on offline demonstrations.

### What

```text
System 1: Conditional VAE         → behavior cloning
System 2: Diffusion Refiner       → standard diffusion policy training
```

### Outputs

```text
source_policy_checkpoint
refinement_policy_checkpoint
```

### Post-training

Both models are frozen. Only the scheduler is trained in later stages.

---

## Stage 2 — Discrepancy-Aware Oracle Label Collection

Generate binary oracle labels by comparing the source action with a
strongly-refined action.

### Procedure

For each rollout state `s_t`:

1. Get System 1 action: `a^(0) = a_init`
2. Get strongest warm-start refined action: `a^(5) = DDIMWarmStartRefine(a_init, O_t, refinement_steps=5)`
3. Compute action discrepancy:

```
δ_t = ||a_t^(0) - a_t^(5)||_1 / D_a
```

4. Generate binary oracle label:

```python
if δ_t < median(δ) * η:
    y_t = 0   # System 1 action is sufficient
else:
    y_t = 1   # needs DDIM refinement
```

Default: `η = 0.5`

### Oracle Dataset Format

```python
{
    "obs":              Tensor[N, T_o, D_o],
    "a_init":           Tensor[N, T_a, D_a],
    "a_strong_refine":  Tensor[N, T_a, D_a],  # 5-step warm-start refinement
    "discrepancy":      Tensor[N],
    "label":            Tensor[N],             # 0 or 1
}
```

### CLI

```bash
dp-collect-oracle \
    --source-checkpoint weights/source_vae.ckpt \
    --refinement-checkpoint weights/diffusion.ckpt \
    --task pusht_lowdim \
    --num-episodes 200 \
    --eta 0.5 \
    --output data/d3rl_oracle/pusht.pt
```

---

## Stage 3 — Scheduler Pretraining

Train the scheduler with conservative soft targets derived from binary oracle
labels.

### Architecture

The scheduler outputs a 4-class distribution over `refinement_steps ∈ {0, 1, 2, 5}`,
but oracle labels are binary. We use conservative soft targets:

```python
if y == 0:   # System 1 sufficient
    target = [1.0, 0.0, 0.0, 0.0]   # steps = [0, 1, 2, 5]

if y == 1:   # needs refinement
    target = [0.0, 0.2, 0.3, 0.5]   # conservative bias toward stronger refinement
```

### Loss

```
KLDivLoss(log_softmax(logits), target_distribution)
```

### Purpose

- Learn a complexity prior.
- Prevent Stage 4 RL cold-start collapse.
- Initialize simple states toward `0` steps.
- Initialize complex states toward non-zero refinement, conservatively biased
  toward `5` steps.

### CLI

```bash
dp-pretrain-scheduler \
    --oracle-dataset data/d3rl_oracle/pusht.pt \
    --obs-dim 2 --action-dim 2 --horizon 16 --n-obs-steps 2 \
    --num-epochs 100 --batch-size 256 --lr 3e-4 \
    --output data/checkpoints/scheduler_pretrained.pt
```

---

## Stage 4 — Lightweight PPO Fine-Tuning

Online RL fine-tuning of the scheduler. This is the main simplification versus
the original D3RL proposal.

### What's Different from the Original

| Original | Current |
|----------|---------|
| PPO + PEGrad multi-objective RL | Standard clipped PPO (lightweight) |
| Train scheduler from scratch | Initialize from Stage 3 checkpoint |
| No KL regularization | KL to frozen pretrained scheduler |
| Constant cost coefficient | Cost warmup schedule |
| No safety mechanism | Success guard with auto-rollback |

### Trainable Parameters

Only the **Dynamic Scheduler πθ**. System 1 and System 2 are frozen.

### Reward

```
r_t = r_task_t - λ_cost * cost(refinement_steps_t)

cost(k) = k / max_refinement_steps

cost(0) = 0/5 = 0.0
cost(1) = 1/5 = 0.2
cost(2) = 2/5 = 0.4
cost(5) = 5/5 = 1.0
```

Task reward uses episode-level reward with discount:

```
r_task_t = γ^(T-1-t) * episode_reward
```

### Cost Warmup

```python
if epoch < task_only_epochs:
    λ_cost = 0.0         # no penalty
else:
    progress = (epoch - task_only_epochs) / cost_warmup_epochs
    λ_cost = cost_coef_target * min(1.0, progress)
```

Config:

```yaml
stage4:
  reward:
    cost_coef_target: 0.05
    task_only_epochs: 10
    cost_warmup_epochs: 20
```

### KL Regularization

Maintain a frozen copy of the Stage 3 scheduler `π_pre`. During PPO:

```
L_total = L_PPO + c_v * L_value - c_ent * H(πθ) + c_kl * KL(πθ || πpre)
```

Config:

```yaml
stage4:
  regularization:
    kl_to_pretrained: true
    kl_coef: 0.05
```

### PPO Details

Each transition:

```python
{
    "obs": O_t,
    "a_init": a_init,
    "action_idx": class index in [0, 1, 2, 3],
    "refinement_steps": refinement_steps,
    "old_log_prob": log π_old(refinement_steps | O_t, a_init),
    "value": V_old(O_t, a_init),
    "reward": r_t,
    "done": done,
}
```

Standard clipped PPO:

```
ratio = exp(log_prob_new - log_prob_old)
L_policy = -mean(min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t))
L_value = MSE(Vθ(s_t), R_t)
```

Advantage: `A_t = R_t - Vθ(s_t)`, then normalize.

### Success Guard

Every `eval_every` epochs, evaluate. Track:

- `best_success_checkpoint`
- `best_tradeoff_checkpoint`

If `eval_success < pretrain_success - success_tolerance`:

- Rollback to best checkpoint
- Reduce λ_cost
- Or early stop

Config:

```yaml
stage4:
  guard:
    enable: true
    eval_every: 5
    success_tolerance: 0.05
    rollback_on_drop: true
```

### CLI

```bash
dp-train \
    experiment=ada_bridger \
    task=pusht_lowdim \
    scheduler.pretrain_checkpoint=data/checkpoints/scheduler_pretrained.pt \
    source_policy_checkpoint=weights/source_vae.ckpt \
    refinement_policy_checkpoint=weights/diffusion.ckpt
```

### Modes

```yaml
stage4:
  mode: lightweight_ppo   # default: standard PPO, no PEGrad
  # mode: ppo_pegrad      # future: PPO + PEGrad multi-objective
```

---

## Training Flow Diagram

```text
Offline Demos
    │
    ▼
Stage 1 ──────────────────────────────────────┐
  Train VAE + Diffusion                        │
  → source_policy.ckpt                         │
  → refinement_policy.ckpt                     │
    │                                          │
    ▼                                          │
Stage 2 ──────────────────────────────────────┤
  Collect oracle labels                        │
  (compare a_init vs a_strong_refine)          │
  → oracle_dataset.pt                          │
    │                                          │
    ▼                                          │
Stage 3 ──────────────────────────────────────┤
  Pretrain scheduler with soft targets         │
  → scheduler_pretrained.ckpt                  │
    │                                          │
    ▼                                          │
Stage 4 ──────────────────────────────────────┘
  Lightweight PPO fine-tuning          (uses all above)
  + cost warmup
  + KL to pretrained
  + success guard
  → final_scheduler.ckpt
```

## Critical Contract Checks

The workspace should verify consistency before training:

1. **Task consistency**: `task.name` matches across source, refinement, and
   scheduler configs.
2. **Shape consistency**: `obs_dim`, `action_dim`, `horizon`, `n_obs_steps`,
   `n_action_steps`.
3. **Action convention**: `abs_action` / `relative_action`, normalizer shapes,
   action space range.
4. **Normalizer consistency**: source and refinement normalizer action stats
   must be compatible. `a_init` is a physical action tensor; if normalizers
   differ, the warm-start point is wrong.
