# D3RL Architecture

D3RL is a dual-process policy that dynamically allocates compute: a fast source
policy produces an initial action, and a scheduler decides how many DDIM
refinement steps a diffusion refiner should spend to improve it.

## Overview

```text
Observation O_t
    ↓
System 1 / Source Policy  (Conditional VAE, fast)
    ↓
initial action a_init
    ↓
Dynamic Scheduler πθ(O_t, a_init)
    ↓
select refinement_steps ∈ {0, 1, 2, 5}
    ↓
if refinement_steps = 0:
    a_final = a_init          ← execute source action directly
else:
    a_final = DDIMWarmStartRefine(a_init, O_t, refinement_steps)
    ↓
execute a_final
```

- **System 1** = fast reflex / source policy
- **System 2** = diffusion refiner / deep thought
- **Scheduler** = directly predicts the actual DDIM warm-start refinement steps

The scheduler action **is** the actual compute budget. There is no separate
symbolic budget `k` with a mapping table to DDIM steps.

## Module Definitions

### System 1 — Source Policy

- **Type**: Conditional VAE
- **Input**: short observation history `O_t` shape `[B, T_o, D_o]`
- **Output**: initial action `a_init` shape `[B, T_a, D_a]`

```
a_init = μ_ϕ(O_t) + σ_ϕ(O_t) ⊙ ε
```

**Contracts**:

- System 1 only depends on observation history.
- The D3RL main path does **not** use a `prev_action`-conditioned Transformer.
- Transformer source policies may exist as legacy/ablation code, but they are
  not the D3RL main path.

**Training**: behavior cloning on offline demonstrations. Frozen after training.

### System 2 — Diffusion Refiner

- **Type**: pretrained conditional diffusion policy
- **Input**: `O_t`, `a_init`, `refinement_steps`
- **Output**: `a_refined` / `a_final`

Refines System 1's `a_init` through **deterministic DDIM warm-start
refinement**.

**Training**: standard diffusion policy training on offline demonstrations.
Frozen after training.

### Dynamic Scheduler πθ

- **Input**: `s_t = (O_t, a_init)`
- **Output**: distribution over `refinement_steps ∈ {0, 1, 2, 5}`
- **Architecture**: Action-aware encoder with cross-attention

```
obs encoder:     O_t → h_o
action encoder:  a_init → h_a
cross attention: Q=h_a, K=h_o, V=h_o
fusion:          concat(attended h_a, h_a) → h
actor head:      h → logits over refinement_steps {0, 1, 2, 5}
critic head:     h → Vθ(O_t, a_init)
```

**Contracts**:

- The scheduler does **not** directly output robot actions.
- `0` means execute the System 1 action directly.
- Non-zero values mean invoke System 2 with that many DDIM refinement steps.

## Refinement Step Action Space

```
refinement_steps ∈ {0, 1, 2, 5}
```

| Steps | Meaning |
|-------|---------|
| 0 | No refinement; execute `a_init` directly |
| 1 | 1-step DDIM warm-start refinement (light) |
| 2 | 2-step DDIM warm-start refinement (medium) |
| 5 | 5-step DDIM warm-start refinement (strong) |

Config:

```yaml
scheduler:
  refinement_steps: [0, 1, 2, 5]
  max_refinement_steps: 5
  hidden_dim: 256
  num_layers: 2
```

## DDIM Warm-Start Refinement

When `refinement_steps > 0`, System 2 performs **deterministic DDIM
refinement** initialized from the System 1 action:

```
a_init → deterministic DDIM refinement (refinement_steps steps) → a_refined
```

DDIM is used (not DDPM) because:

- DDPM injects stochastic noise at every reverse step.
- In extremely low-step regimes, this causes mode flickering and trajectory
  instability.
- DDIM gives a deterministic refinement path that is easier to analyze for low
  NFE.

## Four-Stage Training Pipeline

```text
Stage 1: Base Model Pretraining
    Train System 1 (VAE) + System 2 (Diffusion) on offline demos
    → source_policy_checkpoint, refinement_policy_checkpoint

Stage 2: Discrepancy-Aware Oracle Label Collection
    Roll out source policy, compare a_init vs a_strong_refine
    Label each state: 0 (sufficient) or 1 (needs refinement)
    → oracle dataset

Stage 3: Scheduler Pretraining
    Train scheduler with conservative soft targets from oracle labels
    → pretrained_scheduler_checkpoint

Stage 4: Lightweight PPO Fine-Tuning
    Online RL fine-tuning of scheduler with cost warmup,
    KL regularization, and success guard
    → final scheduler checkpoint
```

## Code Module Layout

```text
src/diffusion_policy/
├── model/
│   └── ada_bridger/
│       ├── ada_scheduler.py       ← Dynamic Scheduler πθ
│       └── pegrad_optimizer.py    ← PEGrad (optional ppo_pegrad mode)
├── policy/
│   └── ada_bridger_policy.py      ← D3RL dual-process policy
├── workspace/
│   ├── train_ada_bridger_workspace.py   ← Stage 4 PPO training
│   ├── d3rl_workspace.py                ← D3RL alias
│   └── ...
├── d3rl/
│   ├── scheduler/                 ← Clean scheduler utilities
│   │   ├── model.py
│   │   ├── cost_schedule.py       ← Cost warmup logic
│   │   ├── guards.py              ← Success guard
│   │   └── supervised_trainer.py  ← Stage 3 pretraining
│   ├── refinement/
│   │   └── ddim_warm_start.py     ← DDIM warm-start refinement
│   └── oracle/
│       └── discrepancy.py         ← Discrepancy computation
├── cli/                           ← CLI entry points
└── dataset/
    └── d3rl_oracle_dataset.py     ← Oracle dataset loader

configs/
├── config.yaml                    ← Root Hydra config
├── experiment/                    ← Experiment presets
│   ├── ada_bridger.yaml
│   ├── d3rl_stage4_ppo.yaml
│   └── ...
├── task/                          ← Task configs
└── training/                      ← Training configs
```

## Config Structure

```yaml
name: train_d3rl_workspace
_target_: diffusion_policy.workspace.train_ada_bridger_workspace.TrainAdaBridgerWorkspace

defaults:
  - _self_
  - task: pusht_lowdim

horizon: 16
n_obs_steps: 2
n_action_steps: 8

scheduler:
  refinement_steps: [0, 1, 2, 5]
  max_refinement_steps: 5
  hidden_dim: 256
  num_layers: 2

checkpoints:
  source_policy: path/to/source_vae.ckpt
  refinement_policy: path/to/diffusion.ckpt
  scheduler_pretrained: path/to/stage3_scheduler.ckpt

stage2:
  enable: true
  num_episodes: 200
  eta: 0.5
  strong_refinement_steps: 5

stage3:
  enable: true
  num_epochs: 100
  batch_size: 256
  lr: 3.0e-4
  target_if_sufficient: [1.0, 0.0, 0.0, 0.0]
  target_if_needs_refine: [0.0, 0.2, 0.3, 0.5]

stage4:
  enable: true
  mode: lightweight_ppo       # or ppo_pegrad
  num_epochs: 50
  reward:
    cost_coef_target: 0.05
    task_only_epochs: 10
    cost_warmup_epochs: 20
  regularization:
    kl_to_pretrained: true
    kl_coef: 0.05
  guard:
    enable: true
    eval_every: 5
    success_tolerance: 0.05
    rollback_on_drop: true
```

## Key Design Decisions

1. **No `k → DDIM steps` mapping**: refinement steps directly equal DDIM
   inference steps. Simplifies reward, cost, logging.
2. **Stage 4 default is lightweight PPO**, not PEGrad. PEGrad is an optional
   future mode.
3. **Stage 4 requires Stage 3 initialization**. No cold-start RL.
4. **KL-to-pretrained** prevents RL from drifting too far from
   discrepancy-pretrained behavior.
5. **Cost warmup**: no compute penalty for the first N epochs.
6. **Success guard**: auto-rollback if success drops below tolerance.
