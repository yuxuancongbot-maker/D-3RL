# AdaBridger 4-way {0,1,2,5} + K5-bias PPO Results

## Method
- K5-bias scheduler init (start from DDPM ceiling, learn to reduce)
- lightweight_ppo (task-only, cost_coef=0)
- 36 training seeds, 300 epochs

## Results (env_runner test_mean_score)

| Task | DDPM K5 | Best Test | Epoch | Checkpoints |
|------|:---:|:---:|:---:|------|
| Lift | 1.000 | **1.000** | 49–249 | `outputs/lift/` |
| Can | 1.000 | **0.980** | 49 | `outputs/can/` |
| Square | 1.000 | **0.920** | 49 | `outputs/square/` |
| Transport | ? | **0.820** | 149 | `outputs/train_transport/` |
| Tool Hang | 0.818 | **0.460** | 49 | `outputs/train_toolhang/` |
| PushT | 0.860 | ⬜ running | — | `outputs/train_pusht/` |

## Per-task Checkpoints

### Lift (`outputs/lift/checkpoints/`)
- epoch=0049-test_mean_score=1.000.ckpt
- epoch=0099-test_mean_score=1.000.ckpt
- epoch=0149-test_mean_score=1.000.ckpt
- epoch=0199-test_mean_score=1.000.ckpt
- epoch=0249-test_mean_score=1.000.ckpt

### Can (`outputs/can/checkpoints/`)
- epoch=0049-test_mean_score=0.980.ckpt
- epoch=0149-test_mean_score=0.960.ckpt
- epoch=0199-test_mean_score=0.940.ckpt
- epoch=0249-test_mean_score=0.980.ckpt
- epoch=0299-test_mean_score=0.980.ckpt

### Square (`outputs/square/checkpoints/`)
- epoch=0049-test_mean_score=0.920.ckpt
- epoch=0099-test_mean_score=0.880.ckpt
- epoch=0149-test_mean_score=0.900.ckpt
- epoch=0199-test_mean_score=0.900.ckpt
- epoch=0249-test_mean_score=0.920.ckpt

### Transport (`outputs/train_transport/checkpoints/`)
- epoch=0049-test_mean_score=0.700.ckpt
- epoch=0149-test_mean_score=0.820.ckpt (best)
- epoch=0199-test_mean_score=0.720.ckpt
- epoch=0249-test_mean_score=0.740.ckpt
- epoch=0299-test_mean_score=0.700.ckpt

### Tool Hang (`outputs/train_toolhang/checkpoints/`)
- epoch=0049-test_mean_score=0.460.ckpt (only one — training may have crashed)

## AAAI Revision — Required Experiments

### Already done:
- [x] Task-only PPO (lightweight_ppo, cost_coef=0) — all 6 tasks

### Still needed:
- [ ] PushT re-run
- [ ] Tool Hang full run (or debug the crash)
- [ ] Matched scalarized PPO ablation (cost_coef > 0, no PEGrad) — PushT only
- [ ] PPO + PEGrad ablation (cost_coef > 0, project_cost_to_task=True) — PushT only
- [ ] Supervised-only scheduler baseline
- [ ] PPO from scratch baseline (K0-bias, no pretrain)
- [ ] Scheduler architecture ablation
- [ ] Reward weight sensitivity
- [ ] Orin latency measurements
