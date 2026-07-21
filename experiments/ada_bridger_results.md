# AdaBridger 4-way {0,1,2,5} + K5-bias PPO — Full Results

## Method
- K5-bias scheduler init (start from DDPM ceiling, learn to reduce k)
- lightweight_ppo (task-only, cost_coef=0)
- 4-way action space: {0, 1, 2, 5}

## PushT Lowdim (50 seeds, seeds 100000–100049, manual eval)

| Epoch | Success | avg_k | Latency | k=0 | k=1 | k=2 | k=5 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4 | 50.0% | 0.00 | — | 100% | — | — | — |
| 49 | 76.0% | 3.22 | 28ms | 17% | 11% | 15% | 56% |
| 99 | 56.0% | 1.94 | 19ms | 42% | 17% | 10% | 32% |
| 149 | 74.0% | 2.72 | 24ms | 18% | 14% | 28% | 41% |
| 199 | 70.0% | 2.08 | 19ms | 29% | 18% | 25% | 28% |
| 249 | 72.0% | 2.46 | — | 22% | 23% | 17% | 38% |
| **399** | **70.0%** | **1.09** | **13ms** | **59%** | 13% | 15% | 13% |
| 449 | 74.0% | 3.29 | — | 14% | 19% | 7% | 59% |
| 499 | 70.0% | 2.61 | — | 18% | 15% | 30% | 37% |

**Best efficiency**: epoch 399 — 70.0% success @ avg_k=1.09 (59% steps use k=0, only 13% use k=5)

Baselines:
| Method | Success | avg_k |
|--------|:---:|:---:|
| VAE only (k=0) | 50.0% | 0 |
| DDPM k=5 | 86.0% | 5 |

## Robomimic Lowdim (env_runner test_mean_score)

| Task | DDPM k=5 | Best Test | Epoch | Checkpoint |
|------|:---:|:---:|:---:|------|
| Lift | 1.000 | **1.000** | 49–249 | `outputs/lift/checkpoints/` |
| Can | 1.000 | **0.980** | 49 | `outputs/can/checkpoints/` |
| Square | 1.000 | **0.920** | 49 | `outputs/square/checkpoints/` |
| Transport | — | **0.820** | 149 | `outputs/train_transport/checkpoints/` |
| Tool Hang | 0.818 | 0.460 | 49 | `outputs/train_toolhang/checkpoints/` |

## PushT Image (env_runner test_mean_score, 20 seeds)

| Method | Score |
|--------|:---:|
| VAE only (transformer predictor) | 12.7% |
| DDPM 100 steps | 94.1% |
| DDIM 5 steps | 91.4% |
| DDIM 4 steps | 88.8% |
| DDIM 2 steps | 86.6% |
| VAE + DDIM warm-start k=5 | 90.9% |
| VAE + DDIM warm-start k=4 | 91.4% |
| VAE + DDIM warm-start k=2 | 81.9% |

Image VAE (ActionPredictorImageVAEPolicy) training in progress — `tmux attach -t img_vae`
