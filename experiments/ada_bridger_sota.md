# AdaBridger 4-way SOTA Reproduction

## Method

- **Architecture**: VAE source policy → AdaScheduler {0,1,2,5} → DDIM warm-start refiner
- **Scheduler init**: K5-bias (starts from DDPM ceiling, learns to reduce k)
- **PPO**: lightweight_ppo mode, entropy_coef=0.05, cost_coef=0 (task-only)
- **Training**: 6 rollouts × 28 envs, 300 epochs, eval every 10 epochs

## PushT Results

| Epoch | Test Score | avg_k |
|-------|-----------|-------|
| 49 | 77.8% (env_runner) / 76.0% (manual) | 3.22 |
| 99 | 63.5% | ~2.0 |
| 149 | 63.4% | ~2.0 |
| 199 | 70.6% | ~2.0 |

**Best**: epoch 49, test=77.8%, avg_k=3.22

Baselines:
- VAE (k=0): 50.0%, 1.4ms
- DDPM (k=5): 80.0% (env_runner) / 86.0% (manual), 35ms

## Can Results

| Epoch | Test Score | avg_k (training) |
|-------|-----------|------------------|
| 49 | 98.0% | ~4.0 |
| 149 | 96.0% | ~3.0 |
| 199 | 94.0% | ~2.5 |
| 249 | 98.0% | ~2.3 |
| 299 | 98.0% | 2.15 |

**Best**: epoch 49, test=98.0%, avg_k=~4.0

Baselines:
- DDPM (k=5): 100.0%

## Lift

Training: `tmux new -s lift "cd d3rl_diffusion_policy && python -m diffusion_policy.cli.train experiment=ada_bridger_lift_4way task=lift_lowdim_abs training=rl_default hydra.run.dir=outputs/train_lift training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/lift.log"`

Status: [ ] Running

## Square

Training: `tmux new -s square "cd d3rl_diffusion_policy && python -m diffusion_policy.cli.train experiment=ada_bridger_square_4way task=square_lowdim_abs training=rl_default hydra.run.dir=outputs/train_square training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/square.log"`

Status: [ ] Running

## Transport

Training: `tmux new -s transport "cd d3rl_diffusion_policy && python -m diffusion_policy.cli.train experiment=ada_bridger_transport_4way task=transport_lowdim_abs training=rl_default hydra.run.dir=outputs/train_transport training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/transport.log"`

Status: [ ] Running

## Tool Hang

Training: `tmux new -s toolhang "cd d3rl_diffusion_policy && python -m diffusion_policy.cli.train experiment=ada_bridger_tool_hang_4way task=tool_hang_lowdim_abs training=rl_default hydra.run.dir=outputs/train_toolhang training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/toolhang.log"`

Status: [ ] Running
