# Commands

All commands are installed as console scripts via `pip install -e .`.

## Quick Start

```bash
pip install -e ".[dev]"
python -c "import diffusion_policy; print('OK')"
```

## Training

### Base Diffusion Policy

```bash
dp-train experiment=diffusion_unet_lowdim task=pusht_lowdim training=debug
```

### Action Predictor (VAE / Transformer)

```bash
dp-train experiment=action_predictor_vae_lowdim task=pusht_lowdim training=debug
dp-train experiment=action_predictor_transformer_lowdim task=pusht_lowdim training=debug
```

### Stage 4 — Ada-BRIDGER PPO Fine-Tuning

```bash
dp-train \
    experiment=ada_bridger \
    task=pusht_lowdim \
    source_policy_checkpoint=weights/source_vae.ckpt \
    refinement_policy_checkpoint=weights/diffusion.ckpt \
    scheduler.pretrain_checkpoint=data/checkpoints/scheduler_pretrained.pt

# Anti-collapse configs for specific tasks
dp-train experiment=ada_bridger_pusht_anti_collapse task=pusht_lowdim
dp-train experiment=ada_bridger_can_anti_collapse task=can_lowdim_abs
dp-train experiment=ada_bridger_transport_anti_collapse task=transport_lowdim_abs
dp-train experiment=ada_bridger_tool_hang_anti_collapse task=tool_hang_lowdim_abs
```

## Oracle Label Collection (Stage 2)

```bash
dp-collect-oracle \
    --source-checkpoint weights/source_vae.ckpt \
    --refinement-checkpoint weights/diffusion.ckpt \
    --task pusht_lowdim \
    --num-episodes 200 \
    --eta 0.5 \
    --output data/d3rl_oracle/pusht.pt
```

## Scheduler Pretraining (Stage 3)

```bash
dp-pretrain-scheduler \
    --oracle-dataset data/d3rl_oracle/pusht.pt \
    --obs-dim 2 --action-dim 2 --horizon 16 --n-obs-steps 2 \
    --num-epochs 100 --batch-size 256 --lr 3e-4 \
    --output data/checkpoints/scheduler_pretrained.pt
```

## Evaluation

### Evaluate a Single Checkpoint

```bash
dp-eval \
    --checkpoint outputs/my_run/checkpoints/latest.ckpt \
    --output-dir outputs/eval_result \
    --device cuda:0
```

### Evaluate Ada-BRIDGER Policy

```bash
dp-eval-ada-bridger \
    --checkpoint outputs/ada_bridger/checkpoints/latest.ckpt \
    --task pusht_lowdim \
    --device cuda:0 \
    --output-dir outputs/eval_ada
```

### Evaluate Combined Inference (D3RL)

```bash
dp-eval-combined \
    --source-checkpoint weights/source_vae.ckpt \
    --scheduler-checkpoint outputs/scheduler.ckpt \
    --refinement-checkpoint weights/diffusion.ckpt \
    --task pusht_lowdim \
    --device cuda:0
```

### D3RL Evaluation (alias)

```bash
dp-eval-d3rl
```

## Demo & Visualization

```bash
# Generate Ada-BRIDGER demo video
dp-demo-ada-bridger \
    --checkpoint outputs/ada_bridger/checkpoints/latest.ckpt \
    --task pusht_lowdim \
    --output demo.mp4

# Combined inference demo
dp-demo-combined \
    --source-checkpoint weights/source_vae.ckpt \
    --scheduler-checkpoint outputs/scheduler.ckpt \
    --refinement-checkpoint weights/diffusion.ckpt \
    --task pusht_lowdim \
    --output demo_combined.mp4
```

## PushT D3RL Inference

```bash
# 完整推理（source VAE + scheduler + diffusion refiner）
dp-infer-d3rl --task pusht \
    --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
    --scheduler data/outputs/ada_scheduler_pusht/scheduler_best.pt

# 只用 VAE 不精炼（k=0）
dp-infer-d3rl --task pusht \
    --source ... --refiner ... --refinement-steps 0
```

## Robomimic D3RL Inference

```bash
# can
dp-infer-d3rl --task can \
    --source weights/lowdim/predictor/vae/vae_can_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/robomimic/cnn/can/epoch=0350-test_mean_score=1.000.ckpt \
    --scheduler data/outputs/ada_scheduler_can/scheduler_best.pt

# lift
dp-infer-d3rl --task lift \
    --source weights/lowdim/predictor/vae/vae_lift_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/robomimic/cnn/lift/epoch=0450-test_mean_score=1.000.ckpt
    # (no scheduler available for lift)

# square
dp-infer-d3rl --task square \
    --source weights/lowdim/predictor/vae/vae_square_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/robomimic/cnn/square/epoch=1750-test_mean_score=1.000.ckpt
    # (no scheduler available for square)

# transport
dp-infer-d3rl --task transport \
    --source weights/lowdim/predictor/vae/vae_transport_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/robomimic/cnn/transport/latest.ckpt \
    --scheduler data/outputs/ada_scheduler_transport/scheduler_best.pt

# tool_hang
dp-infer-d3rl --task tool_hang \
    --source weights/lowdim/predictor/vae/vae_tool_hang_lowdim/checkpoints/latest.ckpt \
    --refiner weights/lowdim/diffusion/robomimic/cnn/tool_hang/epoch=0850-test_mean_score=0.818.ckpt \
    --scheduler data/outputs/ada_scheduler_tool_hang/scheduler_best.pt

# 多 episode 评估
dp-infer-d3rl --task can \
    --source ... --refiner ... --scheduler ... \
    --n-episodes 20
```

## Benchmarking

```bash
# Benchmark DDIM inference speed at different step counts
dp-benchmark-ddim \
    --checkpoint weights/diffusion.ckpt \
    --ddim_steps 1 2 3 5 10 \
    --n-warmup 10 --n-runs 100
```

## Metrics & Analysis

```bash
# Compute metrics from eval logs
dp-metrics --eval-dir outputs/eval_result
```

## Ablation

```bash
dp-eval-ablation --results-dir outputs/ablation
```

## Multi-Run (Hydra Sweep)

```bash
dp-multirun experiment=ada_bridger task=pusht_lowdim \
    stage4.reward.cost_coef_target=0.01,0.05,0.1
```

## Common Overrides

Override any config value from the command line:

```bash
# Debug mode
dp-train experiment=ada_bridger task=pusht_lowdim training=debug

# Change device
dp-train ... training.device=cpu

# Disable wandb
dp-train ... logging.mode=disabled

# Resume
dp-train ... training.resume=true

# Longer training
dp-train ... training.num_epochs=500 training.n_rollouts_per_epoch=6
```

## Config Structure

```
experiment=ada_bridger     ← which experiment config to use
task=pusht_lowdim          ← which task/environment
training=debug             ← training hyperparameters
logging=disabled           ← logging config
checkpoint=default         ← checkpoint saving config
```

Config files live in `configs/experiment/`, `configs/task/`, `configs/training/`, etc.
