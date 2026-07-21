# Square image distribution-matched random routing

This note records the first priority causal control for Square image AdaDP/D3RL routing.

## Question

The dynamic Square image scheduler selects `k=0` and `k=5` with the following marginal distribution over the 50-episode evaluation:

```text
P(k=0) = 62.5%
P(k=5) = 37.5%
Avg k  = 1.87
```

A critical reviewer question is whether dynamic routing works because it is state-dependent, or merely because this marginal compute-budget distribution is good. To test this, we evaluate a state-independent random router with the same marginal distribution.

## Setup

Common setup:

```text
Task: square_image
Episodes: 50
Environment seeds: 42-91
Source policy: outputs/train_img_square_abs/checkpoints/best_val.ckpt
Refiner: weights/image/square/latest.ckpt
Dataset: data/robomimic_image/datasets/square/ph/image_abs.hdf5
```

Matched random routing:

```text
Routing: state-independent random
P(k=0) = 0.625
P(k=5) = 0.375
Random routing seed: 42
Log: /tmp/eval_square_image_random05_50eps.log
```

Implementation note: `src/diffusion_policy/cli/infer_d3rl.py` now supports:

```bash
--random-routing-steps 0 5 \
--random-routing-probs 0.625 0.375 \
--random-routing-seed 42
```

A 200-episode matched-random run was started but intentionally stopped after the 50-episode result was clearly sufficient for the immediate comparison.

## Results

All rows below use 50 episodes with `seed_start=42`.

| Method | Success | Wilson 95% CI | Mean reward | Avg k | Avg inference | k distribution |
|---|---:|---:|---:|---:|---:|---|
| Source-only / k=0 | 4.00% | [1.1%, 13.5%] | 0.040 | 0.00 | 3.2 ms | k=0: 100.0% |
| Fixed k=1 | 16.00% | [8.3%, 28.5%] | 0.160 | 1.00 | 22.0 ms | k=1: 100.0% |
| Distribution-matched random k={0,5} | 30.00% | [19.1%, 43.8%] | 0.300 | 1.83 | 22.4 ms | k=0: 63.3%, k=5: 36.7% |
| Dynamic PPO+PEGrad | 92.00% | [81.2%, 96.8%] | 0.920 | 1.87 | 23.6 ms | k=0: 62.5%, k=5: 37.5% |
| Fixed k=2 | 96.00% | [86.5%, 98.9%] | 0.960 | 2.00 | 31.1 ms | k=2: 100.0% |

## Interpretation

The matched random router has almost the same compute budget as dynamic routing:

```text
Random matched: Avg k = 1.83, latency = 22.4 ms, k=0/5 = 63.3%/36.7%
Dynamic:        Avg k = 1.87, latency = 23.6 ms, k=0/5 = 62.5%/37.5%
```

However, success differs sharply:

```text
Random matched: 30.0%
Dynamic:        92.0%
```

This directly addresses the marginal-budget confound. The dynamic scheduler's Square image performance is not explained merely by using the same overall distribution over refinement steps. The state-dependent routing decision is essential in this comparison.

## Safe paper wording

A conservative statement supported by this result is:

> On Square with image observations, dynamic D3RL achieves 92% success over 50 episodes at 23.6 ms. A state-independent random router matched to the same marginal refinement-step distribution achieves only 30% success at a similar 22.4 ms latency, indicating that the benefit is not explained by the marginal compute-budget distribution alone.

Do not claim broad image generalization from this single task. Also avoid saying dynamic strictly outperforms all fixed-step baselines, since fixed k=2 has slightly higher success at higher latency.

## Reproduce

Matched random routing command:

```bash
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export MUJOCO_GL=osmesa
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8

python -u -m diffusion_policy.cli.infer_d3rl \
  --task square_image \
  --source outputs/train_img_square_abs/checkpoints/best_val.ckpt \
  --refiner weights/image/square/latest.ckpt \
  --random-routing-steps 0 5 \
  --random-routing-probs 0.625 0.375 \
  --random-routing-seed 42 \
  --image-dataset data/robomimic_image/datasets/square/ph/image_abs.hdf5 \
  --n-episodes 50 \
  --seed-start 42 \
  --device cuda:0
```
