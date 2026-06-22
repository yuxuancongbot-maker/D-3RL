# D3RL Pipeline

The main D3RL implementation uses the simplified design confirmed during planning.

## System 1

Conditional VAE source policy:

```text
O_t -> a_init
```

## System 2

DDIM warm-start refinement initialized from `a_init`:

```text
refinement_steps ∈ {0, 2, 5, 10}
```

`0` means execute `a_init` directly.

## Scheduler

The scheduler directly predicts actual `refinement_steps`; there is no `k -> DDIM steps` mapping.

## Stage 3 targets

```text
y=0 -> [1.0, 0.0, 0.0, 0.0]
y=1 -> [0.0, 0.2, 0.3, 0.5]
```

## Stage 4 reward

```text
r_t = r_task_t - lambda_cost * refinement_steps_t / 10
```
