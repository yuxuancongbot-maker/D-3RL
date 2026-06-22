# Architecture

This repository is rebuilt around the original Diffusion Policy execution model:

```text
CLI -> Hydra config -> Workspace -> Policy -> Dataset / EnvRunner -> Checkpoint / Logs
```

The first migration priority is compatibility. The package name remains `diffusion_policy` so old Hydra targets and checkpoints can be supported through the compatibility layer.

Key layers:

- `workspace/`: high-level experiment lifecycle and checkpoint payload compatibility.
- `training/`: shared training loops, callbacks, validation, rollout, logging.
- `evaluation/`: checkpoint loading, policy selection, env-runner execution, report writing.
- `policy/`: policy interfaces and algorithm-specific policy implementations.
- `d3rl/`: explicit D3RL four-stage pipeline.
- `configs/`: composable Hydra groups.
