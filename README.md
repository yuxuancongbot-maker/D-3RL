# D3RL Diffusion Policy

This repository is a cleaned and modularized sibling implementation of the original Diffusion Policy + D3RL research codebase.

Goals:

- preserve the original training and inference workflows;
- keep checkpoint compatibility with the legacy `diffusion_policy` package path where possible;
- provide a cleaner `src/` layout, CLI entry points, config groups, and tests;
- make the D3RL pipeline explicit:
  - System 1: Conditional VAE source policy;
  - System 2: DDIM warm-start diffusion refiner;
  - Scheduler: predicts `refinement_steps ∈ {0, 2, 5, 10}`;
  - Stage 4: lightweight PPO fine-tuning with KL regularization, cost warmup, and success guards.

## Status

This repository is being rebuilt in phases. See:

- `docs/architecture.md`
- `docs/migration.md`
- `docs/d3rl_pipeline.md`
- `docs/commands.md`

## Quick smoke check

```bash
python -m pip install -e ".[dev]"
python -c "import diffusion_policy"
dp-train --help
dp-eval --help
```
