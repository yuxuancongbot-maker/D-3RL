# Migration Guide

The original repository remains the source of truth during migration. This sibling repository is rebuilt in phases.

## Principles

1. Preserve checkpoint payload compatibility: `cfg`, `state_dicts`, `pickles`.
2. Keep the Python package name `diffusion_policy` during migration.
3. Keep legacy Hydra targets through `workspace/legacy_aliases.py`.
4. Move repeated training/evaluation logic into shared modules.
5. Keep D3RL's main path separate from legacy Action Predictor and Ada-BRIDGER experiments.

## First MVP

- New package skeleton and CLI entry points.
- BaseWorkspace/checkpoint compatibility.
- PushT lowdim diffusion debug train/eval.
- Action Predictor VAE lowdim debug train/eval.
- D3RL Stage 1-4 debug pipeline.
