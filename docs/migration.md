# Migration Guide

This repository (`d3rl_diffusion_policy`) is the single authoritative codebase
for D3RL / Diffusion Policy research. It no longer depends on external repos.

## Key Changes from the Old Repo

### Structure

| Old | New |
|-----|-----|
| `diffusion_policy/` at repo root | `src/diffusion_policy/` (src layout) |
| `setup.py` / `requirements.txt` | `pyproject.toml` |
| Root-level scripts | `src/diffusion_policy/cli/` + console scripts |
| Config mixed in `diffusion_policy/config/` | `configs/` with Hydra composition groups |
| `step_options` / `ddim_steps` | `refinement_steps` (direct DDIM steps) |
| PEGrad required | `lightweight_ppo` default, PEGrad optional |

### Import Compatibility

```python
# Both work identically — the package name is preserved
import diffusion_policy
from diffusion_policy.policy.ada_bridger_policy import AdaBridgerPolicy
```

### CLI Changes

| Old Script | New Command |
|------------|-------------|
| `python train.py` | `dp-train` |
| `python eval.py` | `dp-eval` |
| `python collect_oracle_labels.py` | `dp-collect-oracle` |
| `python pretrain_scheduler.py` | `dp-pretrain-scheduler` |
| `python eval_ada_bridger.py` | `dp-eval-ada-bridger` |
| `python eval_combined_inference.py` | `dp-eval-combined` |

### Config Changes

Old config path (relative to script):
```bash
python train_ada_bridger_workspace.py task=pusht_lowdim
```

New config path:
```bash
dp-train experiment=ada_bridger task=pusht_lowdim
```

Config files moved from `diffusion_policy/config/task/` to `configs/task/` etc.

### Terminology Changes

| Old | New |
|-----|-----|
| `step_options: [0, 1, 2, 5]` | `refinement_steps: [0, 1, 2, 5]` |
| `ddim_steps: [0, 1, 2, 5]` | *(removed — refinement_steps IS DDIM steps)* |
| `_k_to_ddim` mapping | *(removed — no mapping needed)* |
| `max_refinement_steps: 5` | *(unchanged)* |
| PEGrad required | `stage4.mode: lightweight_ppo` |

### Loading Old Checkpoints

Old checkpoints load without modification. The `AdaBridgerPolicy.load_state_dict`
filters legacy keys. The `AdaScheduler` accepts `step_options` as a deprecated
kwarg.

## Principles

1. Preserve checkpoint payload compatibility: `cfg`, `state_dicts`, `pickles`.
2. Keep the Python package name `diffusion_policy`.
3. No runtime dependency on external repos or `vendor/`.
4. D3RL main path uses VAE source policy, not `prev_action` Transformer.
