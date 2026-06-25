# Checkpoint Format

## Payload Contract

Checkpoints follow the Diffusion Policy convention:

```python
{
    "cfg": cfg,                    # OmegaConf / Hydra config
    "state_dicts": {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if use_ema else None,
        "optimizer": optimizer.state_dict(),
    },
    "pickles": {
        "global_step": dill.dumps(global_step),
        "epoch": dill.dumps(epoch),
        "_output_dir": dill.dumps(output_dir),
    },
}
```

## Loading a Checkpoint

```python
import dill
import torch
import hydra

payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
cfg = payload['cfg']
policy = hydra.utils.instantiate(cfg.policy)

# Load weights
if 'model' in payload['state_dicts']:
    policy.load_state_dict(payload['state_dicts']['model'], strict=False)

# Load normalizer
if 'normalizer' in payload['state_dicts']:
    policy.normalizer.load_state_dict(
        payload['state_dicts']['normalizer'], strict=False
    )
```

## Ada-BRIDGER Scheduler Checkpoint

Stage 3 pretraining saves a scheduler-only checkpoint:

```python
{
    "scheduler_state_dict": scheduler.state_dict(),
    "config": {
        "obs_dim": ...,
        "action_dim": ...,
        "horizon": ...,
        "n_obs_steps": ...,
        "refinement_steps": [0, 1, 2, 5],
    },
    "val_acc": float,
    "epoch": int,
}
```

Load a pretrained scheduler into the Stage 4 workspace:

```yaml
scheduler:
  pretrain_checkpoint: "data/checkpoints/scheduler_pretrained.pt"
```

## Backward Compatibility

### Package Name

The Python package name remains `diffusion_policy` so legacy Hydra `_target_`
strings and checkpoints resolve correctly:

```python
import diffusion_policy  # resolves from src/diffusion_policy/
```

### Legacy Aliases

`workspace/legacy_aliases.py` maps old Hydra targets to current classes.

### Old step_options / ddim_steps

The `AdaScheduler` and `AdaBridgerPolicy` accept `step_options` and
`ddim_steps` as deprecated keyword arguments for backward compatibility. They
are automatically converted to `refinement_steps`.

Old checkpoints using the 4-way `step_options=[0,1,2,5]` action space load
directly since the new `refinement_steps=[0,1,2,5]` has the same cardinality
and semantics.

### state_dict Loading

`AdaBridgerPolicy.load_state_dict` filters out known legacy keys
(`_normalizer.*`, `source_policy.*`, `refinement_policy.*`, `_dummy_variable`)
to support checkpoints from different versions.
