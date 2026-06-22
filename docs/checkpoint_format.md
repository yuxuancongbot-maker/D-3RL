# Checkpoint Format

The repository preserves the legacy Diffusion Policy payload contract:

```python
{
    "cfg": cfg,
    "state_dicts": {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    "pickles": {
        "global_step": dill.dumps(global_step),
        "epoch": dill.dumps(epoch),
        "_output_dir": dill.dumps(output_dir),
    },
}
```

The compatibility loader is implemented in:

```text
src/diffusion_policy/evaluation/checkpoint_loader.py
```

Legacy `_target_` strings are resolved through:

```text
src/diffusion_policy/workspace/legacy_aliases.py
```
