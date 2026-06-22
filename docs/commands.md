# Commands

## Smoke

```bash
python -m pip install -e ".[dev]"
python -c "import diffusion_policy"
dp-train --help
dp-eval --help
```

## Planned examples

```bash
dp-train experiment=diffusion_unet_lowdim task=pusht_lowdim training=debug
dp-train experiment=action_predictor_vae_lowdim task=pusht_lowdim training=debug
dp-collect-oracle experiment=d3rl_stage3_collect task=pusht_lowdim training=debug
dp-pretrain-scheduler experiment=d3rl_stage3_scheduler_pretrain task=pusht_lowdim training=debug
dp-train experiment=d3rl_stage4_ppo task=pusht_lowdim training=debug
dp-eval checkpoint=<debug_ckpt> output_dir=<tmp_eval>
```
