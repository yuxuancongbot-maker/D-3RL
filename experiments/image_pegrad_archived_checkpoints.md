# Image PEGrad archived checkpoints

This file records scheduler checkpoints archived for reproducible image PEGrad evaluation.

## Summary

| Task | Archive | Full checkpoint source | Eval episodes | Seed start | Success | Mean reward | Avg k | Avg inference |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Square image | `weights/image/scheduler/square_image_action_diff_pegrad_epoch119_eval90/` | `outputs/train_pegrad_square_image_action_diff_long/checkpoints/epoch=0119-test_mean_score=0.880.ckpt` | 20 | 42 | 90.00% | 0.900 | 1.90 | 24.4 ms |
| Transport image | `weights/image/scheduler/transport_image_action_diff_pegrad_epoch29_eval50/` | `outputs/train_pegrad_transport_image_action_diff_long/checkpoints/epoch=0029-test_mean_score=0.580.ckpt` | 20 | 42 | 50.00% | 0.500 | 2.85 | 40.8 ms |

## Archived contents

Each archive contains:

- `scheduler_best.pt` — exported scheduler-only checkpoint usable by `diffusion_policy.cli.infer_d3rl`
- `full_workspace_*.ckpt` — full PEGrad workspace checkpoint
- `scheduler_pretrain_action_diff.pt` — supervised action-diff scheduler checkpoint used before PEGrad
- `scheduler_labels_action_diff.pt` — action-diff label dataset
- `eval_20eps_seed42.log` — exact 20-episode evaluation log
- `README.md` — reproduction command and notes

## Notes

- Both schedulers use candidate refinement steps `[0, 1, 2, 5]`.
- Square's evaluated policy selected only `k=0` and `k=5`, reflecting the binary nature of the current action-diff label strategy.
- Transport's evaluated policy selected `k=0`, `k=2`, and `k=5`, but achieved only 50% success in this 20-episode eval.
