# Weighted PPO PushT lowdim evaluation

- Checkpoint: `/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/icml_and_iros/d3rl_diffusion_policy/outputs/train/checkpoints/latest.ckpt`
- Selection: final/latest checkpoint after full cost warmup and success-guard checkpoint save
- Protocol: 3 runs × 50 episodes; seeds [100000, 100050, 100100]

| Method | Task | Success | Latency(ms) | Avg NFE |
|---|---|---:|---:|---:|
| Weighted PPO | PushT lowdim | 73.3±9.0% | 33.0±0.8 | 4.50±0.06 |
