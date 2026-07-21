# PushT discrepancy label validity

- Checkpoint: `/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/icml_and_iros/d3rl_diffusion_policy/outputs/train/checkpoints/latest.ckpt`
- Protocol: matched initial states, seeds 100000-100149; gain = fixed k=5 rollout minus fixed k=0 rollout.

| Metric | Value |
|---|---:|
| Spearman(discrepancy, reward gain) | 0.124 |
| Spearman(discrepancy, success gain) | 0.188 |
| Mean reward gain | 0.290±0.402 |
| Mean success gain | 0.280±0.531 |
| k=0 success | 46.7% |
| k=5 success | 74.7% |

| Discrepancy bin | n | discrepancy | reward gain | success gain | k=0 succ | k=5 succ |
|---|---:|---:|---:|---:|---:|---:|
| low | 50 | 12.2519 | 0.214 | 0.160 | 58.0% | 74.0% |
| mid | 50 | 22.8411 | 0.306 | 0.300 | 44.0% | 74.0% |
| high | 50 | 42.1695 | 0.351 | 0.380 | 38.0% | 76.0% |
