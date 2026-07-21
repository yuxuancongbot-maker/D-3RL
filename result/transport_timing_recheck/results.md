# Transport timing recheck

- Device: `cuda:2`
- Eval seeds: 100000-100019
- Warmup episodes: 3
- Timing: same script/env/hardware; external wall-clock excludes first decision per episode; internal timing aggregated per episode.

| Method | Success | Avg NFE | External ms | Internal total ms | source | scheduler | refine | k distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ppo_scheduler149 | 65.0% | 1.68 | 16.2 | 15.5 | 1.1 | 1.5 | 12.9 | k=0:24.3%, k=1:15.2%, k=2:50.0%, k=5:10.5% |
| pegrad_ckpt199 | 85.0% | 2.12 | 19.4 | 18.7 | 1.1 | 1.5 | 16.1 | k=0:13.2%, k=1:24.7%, k=2:41.0%, k=5:21.1% |
