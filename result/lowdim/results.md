# AdaBridger 4-way {0,1,2,5} + K5-bias PPO — Lowdim Results

Success from env_runner (50 test episodes). Latency from single-env 5×10 episodes.

## Final Table

| Task | Success | avg_k | Weighted Latency | DDPM k=5 | Saving |
|------|---------|-------|-----------------|----------|--------|
| PushT | 0.778 | 4.37±0.43 | 33.7±2.9ms | 37.9±1.0ms | 11% |
| Can | 0.980 | 1.87±0.00 | 15.2±0.1ms | 34.7±0.6ms | **56%** |
| Lift | 1.000 | 1.64±0.00 | 13.5±0.1ms | 34.9±0.4ms | **61%** |
| Square | 0.920 | 2.00±0.00 | 16.2±0.1ms | 35.3±0.3ms | **54%** |

## k-Distribution & Per-k Latency

| Task | k=0 | k=1 | k=2 | k=5 |
|------|-----|-----|-----|-----|
| PushT | 6.3% (4ms) | 7.9% (33ms) | 0.1% (103ms) | 85.7% (38ms) |
| Can | 6.8% (2ms) | 4.6% (10ms) | 86.8% (16ms) | 1.8% (35ms) |
| Lift | 23.0% (2ms) | 2.6% (10ms) | 70.2% (16ms) | 4.2% (35ms) |
| Square | - | 24.2% (10ms) | 67.6% (16ms) | 8.2% (35ms) |

## Per-k Latency Reference (single batch, forced-k)

| k | PushT | Can | Lift | Square |
|---|-------|-----|------|--------|
| 0 | 0.9ms | 0.9ms | 0.9ms | 0.9ms |
| 1 | 7.9ms | 8.1ms | 8.7ms | 8.8ms |
| 2 | 14.1ms | 14.1ms | 15.2ms | 15.1ms |
| 5 | 32.2ms | 32.4ms | 34.5ms | 34.5ms |
