# Current evidence assessment for lowdim PEGrad/PPO, image, and Orin results

This note summarizes the current state of evidence after the recent lowdim PPO evals, image PEGrad evals, and Orin latency discussion.

## 1. Lowdim PPO vs PEGrad macro-average should be updated

The previous estimate was directionally useful, but it needs to be corrected using the actual 50-episode lowdim PPO evals for Transport and ToolHang.

### PPO+PEGrad lowdim mean/std

From:

```text
result/lowdim_pegrad_meanstd/combined_results.md
```

| Task | Success | Latency | Avg NFE |
|---|---:|---:|---:|
| PushT | 64.0 | 11.4 | 1.17 |
| Can | 98.3 ± 1.2 | 22.5 ± 0.1 | 2.10 ± 0.01 |
| Lift | 100.0 | 14.2 | 1.60 |
| Square | 87.3 | 12.9 | 1.37 |
| Transport | 78.7 | 19.4 | 2.19 |
| ToolHang | 42.7 | 20.9 | 2.54 |

Macro-average:

```text
Success ≈ 78.5%
Latency ≈ 16.9 ms
Avg NFE ≈ 1.83
```

### Lightweight PPO lowdim

Using the existing table for PushT/Can/Lift/Square and the newly run 50-episode evals for Transport/ToolHang:

| Task | Success | Latency | Avg NFE |
|---|---:|---:|---:|
| PushT | 77.8 | 33.7 | 4.37 |
| Can | 98.0 | 15.2 | 1.87 |
| Lift | 100.0 | 13.5 | 1.64 |
| Square | 92.0 | 16.2 | 2.00 |
| Transport | 66.0 | 22.0 | 1.91 |
| ToolHang | 36.0 | 26.4 | 2.56 |

Macro-average:

```text
Success ≈ 78.3%
Latency ≈ 21.2 ms
Avg NFE ≈ 2.39
```

So the corrected aggregate comparison is closer to:

```text
PPO+PEGrad      78.5%  16.9 ms  1.83
Lightweight PPO 78.3%  21.2 ms  2.39
```

## 2. Corrected conclusion: PEGrad mainly improves efficiency, not aggregate success

The earlier statement that PEGrad improves success by about 2 percentage points is not supported by the current actual evals.

A safer summary is:

```text
Success: roughly unchanged, +0.2 percentage point
Avg NFE: reduced by about 23%
Latency: reduced by about 20%
```

Recommended paper wording:

> PPO+PEGrad preserves the average success rate of lightweight PPO while substantially reducing the average refinement cost and latency. The benefit is most pronounced on more challenging tasks such as Transport and ToolHang.

Avoid claiming that PEGrad is better on every task.

## 3. Per-task interpretation

### PushT

```text
PPO:    77.8%, NFE 4.37, latency 33.7 ms
PEGrad: 64.0%, NFE 1.17, latency 11.4 ms
```

PPO has higher success, while PEGrad is much faster and uses far fewer refinement steps. This is a trade-off, not a clear PEGrad win.

### Can

```text
PPO:    98.0%, NFE 1.87, latency 15.2 ms
PEGrad: 98.3 ± 1.2%, NFE 2.10 ± 0.01, latency 22.5 ± 0.1 ms
```

PPO is more efficient on Can. Do not claim PEGrad improves Can; the longer 3×100 episode rerun confirms Can success is stable at 98.3 ± 1.2%, but its measured latency is higher in this timing run.

### Lift

```text
PPO:    100.0%, NFE 1.64, latency 13.5 ms
PEGrad: 100.0%, NFE 1.60, latency 14.2 ms
```

Lift is essentially tied. PPO has slightly lower latency, while PEGrad has slightly lower NFE.

### Square

```text
PPO:    92.0%, NFE 2.00, latency 16.2 ms
PEGrad: 87.3%, NFE 1.37, latency 12.9 ms
```

PPO has higher success; PEGrad is more efficient.

### Transport

```text
PPO:    66.0%, NFE 1.91, latency 22.0 ms
PEGrad: 78.7%, NFE 2.19, latency 19.4 ms
```

PEGrad has substantially higher success. NFE is higher, but latency is lower, which should be checked for timing consistency.

### ToolHang

```text
PPO:    36.0%, NFE 2.56, latency 26.4 ms
PEGrad: 42.7%, NFE 2.54, latency 20.9 ms
```

PEGrad is better, but both are weak in absolute success.

## 4. Suggested lowdim ablation wording

A concise and safer formulation:

> Across six low-dimensional tasks, PPO+PEGrad achieves comparable average success to lightweight PPO while reducing average NFE from 2.39 to 1.83 and latency from 21.2 ms to 16.9 ms. Lightweight PPO remains competitive on simpler tasks, whereas PPO+PEGrad provides a better success–efficiency trade-off on more challenging tasks such as Transport and ToolHang.

## 5. Image results are promising but still preliminary

Current Square image results, all using 50 episodes with `seed_start=42`:

| Method | Success | Mean reward | Avg k | Avg inference | Notes |
|---|---:|---:|---:|---:|---|
| Source-only / k=0 | 4.00% | 0.040 | 0.00 | 3.2 ms | Very fast, but fails almost all episodes |
| Fixed k=1 | 16.00% | 0.160 | 1.00 | 22.0 ms | Does not dominate dynamic; success is far lower |
| Matched random k={0,5} | 30.00% | 0.300 | 1.83 | 22.4 ms | State-independent routing; k=0 63.3%, k=5 36.7% |
| Dynamic PPO+PEGrad | 92.00% | 0.920 | 1.87 | 23.6 ms | State-dependent scheduler; k=0 62.5%, k=5 37.5% |
| Fixed k=2 | 96.00% | 0.960 | 2.00 | 31.1 ms | Strong fixed-step baseline near dynamic Avg NFE |

The Square image result supports the claim that the method can extend to image observations, but it does not yet establish broad image generalization. The required fixed-k=1 baseline does not dominate dynamic: fixed k=1 is slightly faster than dynamic, but success is only 16%. Dynamic PPO+PEGrad is currently on the success-latency Pareto frontier: compared with source-only and fixed k=1, it substantially improves success; compared with fixed k=2, it reduces latency by about 24.1% while incurring a 4-point success difference. With 50 episodes, this is stronger evidence than the previous 20-episode check, but still should not be written as a multi-seed mean/std result.

Safe image wording for now:

> On Square with image observations, dynamic D3RL achieves 92% success over 50 episodes at 23.6 ms, reducing latency by 24.1% relative to fixed two-step refinement while incurring a 4-point success difference.

Do not write that dynamic is strictly better than fixed-step. It provides a different success-latency operating point, and the fixed k=1 baseline does not cover that point.

Minimum image baselines for Square:

- Source-only / k=0 — done: 4.00%, 3.2 ms over 50 episodes
- Fixed k=1 — done: 16.00%, 22.0 ms over 50 episodes
- Fixed k=2 — done: 96.00%, 31.1 ms over 50 episodes
- Dynamic PPO+PEGrad — done: 92.00%, avg k 1.87, 23.6 ms over 50 episodes
- Fixed k=5 — optional, useful as a high-budget upper bound

Avg k / Avg NFE must be unified before the table enters the paper. If image `k={0,1,2,5}` denotes actual refinement steps, then Avg k can be reported directly as Avg NFE. If the paper uses a class-to-budget mapping such as `N(k)={0,2,5,10}`, then Avg k and Avg NFE are not interchangeable and the table must use the mapped NFE values.

## 6. Orin latency data is useful, but needs two missing measurements

Existing Orin fixed-step latency:

| Modality | 1 step | 2 steps | 5 steps |
|---|---:|---:|---:|
| Lowdim | 28.0 ms | 53.3 ms | 132.0 ms |
| Image | 91.3 ms | 179.4 ms | 422.0 ms |

This supports the hardware motivation: diffusion steps are expensive on Orin, so adaptive zero/low-refinement paths are meaningful.

Still missing:

1. k=0 source-only + scheduler latency.
2. Full dynamic D3RL actual latency on Orin.

Do not rely only on `Avg NFE × single-step latency`, because dynamic inference has different fixed costs, scheduler overhead, source-policy overhead, and k-distribution effects.

## 7. Data-quality checks needed

### Latency vs Avg NFE non-monotonicity

Some latency values do not follow Avg NFE monotonically, e.g.:

```text
PushT:
PEGrad NFE 1.17, latency 11.4
PPO    NFE 4.37, latency 33.7
```

This case is monotonic after correction, but Transport remains suspicious:

```text
Transport:
PEGrad NFE 2.19, latency 19.4
PPO    NFE 1.91, latency 22.0
```

Need to ensure:

- same hardware
- same warmup exclusion
- same source/refiner checkpoints
- same internal timing path
- latency includes/excludes source and scheduler consistently
- k-to-NFE mapping is consistent

## 8. Discrepancy label validity: positive but noisy proxy

A strict paired PushT lowdim analysis was run to test whether the discrepancy signal is associated with actual refinement utility, rather than only measuring action-space difference.

From:

```text
result/pusht_discrepancy_validity/results.md
```

Protocol:

```text
Task: PushT lowdim
Checkpoint: outputs/train/checkpoints/latest.ckpt
Seeds: 100000-100149
N = 150 matched initial states
Discrepancy = initial-state mean |a_k0 - a_k5|
Gain = fixed k=5 full rollout - fixed k=0 full rollout
```

Main results:

| Metric | Value |
|---|---:|
| Spearman(discrepancy, reward gain) | 0.124 |
| Spearman(discrepancy, success gain) | 0.188 |
| Mean reward gain | 0.290 ± 0.402 |
| Mean success gain | 0.280 ± 0.531 |
| k=0 success | 46.7% |
| k=5 success | 74.7% |

Discrepancy-bin analysis:

| Discrepancy bin | n | Mean discrepancy | Reward gain | Success gain | k=0 success | k=5 success |
|---|---:|---:|---:|---:|---:|---:|
| Low | 50 | 12.2519 | 0.214 | 0.160 | 58.0% | 74.0% |
| Mid | 50 | 22.8411 | 0.306 | 0.300 | 44.0% | 74.0% |
| High | 50 | 42.1695 | 0.351 | 0.380 | 38.0% | 76.0% |

Interpretation:

- Fixed k=5 substantially improves PushT success over k=0 on matched initial states: 74.7% vs 46.7%.
- Higher-discrepancy states show monotonically larger average reward and success gains from refinement.
- The Spearman correlations are positive but weak, so the paper should not claim that discrepancy is a strong predictor.
- Safe wording: discrepancy provides a useful but noisy proxy for refinement utility.

Recommended paper wording:

> To validate the discrepancy signal, we compare fixed k=5 and k=0 rollouts from matched PushT initial states. Higher-discrepancy states receive larger average gains from refinement: the success gain increases from 0.16 in the low-discrepancy tertile to 0.38 in the high-discrepancy tertile. The rank correlations are positive but modest, indicating that discrepancy is a useful but noisy proxy for refinement utility.

## 9. Revised role of discrepancy pretraining

The most defensible claim is not that discrepancy pretraining raises the final asymptotic performance ceiling after sufficient PPO+PEGrad training. Its role is better framed as an optimization and initialization benefit:

- faster convergence to a useful success–efficiency trade-off;
- lower probability of early collapse into a trivial k=0 policy or a high-budget policy;
- reduced PPO training variance under sparse task rewards;
- fewer online environment interactions and lower training cost to reach a target operating region.

Therefore, only comparing the final checkpoint may underestimate the value of discrepancy pretraining. If random initialization and discrepancy pretraining reach similar final performance under the same large training budget, the correct interpretation is that pretraining does not substantially change the fully trained upper bound, not that pretraining is useless.

Recommended ablation metrics:

| Metric | Meaning |
|---|---|
| Success–NFE curve over epochs | How the success–efficiency trade-off evolves during training |
| Epochs to target | Number of training epochs needed to enter a specified operating region |
| Return/NFE AUC | Overall learning efficiency under a fixed training budget |
| Collapse rate | Fraction of seeds that converge to trivial k=0 routing or high-budget routing |
| Final performance | Whether pretraining changes the fully trained performance ceiling |
| Across-seed variance | Whether pretraining improves training stability |

A concrete target operating region for PushT can be:

```text
Success >= 60%, AvgNFE <= 2
```

The key report should be:

> With discrepancy pretraining, the scheduler reaches the target operating region after X epochs, compared with Y epochs from random initialization.

Safe paper wording:

> Discrepancy pretraining provides a useful initialization that improves convergence speed and training stability under sparse task rewards.

Avoid this stronger claim unless the final multi-seed data clearly supports it:

> Discrepancy pretraining is necessary for achieving high final performance.

Minimum-cost experimental protocol:

- Run PushT with three training seeds if budget permits.
- Save or extract success and AvgNFE every 10 epochs.
- Compare both the training trajectory and the final checkpoint.
- The final checkpoint comparison should still use the same training budget and the same evaluation seeds.

The intended evidence chain is:

```text
discrepancy has a weak positive association with refinement utility
→ discrepancy can serve as a heuristic scheduler initialization signal
→ pretraining helps PPO find a useful routing trade-off faster
→ the claim is faster/stabler optimization, not a higher final performance ceiling
```

## 10. What remains for final AAAI evidence

Updated priority order:

1. Finish the without-discrepancy-pretraining control: random scheduler initialization with task+cost PPO+PEGrad, same epochs/settings, same final-checkpoint selection rule, and the same evaluation seeds.
2. Compare the pretraining-vs-from-scratch training trajectories on PushT: Success–NFE curves, epochs-to-target, AUC, collapse behavior, final performance, and seed variance if multiple seeds are available.
3. Add Orin k=0 and full dynamic D3RL latency.
4. Confirm lowdim PEGrad/PPO seeds, episodes, and all standard deviations.
5. Check the Transport latency timing path.
6. Replace the old Push-T three-row ablation table with the six-task PPO/PEGrad table plus the Weighted PPO control.
7. Add scheduler accuracy details: split, F1, class proportions.
8. Add the dataset/demo protocol table.
9. Optional: scheduler architecture ablations and cost-weight sensitivity on PushT or Square image.

Reward sensitivity, Franka, fixed k=5 image, and more image tasks are useful additions but should not block the core evidence.

## 11. Overall judgment

The lowdim strategy ablation is now stronger with the Weighted PPO control and the discrepancy-validity analysis, but the without-discrepancy-pretraining run remains the most important missing control for the discrepancy-aware initialization claim. The main analysis should emphasize convergence speed and stability rather than final-performance superiority.

Image and Orin results are positive but still need key controls. Overall, the current evidence supports about 75% of the final AAAI story: the lowdim core is close, while discrepancy pretraining trajectory analysis, hardware deployment, and statistical protocol details remain the main gaps.
