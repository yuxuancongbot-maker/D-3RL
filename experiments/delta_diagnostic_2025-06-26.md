# δ (Action Discrepancy) Diagnostic Report

**Date**: 2025-06-26  
**Task**: PushT  
**Models**: VAE source (`vae_pusht_lowdim`) + CNN diffusion refiner (`pusht/cnn`)  
**Script**: `diffusion_policy.cli.diagnose_delta`

---

## What We Tested

Stage 2 用 `δ = ||a_init − a_refined_k5||₁ / D_a` 来判定一个状态"是否需要精炼"——

```
δ 高 → 标签=1 (needs refinement)
δ 低 → 标签=0 (source sufficient)
```

这个假设对不对？我们做了 counterfactual 实验：在 rollout 轨迹上选 N 个状态，从同一状态分别用 k=0 和 k=5 跑到 episode 结束，比最终 reward。

---

## Results

### Q1: 精炼整体有用吗？

| | k=0 (pure VAE) | k=5 (always refine) |
|---|---|---|
| Mean reward | 0.401 ± 0.408 | 0.562 ± 0.399 |
| Success rate | 50.0% | 50.0% |

结论：**整体 inconclusive**。有些状态精炼救活了（Ep 1: 0→0.779），有些精炼反而坏了（Ep 3: 0.789→0.001）。精炼不是无脑好。

### Q2: δ 和 benefit 相关吗？

- **Spearman ρ = 0.048** (p = 0.705)
- **Pearson r = −0.139** (p = 0.272)

结论：**完全无关**。ρ≈0 意味着 δ 高的状态和 benefit 高低之间没有任何统计关联。p=0.71 意味着这个 ρ 值极大概率只是噪声。

### Q3: δ 当分类标签准吗？

δ 说"需要精炼"的状态里，实际 benefit>0 的比例：

| | 预测"需要" | 预测"不需要" |
|---|---|---|
| 实际 benefit>0 | 26 (TP) | 14 (FN) |
| 实际 benefit≤0 | 17 (FP) | 7 (TN) |

- **Accuracy**: 51.6%
- **Precision**: 60.5%  — δ 说"要精炼"，实际有用的概率只有六成
- **Recall**: 65.0%

接近抛硬币。17 个"δ 说需要精炼但实际没用/负作用"的状态，以及 14 个"δ 说不需要但其实精炼能帮"的状态。

---

## Root Cause

δ 测的是**动作变化**（"refiner 能不能产生不同的动作"），但 scheduler 需要的是**任务改善**（"不同的动作是不是更好"）。这两个量不相关。

一个极端例子：某个状态 refiner 输出和 source 完全不同的动作（δ 很高），但指向错误方向——切到 k=5 后 reward 反而下降。δ 把这个状态标为"需要精炼"，错了。

## Why This Matters

整条 D3RL 管线盖在这个假设上：

```
Stage 2: δ → binary label
Stage 3: label → scheduler prior (cross-entropy with soft targets)
Stage 4: scheduler prior → PPO fine-tuning
```

地基歪了。Stage 3 从几乎随机的标签学先验 → 先验不可靠 → Stage 4 用昂贵的 online RL 修正 → 贵 + 塌缩 + 超参敏感。

---

## Impact on Observed Symptoms

| Symptom | Root |
|---|---|
| Stage 4 训练太慢/太贵 | RL 要 unlearn Stage 3 的错误先验，同时还要学习新策略。本来应该"微调"，实际在"从头训"。|
| 容易塌缩到 k=0 | Scheduler 初始化偏置 (bias[0]=3.0) + 不可靠先验 + 稀疏 reward → 早期一旦踩到 k=0 就拿不回来的正反馈。 |
| 超参数极敏感 | Reward 稀疏 + 高方差 + 先验噪声 → λ_cost、entropy_coef、clip_epsilon 任何一个稍微不对就崩。|

---

## Recommended Fix

**Replace Stage 2's δ-based labeling with task-space counterfactual labels.**

不再用 ||a_init − a_refined|| 打标签，改用每个状态分别跑 k=0 和 k=5 到底的 episode success 差。成本是一次性离线采集（比每 epoch rollout 便宜），但标签是真值——直接回答"在这个状态下精炼到底有没有帮助"。

Next: design and benchmark the counterfactual labeling approach.
