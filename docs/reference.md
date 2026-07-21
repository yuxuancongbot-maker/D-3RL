# D3RL 项目参考信息

更新日期：2026-07-15

## Git 状态

当前分支: `integrate-d3rl-single-repo`

最近提交:
```
3e4890b chore: report D3RL internal timing breakdown
64b740c fix: support image diffusion policies as D3RL source
e9f9542 feat: adapt Ada-BRIDGER to image observations
8c3c904 checkpoint: save D3RL adaptation work
```

未提交文件:
```
eval_toolhang_baseline.py   # 评估脚本，非主线
```

---

## 权重复制: 已整理的路径

### Image Source / Predictor (System 1)

| 任务 | 路径 |
|------|------|
| PushT | `weights/image/predictor/pusht/checkpoints/latest.ckpt` |
| Can | `weights/image/predictor/can/checkpoints/latest.ckpt` |
| Lift | `weights/image/predictor/lift/checkpoints/latest.ckpt` |
| Square | `weights/image/predictor/square/checkpoints/latest.ckpt` |
| Tool Hang | `weights/image/predictor/tool_hang/checkpoints/latest.ckpt` |
| Transport | `weights/image/predictor/transport/checkpoints/latest.ckpt` |

源路径: `/inspire/.../icml_and_iros/diffusion_policy_bridger/weights/image/predictor/<task>/checkpoints/latest.ckpt`
(通过 symlink 连接)

### Image Diffusion / Refiner (System 2)

| 任务 | 路径 |
|------|------|
| PushT | `weights/image/diffusion/pusht/cnn/latest.ckpt` |
| Can | `weights/image/diffusion/robomimic/cnn/can/latest.ckpt` |
| Lift | `weights/image/diffusion/robomimic/cnn/lift/latest.ckpt` |
| Square | `weights/image/diffusion/robomimic/cnn/square/latest.ckpt` |
| Tool Hang | `weights/image/diffusion/robomimic/cnn/tool_hang/latest.ckpt` |
| Transport | `weights/image/diffusion/robomimic/cnn/transport/latest.ckpt` |

Refiner 类型: `DiffusionUnetHybridImagePolicy` (不含 warm-start 逻辑，需通过 `_sdedit_refine_image` 手动构造)

### Lowdim VAE Source (System 1)

| 任务 | 路径 |
|------|------|
| PushT | `weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt` |
| Can | `weights/lowdim/predictor/vae/vae_can_lowdim/checkpoints/latest.ckpt` |
| Lift | `weights/lowdim/predictor/vae/vae_lift_lowdim/checkpoints/latest.ckpt` |
| Square | `weights/lowdim/predictor/vae/vae_square_lowdim/checkpoints/latest.ckpt` |
| Tool Hang | `weights/lowdim/predictor/vae/vae_tool_hang_lowdim/checkpoints/latest.ckpt` |
| Transport | `weights/lowdim/predictor/vae/vae_transport_lowdim/checkpoints/latest.ckpt` |

### Lowdim Diffusion / Refiner (System 2)

| 任务 | 路径 |
|------|------|
| PushT | `weights/lowdim/diffusion/pusht/cnn/latest.ckpt` |
| Can | `weights/lowdim/diffusion/robomimic/cnn/can/epoch=0350-test_mean_score=1.000.ckpt` |
| Lift | `weights/lowdim/diffusion/robomimic/cnn/lift/epoch=0450-test_mean_score=1.000.ckpt` |
| Square | `weights/lowdim/diffusion/robomimic/cnn/square/epoch=1750-test_mean_score=1.000.ckpt` |
| Tool Hang | `weights/lowdim/diffusion/robomimic/cnn/tool_hang/epoch=0850-test_mean_score=0.818.ckpt` |
| Transport | `weights/lowdim/diffusion/robomimic/cnn/transport/latest.ckpt` |

### Image Scheduler

暂未训练。需要先完成 Stage 2 (oracle collection) 和 Stage 3 (pretrain)。

---

## 正在训练: Image Predictor PushT

```bash
# 启动命令
PYTHONPATH=src python -m diffusion_policy.cli.train \
  --config-dir configs \
  --output-dir data/outputs/train_image_predictor_pusht \
  experiment=action_predictor_image task=pusht_image \
  training.num_epochs=300 training.debug=false \
  training.max_train_steps=null training.max_val_steps=null \
  training.checkpoint_every=50 training.val_every=20 \
  training.lr_warmup_steps=500 training.lr_scheduler=cosine training.seed=42 \
  checkpoint=action_predictor logging.mode=online
```

输出目录: `data/outputs/train_image_predictor_pusht/`

当前状态:
- Epoch 49/300, val_loss ≈ 0.0067
- 预计完成 ~60h
- checkpoint 每 50 epoch 保存

旧 predictor 对比:
- 只训练了 37 epoch, val_loss ≈ 0.007
- 50-episode fixed-step 评估: success=0%, max reward≈0.47

---

## Image 适配: 核心改动文件

| 文件 | 改动内容 |
|------|---------|
| `src/diffusion_policy/policy/ada_bridger_policy.py` | image obs 输入、`_get_scheduler_obs`、`_sdedit_refine_image`、BRIDGER feedback |
| `src/diffusion_policy/policy/action_predictor_image_policy.py` | `encode_obs()`、`obs_feat` 输出、robust obs input |
| `src/diffusion_policy/policy/action_predictor_image_vae_policy.py` | `encode_obs()`、`obs_feat` 输出、prev_action feedback |
| `src/diffusion_policy/model/ada_bridger/ada_scheduler.py` | `AdaSchedulerForImages` wrapper |
| `src/diffusion_policy/common/obs_utils.py` | 通用 obs dict 处理工具 (新增) |
| `src/diffusion_policy/workspace/train_ada_bridger_workspace.py` | image scheduler 实例化、rollout dict obs、PPO update dict obs |
| `src/diffusion_policy/cli/infer_d3rl.py` | image env、obs-mode、internal timing |
| `src/diffusion_policy/workspace/train_action_predictor_image_workspace.py` | wandb fix |
| `configs/experiment/ada_bridger_image.yaml` | image Stage 4 配置模板 (新增) |
| `configs/checkpoint/action_predictor.yaml` | predictor 专用 checkpoint 配置 (新增) |

---

## 核心架构: Image D3RL 数据流

```
raw image obs dict {image, agent_pos, ...}
    │
    ▼
MultiImageObsEncoder → obs_feat [B, T, D]
    │
    ├─→ source policy → init_action [B, H, Da]
    │
    ├─→ AdaScheduler(obs_feat, init_action) → k ∈ {0,1,2,5}
    │
    └─→ refiner (image encoder + diffusion model)
            │
            k=0: init_action 直接输出
            k>0: DDIM warm-start refine
            │
            ▼
        action [B, n_action_steps, Da]
```

Lowdim 路径保持不变: `{"obs": Tensor[B, T, D]}` → 原逻辑不变。

---

## PEGrad / PPO 配置

### 代码位置

- PEGrad 实现: `src/diffusion_policy/model/ada_bridger/pegrad_optimizer.py`
- PPO 训练: `src/diffusion_policy/workspace/train_ada_bridger_workspace.py`

### 两种模式

```python
# stage4.mode = "lightweight_ppo" (默认)
# 标准 clipped PPO，无 gradient projection
self.pegrad_optimizer = PEGradOptimizer(
    task_weight=1.0, cost_weight=0.0,
    project_cost_to_task=False,
)

# stage4.mode = "ppo_pegrad"
# PEGrad 多目标投影
self.pegrad_optimizer = PEGradOptimizer(
    task_weight=cfg.pegrad.task_weight,
    cost_weight=cfg.pegrad.cost_weight,
    project_cost_to_task=cfg.pegrad.project_cost_to_task,
)
```

### 论文对照表

| 方法 | Discrepancy Pretrain | PPO | PEGrad |
|------|:---:|:---:|:---:|
| Supervised-only | ✓ | | |
| PPO from scratch | | ✓ | |
| Vanilla PPO | ✓ | ✓ | |
| PPO + PEGrad | ✓ | ✓ | ✓ |

---

## 关键 Latency 数据 (PushT image, cuda:0)

内部 timing (`AdaBridgerPolicy.get_inference_stats()`):

| k | source (ms) | scheduler (ms) | refine (ms) | total (ms) |
|--:|--:|--:|--:|--:|
| 0 | 3.0 | 0.1 | 0.1 | 3.3 |
| 1 | 2.9 | 0.1 | 11.0 | 14.0 |
| 2 | 3.0 | 0.1 | 18.4 | 21.5 |
| 4 | 3.1 | 0.1 | 32.8 | 36.0 |

外层 `Avg inference` (含 env wrapper, 不可靠): k=0 ~25ms, k=2 ~52ms, k=4 ~65ms。

论文表格应使用 internal total。

---

## 常用命令

### Lowdim D3RL 推理
```bash
PYTHONPATH=src python -m diffusion_policy.cli.infer_d3rl \
  --task pusht \
  --source weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt \
  --refiner weights/lowdim/diffusion/pusht/cnn/latest.ckpt \
  --scheduler <scheduler.pt> --n-episodes 50 --device cuda:0
```

### Image D3RL 推理 (fixed-k)
```bash
PYTHONPATH=src python -m diffusion_policy.cli.infer_d3rl \
  --task pusht_image --obs-mode image \
  --source weights/image/predictor/pusht/checkpoints/latest.ckpt \
  --refiner weights/image/diffusion/pusht/cnn/latest.ckpt \
  --refinement-steps 2 --n-episodes 50 --device cuda:0
```

### Image predictor 训练
```bash
PYTHONPATH=src python -m diffusion_policy.cli.train \
  --config-dir configs \
  --output-dir data/outputs/train_image_predictor_<task> \
  experiment=action_predictor_image task=<task>_image \
  training.num_epochs=300 training.debug=false \
  training.max_train_steps=null training.max_val_steps=null \
  checkpoint=action_predictor
```

### Ada-BRIDGER Stage 4 训练
```bash
PYTHONPATH=src python -m diffusion_policy.cli.train \
  --config-dir configs \
  experiment=ada_bridger task=pusht_lowdim \
  source_policy_checkpoint=... refinement_policy_checkpoint=... \
  scheduler.pretrain_checkpoint=...
```

---

## 已知问题

1. **Image predictor 训练不充分**: 旧 predictor 只 37 epoch，score=0%。新 predictor 训练中 (300 epoch)。
2. **Image refiner 类型**: 是 `DiffusionUnetHybridImagePolicy`，不含原生 warm-start。`_sdedit_refine_image` 手动构造 DDIM warm-start，逻辑与 lowdim 一致。
3. **Source/refiner feature dim 不一致**: PushT image: source=128, refiner=66。scheduler 不能直接复用 source obs_feat，需要 `AdaSchedulerForImages` 自己 encode。
4. **Image scheduler 未训练**: 缺少 Stage 2 oracle collection 和 Stage 3 pretrain。
5. **外层 Avg inference 偏高**: `infer_d3rl.py` 的 outer timing 含 env wrapper 开销，论文应使用 `get_inference_stats()` 的 internal total。
