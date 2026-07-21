# Transport Image D3RL 推理状态报告

## 当前结果

| 指标 | 值 |
|---|---|
| VAE checkpoint | epoch=0002, val_loss=0.0261 (20-dim rotation_6d, abs_action) |
| DDIM refiner | weights/image/transport/latest.ckpt (DiffusionUnetHybridImagePolicy) |
| k=4 成功率 | 100% (3/3 episodes) |
| k=4 延迟 | ~47ms (~22Hz) |
| k=2 | ❌ 跑不了（2 步 DDIM 去噪不够，MuJoCo 卡死） |

内部耗时分解 (k=4):
- Source VAE: ~2ms
- Scheduler: ~0.1ms
- Refine (DDIM 4步): ~45ms
- Total: ~47ms

---

## 代码改动

### 1. `infer_d3rl.py` — 两个修复

**a) render_obs_key 自动检测** (line 228-233)

`create_image_env()` 里 `RobomimicImageWrapper` 默认用 `render_obs_key='agentview_image'`，但 transport 用的是 `shouldercamera0_image`。改为自动取 shape_meta 里第一个 rgb key。

**b) abs_action 环境适配** (line 226-241)

当 action_dim=20（rotation_6d, abs_action）时：
- 设置 `env_meta['controller_configs']['control_delta'] = False`
- 创建 `RotationTransformer('axis_angle', 'rotation_6d')`
- 挂在 `env._rotation_transformer` 和 `env._abs_action` 上

这样 `run_episode` 里的 `_undo_transform_action` 才能在 `env.step()` 前把 20-dim rotation_6d → 14-dim axis_angle。

### 2. `ada_bridger_policy.py` — 一个修复

**model 调用改成 keyword args** (line 444-449)

`_sdedit_refine_image` 里把：
```python
model_output = model(trajectory, t, local_cond=..., global_cond=...)
```
改为（跟 lowdim 路径一致）：
```python
model_output = model(sample=trajectory, timestep=t, local_cond=..., global_cond=...)
```

原因是 `ConditionalUnet1D.forward()` 的签名是 `forward(self, sample, timestep, ...)`，positional args 在某些情况下映射异常（可能是 CUDA 同步问题），改 keyword args 后正常。

### 3. 不需要的改动（已回滚）

- ~~`_convert_action_rot_format()` — axis_angle ↔ rotation_6d 转换~~ — 需要这个是因为 14-dim VAE + 20-dim refiner 不兼容，重训 VAE 后不需要了
- ~~`min_warm_start_steps` 相关改动~~ — git checkout 恢复原始代码

---

## 模型兼容性

| | Image VAE (source) | Image Refiner (DDIM) | 兼容？ |
|---|---|---|---|
| action_dim | 20 | 20 | ✅ |
| 旋转表示 | rotation_6d | rotation_6d | ✅ |
| 动作语义 | absolute | absolute | ✅ |
| abs_action | True | True | ✅ |

之前旧的 image VAE 是 14-dim delta axis_angle，跟 20-dim absolute rotation_6d refiner **不兼容**，需要重训 VAE。

---

## 训练

**训练命令**：
```bash
MUJOCO_GL=osmesa LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 \
python -u -m diffusion_policy.cli.train \
  --output-dir outputs/train_img_transport_abs \
  experiment=action_predictor_vae_image task=transport_image_abs \
  training=vae_image checkpoint=vae_val \
  '++n_obs_steps=2' '++dataset_obs_steps=2' +encoder_output_dim=128 \
  training.device=cuda:3 logging.mode=offline
```

**当前状态**：训练被中断（epoch 6, val_loss=0.0179），需要 `training.resume=true` 续训。

**GPU 分配**：
- GPU 0: img_pusht (tmux)
- GPU 1: img_lift (tmux)
- GPU 2: img_can (tmux)
- GPU 3: transport_abs (需要重启)

**Checkpoint 位置**：
- 训练输出：`outputs/train_img_transport_abs/checkpoints/`
- 已拷贝到 weights：`weights/image/predictor/vae_transport/checkpoints/epoch=0002-val_loss=0.0261.ckpt`

---

## 推理命令

```bash
MUJOCO_GL=osmesa LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 \
python -u -m diffusion_policy.cli.infer_d3rl \
    --task transport_image \
    --source weights/image/predictor/vae_transport/checkpoints/latest.ckpt \
    --refiner weights/image/transport/latest.ckpt \
    --refinement-steps 4 \
    --n-episodes 1 \
    --device cuda:0
```

---

## 已知问题

1. **k=2 不能跑**：模型（100 timesteps DDPM 训练）最少需要 k=4 才能稳定去噪。k=2 时 `set_timesteps(2)` → [100, 0]，只有 1 步有效去噪，输出乱掉导致 MuJoCo 卡死。

2. **HDF5 I/O 慢**：`create_image_env` 用 `use_image_obs=True, render_offscreen=False`，每步从 16GB HDF5 文件读图像，连续跑多 episode 会越来越慢。

3. **EP 5 偶发卡死**：seed=100004 的 episode 有时卡在 MuJoCo 里不返回，可能是特定初始状态导致物理模拟发散。

4. **VAE 训得太少**：epoch 2 (val_loss=0.0261) 就能 100% success，说明 refiner 的 DDIM 精炼很强。训到 val_loss < 0.01 应该更好。对比 pusht image VAE 训了 176 epoch。

5. **LD_PRELOAD 必须**：`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8` 必须加，不然 robosuite 初始化 EGL 会失败。
