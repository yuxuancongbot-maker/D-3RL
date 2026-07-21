# 命令速查

```bash
cd /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/icml_and_iros/d3rl_diffusion_policy
```

## 训练

```bash
# ── PEGRAD PPO PushT scheduler ──
tmux new -s pusht_pegrad "python -u -m diffusion_policy.cli.train \
  experiment=ada_bridger_pusht_4way_pegrad task=pusht_lowdim training=rl_default \
  hydra.run.dir=outputs/train_pusht_pegrad task.env_runner.n_train=36 \
  training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 \
  training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/pusht_pegrad.log"

# ── 重训 image VAE（checkpoint bug 已修复，4 块 GPU 并行）──
tmux new -s img_pusht     "python -u -m diffusion_policy.cli.train experiment=action_predictor_vae_image task=pusht_image      training=vae_image checkpoint=vae_val hydra.run.dir=outputs/train_img_pusht     +encoder_output_dim=128 training.device=cuda:0 logging.mode=offline 2>&1 | tee /tmp/img_pusht.log"
tmux new -s img_lift      "python -u -m diffusion_policy.cli.train experiment=action_predictor_vae_image task=lift_image       training=vae_image checkpoint=vae_val hydra.run.dir=outputs/train_img_lift      +encoder_output_dim=128 training.device=cuda:1 logging.mode=offline 2>&1 | tee /tmp/img_lift.log"
tmux new -s img_can       "python -u -m diffusion_policy.cli.train experiment=action_predictor_vae_image task=can_image        training=vae_image checkpoint=vae_val hydra.run.dir=outputs/train_img_can       +encoder_output_dim=128 training.device=cuda:2 logging.mode=offline 2>&1 | tee /tmp/img_can.log"
tmux new -s img_transport "python -u -m diffusion_policy.cli.train experiment=action_predictor_vae_image task=transport_image  training=vae_image checkpoint=vae_val hydra.run.dir=outputs/train_img_transport +encoder_output_dim=128 training.device=cuda:3 logging.mode=offline 2>&1 | tee /tmp/img_transport.log"

# ── 并行训练 robomimic scheduler（5 个任务，维度全部 OK）──
tmux new -s train_lift      "python -u -m diffusion_policy.cli.train experiment=ada_bridger_lift_4way      task=lift_lowdim_abs      training=rl_default hydra.run.dir=outputs/train_lift_v2      training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/train_lift.log"
tmux new -s train_can       "python -u -m diffusion_policy.cli.train experiment=ada_bridger_can_4way       task=can_lowdim_abs       training=rl_default hydra.run.dir=outputs/train_can_v2       training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/train_can.log"
tmux new -s train_square    "python -u -m diffusion_policy.cli.train experiment=ada_bridger_square_4way    task=square_lowdim_abs    training=rl_default hydra.run.dir=outputs/train_square_v2    training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/train_square.log"
tmux new -s train_transport "python -u -m diffusion_policy.cli.train experiment=ada_bridger_transport_4way task=transport_lowdim_abs training=rl_default hydra.run.dir=outputs/train_transport_v2 training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/train_transport.log"
tmux new -s train_toolhang  "python -u -m diffusion_policy.cli.train experiment=ada_bridger_tool_hang_4way  task=tool_hang_lowdim_abs  training=rl_default hydra.run.dir=outputs/train_toolhang_v2  training.num_epochs=300 training.eval_every=10 training.checkpoint_every=50 training.n_rollouts_per_epoch=6 logging.mode=offline 2>&1 | tee /tmp/train_toolhang.log"
```

## 评测 —— 所有 checkpoint 遍历

```bash
# Can
tmux new -s eval_can   "python -u eval_all_tasks.py can   2>&1 | tee /tmp/eval_can.log"
# Square
tmux new -s eval_sq    "python -u eval_all_tasks.py square 2>&1 | tee /tmp/eval_sq.log"
# Transport
tmux new -s eval_tr    "python -u eval_all_tasks.py transport 2>&1 | tee /tmp/eval_tr.log"
# Tool Hang
tmux new -s eval_th    "python -u eval_all_tasks.py tool_hang 2>&1 | tee /tmp/eval_th.log"
# 全部
tmux new -s eval_all   "python -u eval_all_tasks.py all 2>&1 | tee /tmp/eval_all.log"
```

## 评测 —— 单个 checkpoint 3 组 seed（均值方差）

```bash
python -u scripts/can_149_3run.py
python -u scripts/square_49_3run.py
python -u scripts/square_149_3run.py
```

## 查看进度

```bash
tmux attach -t <session>      # Ctrl+B D 退出
grep ep /tmp/eval_<task>.log  # 看已出结果
```

## 结果

| Task | Epoch | Success | avg_k | Latency |
|------|:---:|:---:|:---:|:---:|
| PushT | 399 | 70.0% | 1.10 | 12.9ms |
| Lift | 99 | 100.0% ± 0.0% | 0.91 ± 0.07 | 11.6 ± 0.7ms |
| Can | 49 | 96.0% | 2.82 | 28.7ms |
| Can | 149 | 97.3% ± 1.9% | 2.99 ± 0.06 | 20.8 ± 0.3ms |

## Transport image VAE + DDIM k=4

```bash
MUJOCO_GL=osmesa LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.8 \
python -u -m diffusion_policy.cli.infer_d3rl \
  --task transport_image \
  --source weights/image/predictor/vae_transport/checkpoints/latest.ckpt \
  --refiner weights/image/transport/latest.ckpt \
  --image-dataset data/robomimic_image/datasets/transport/ph/image_abs.hdf5 \
  --refinement-steps 4 \
  --n-episodes 1 \
  --device cuda:0 2>&1 | tee /tmp/transport_image_d3rl_k4.log
```
