#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Ada-BRIDGER Tool-Hang full pipeline (anti-collapse)
# Stages:
#   1) Collect oracle labels
#   2) Pretrain scheduler
#   3) RL stage-1 (high exploration, low efficiency penalty)
#   4) RL stage-2 fine-tune (lower exploration, higher efficiency penalty)
#   5) Evaluate final checkpoint
# ============================================================

# ---------- Paths (aligned with train_ada_bridger_workspace_tool_hang_anti_collapse.yaml) ----------
VAE_CKPT="weights/lowdim/predictor/vae/vae_tool_hang_lowdim/checkpoints/latest.ckpt"
DIFF_CKPT="weights/lowdim/diffusion/robomimic/cnn/tool_hang/epoch=0850-test_mean_score=0.818.ckpt"

ORACLE_DIR="data/oracle_labels/tool_hang_abs_v1"
SCHED_DIR="data/outputs/ada_scheduler_tool_hang"
RUN_DIR="data/outputs/ada_bridger_rl_tool_hang_v1"
EVAL_DIR="data/outputs/eval_ada_bridger_tool_hang_v1"

# ---------- Common ----------
DEVICE="cuda:0"
CONFIG_NAME="train_ada_bridger_workspace_tool_hang_anti_collapse"

# ---------- Stage 1: Oracle labels ----------
python collect_oracle_labels.py \
  --source_checkpoint "${VAE_CKPT}" \
  --refine_checkpoint "${DIFF_CKPT}" \
  --output_dir "${ORACLE_DIR}" \
  --n_episodes 200 \
  --max_steps 700 \
  --threshold 0.1 \
  --device "${DEVICE}"

# ---------- Stage 2: Scheduler supervised pretraining ----------
python pretrain_scheduler.py \
  --oracle_data "${ORACLE_DIR}/oracle_labels.pt" \
  --source_ckpt "${VAE_CKPT}" \
  --refine_ckpt "${DIFF_CKPT}" \
  --output_dir "${SCHED_DIR}" \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-3 \
  --val_split 0.1 \
  --device "${DEVICE}"

# ---------- Stage 3: RL stage-1 (anti-collapse) ----------
python train.py \
  --config-name="${CONFIG_NAME}" \
  training.device="${DEVICE}" \
  training.resume=False \
  source_policy_checkpoint="${VAE_CKPT}" \
  refinement_policy_checkpoint="${DIFF_CKPT}" \
  scheduler.pretrain_checkpoint="${SCHED_DIR}/scheduler_best.pt" \
  ppo.entropy_coef=0.15 \
  ppo.efficiency_coef=0.001 \
  hydra.run.dir="${RUN_DIR}"

# ---------- Stage 4: RL stage-2 fine-tune (efficiency recovery) ----------
python train.py \
  --config-name="${CONFIG_NAME}" \
  training.device="${DEVICE}" \
  training.resume=True \
  source_policy_checkpoint="${VAE_CKPT}" \
  refinement_policy_checkpoint="${DIFF_CKPT}" \
  scheduler.pretrain_checkpoint="${SCHED_DIR}/scheduler_best.pt" \
  ppo.entropy_coef=0.10 \
  ppo.efficiency_coef=0.006 \
  hydra.run.dir="${RUN_DIR}"

# ---------- Stage 5: Evaluate ----------
python eval_ada_bridger.py \
  --checkpoint "${RUN_DIR}/checkpoints/latest.ckpt" \
  --source_checkpoint "${VAE_CKPT}" \
  --refine_checkpoint "${DIFF_CKPT}" \
  --output_dir "${EVAL_DIR}" \
  --n_episodes 50 \
  --device "${DEVICE}"

echo "[DONE] Tool-Hang anti-collapse full pipeline finished."
