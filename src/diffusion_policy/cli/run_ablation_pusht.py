#!/usr/bin/env python
"""
Push-T 消融实验全自动流水线 (TABLE I)

三个变体，全部端到端自动完成：
  full        — 收集 Oracle → 监督预训练 → PPO+PEGrad 微调
  no_pegrad   — 收集 Oracle → 监督预训练 → PPO(naive scalarization) 微调
  no_pretrain — 跳过 Oracle + 预训练 → PPO+PEGrad 微调（随机初始化 scheduler）

用法：
  python run_ablation_pusht.py --variant full
  python run_ablation_pusht.py --variant all
  python run_ablation_pusht.py --variant all --seeds 42 123 456
  python run_ablation_pusht.py --variant all --dry_run   # 只打印命令
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_NAME = "train_ada_bridger_workspace_pusht_anti_collapse"
BASE_OUTPUT = "data/outputs/ablation_pusht"

# ── 预训练模型路径（与 config yaml 保持一致）──
SOURCE_CKPT = "weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt"
REFINE_CKPT = "weights/lowdim/diffusion/pusht/cnn/latest.ckpt"

VARIANTS = {
    "full": {
        "display": "Full D3RL (Ours)",
        "need_pretrain": True,
        "overrides": [],
    },
    "no_pegrad": {
        "display": "w/o PEGrad (naive scalarization)",
        "need_pretrain": True,
        "overrides": [
            "pegrad.project_cost_to_task=False",
        ],
    },
    "no_pretrain": {
        "display": "w/o Discrepancy Pretraining",
        "need_pretrain": False,
        "overrides": [
            "++scheduler.pretrain_checkpoint=null",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
#  Stage 1 — 收集 Oracle Labels
# ═══════════════════════════════════════════════════════════════

def run_collect_oracle(oracle_dir: str, device: str, n_episodes: int,
                       dry_run: bool) -> str:
    """返回 oracle_labels.pt 的路径"""
    oracle_path = os.path.join(oracle_dir, "oracle_labels.pt")

    # 已存在则跳过
    if os.path.exists(oracle_path):
        print(f"  [SKIP] Oracle labels already exist: {oracle_path}")
        return oracle_path

    cmd = [
        sys.executable, "collect_oracle_labels.py",
        "--source_checkpoint", SOURCE_CKPT,
        "--refine_checkpoint", REFINE_CKPT,
        "--output_dir", oracle_dir,
        "--n_episodes", str(n_episodes),
        "--device", device,
    ]
    _run_stage("Stage 1 — Collect Oracle Labels", cmd, dry_run)
    return oracle_path


# ═══════════════════════════════════════════════════════════════
#  Stage 2 — 监督预训练 Scheduler
# ═══════════════════════════════════════════════════════════════

def run_pretrain_scheduler(oracle_path: str, pretrain_dir: str,
                           device: str, pretrain_epochs: int,
                           dry_run: bool) -> str:
    """返回 scheduler_best.pt 的路径"""
    best_path = os.path.join(pretrain_dir, "scheduler_best.pt")

    if os.path.exists(best_path):
        print(f"  [SKIP] Pretrained scheduler already exists: {best_path}")
        return best_path

    cmd = [
        sys.executable, "pretrain_scheduler.py",
        "--oracle_data", oracle_path,
        "--source_ckpt", SOURCE_CKPT,
        "--refine_ckpt", REFINE_CKPT,
        "--output_dir", pretrain_dir,
        "--epochs", str(pretrain_epochs),
        "--device", device,
    ]
    _run_stage("Stage 2 — Supervised Pretrain Scheduler", cmd, dry_run)
    return best_path


# ═══════════════════════════════════════════════════════════════
#  Stage 3 — PPO RL 微调
# ═══════════════════════════════════════════════════════════════

def run_rl_finetune(variant_key: str, seed: int, device: str,
                    num_epochs: int, scheduler_ckpt: str,
                    dry_run: bool) -> str:
    """返回 run 目录路径"""
    variant = VARIANTS[variant_key]
    run_name = f"{variant_key}_seed{seed}"
    out_dir = os.path.join(BASE_OUTPUT, run_name)

    cmd = [
        sys.executable, "train.py",
        f"--config-name={CONFIG_NAME}",
        f"hydra.run.dir={out_dir}",
        f"logging.name={run_name}",
        f"logging.group=ablation_pusht",
        f"training.seed={seed}",
        f"training.device={device}",
    ]
    if num_epochs > 0:
        cmd.append(f"training.num_epochs={num_epochs}")

    # 指向本次流水线产出的 pretrained scheduler
    # 使用 ++ 前缀（add-or-override），兼容 config 中已定义 pretrain_checkpoint 的情况
    if scheduler_ckpt is not None:
        cmd.append(f"++scheduler.pretrain_checkpoint={scheduler_ckpt}")

    cmd += variant["overrides"]

    _run_stage(f"Stage 3 — PPO Fine-Tune [{variant['display']}]", cmd, dry_run)
    return out_dir


# ═══════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════

def _run_stage(stage_name: str, cmd: list, dry_run: bool):
    print(f"\n{'=' * 70}")
    print(f"  {stage_name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("[DRY RUN] Skipping.\n")
        return

    t0 = time.time()
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} FAILED (code {result.returncode}) after {elapsed:.0f}s"
        )
    print(f"  [{stage_name}] done in {elapsed / 60:.1f} min\n")


def run_pipeline(variant_key: str, seed: int, device: str,
                 num_epochs: int, oracle_episodes: int,
                 pretrain_epochs: int, dry_run: bool):
    """运行单个变体的完整流水线"""
    variant = VARIANTS[variant_key]
    run_name = f"{variant_key}_seed{seed}"
    print(f"\n{'#' * 70}")
    print(f"  Pipeline: {variant['display']}  (seed={seed})")
    print(f"{'#' * 70}")

    # Oracle + Pretrain 目录（同 seed 的 full 和 no_pegrad 共享）
    shared_dir = os.path.join(BASE_OUTPUT, f"shared_seed{seed}")
    oracle_dir = os.path.join(shared_dir, "oracle_labels")
    pretrain_dir = os.path.join(shared_dir, "scheduler_pretrain")

    scheduler_ckpt = None

    if variant["need_pretrain"]:
        # Stage 1
        oracle_path = run_collect_oracle(oracle_dir, device, oracle_episodes, dry_run)
        # Stage 2
        scheduler_ckpt = run_pretrain_scheduler(
            oracle_path, pretrain_dir, device, pretrain_epochs, dry_run
        )
    else:
        print("  [SKIP] Stages 1-2 (no_pretrain variant)")

    # Stage 3 — RL
    out_dir = run_rl_finetune(
        variant_key, seed, device, num_epochs, scheduler_ckpt, dry_run
    )

    # 写 meta
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "ablation_meta.json")
    meta = {
        "variant": variant_key,
        "display_name": variant["display"],
        "seed": seed,
        "device": device,
        "scheduler_ckpt": scheduler_ckpt,
        "num_epochs": num_epochs,
        "oracle_episodes": oracle_episodes,
        "pretrain_epochs": pretrain_epochs,
        "started_at": datetime.now().isoformat(),
    }
    if not dry_run:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run Push-T ablation experiments — full pipeline (TABLE I)"
    )
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "full", "no_pegrad", "no_pretrain"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="RL fine-tuning epochs (Stage 3)")
    parser.add_argument("--oracle_episodes", type=int, default=100,
                        help="Episodes for oracle label collection (Stage 1)")
    parser.add_argument("--pretrain_epochs", type=int, default=50,
                        help="Supervised pretrain epochs (Stage 2)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    print(f"\n{'#' * 70}")
    print(f"  Ablation Pipeline")
    print(f"  Variants  : {variants_to_run}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  Stages    : Oracle({args.oracle_episodes}ep) → Pretrain({args.pretrain_epochs}ep) → RL({args.num_epochs}ep)")
    print(f"  Device    : {args.device}")
    print(f"  Output    : {BASE_OUTPUT}/")
    print(f"{'#' * 70}")

    for seed in args.seeds:
        for variant_key in variants_to_run:
            run_pipeline(
                variant_key=variant_key,
                seed=seed,
                device=args.device,
                num_epochs=args.num_epochs,
                oracle_episodes=args.oracle_episodes,
                pretrain_epochs=args.pretrain_epochs,
                dry_run=args.dry_run,
            )

    print(f"\n{'#' * 70}")
    print(f"  All ablation pipelines completed.")
    print(f"  Next: python eval_ablation_table.py --n_episodes 100")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
