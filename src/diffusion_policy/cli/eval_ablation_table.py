#!/usr/bin/env python
"""
消融实验评估 + 输出 TABLE I (Push-T)

自动寻找各变体最佳 checkpoint → 跑 eval → 汇总为论文表格。

用法：
  # 评估所有变体（默认 100 episodes）
  python eval_ablation_table.py

  # 指定 episodes 数、设备
  python eval_ablation_table.py --n_episodes 200 --device cuda:1

  # 只评估某个变体
  python eval_ablation_table.py --variant full

  # 指定目录（多种子取平均）
  python eval_ablation_table.py --ablation_dir data/outputs/ablation_pusht
"""

import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

ABLATION_DIR = "data/outputs/ablation_pusht"

VARIANT_KEYS = ["full", "no_pegrad", "no_pretrain"]
DISPLAY_NAMES = {
    "full": "Full D3RL (Ours)",
    "no_pegrad": "w/o PEGrad (naive scalarization)",
    "no_pretrain": "w/o Discrepancy Pretraining",
}


# ─────────────────── Checkpoint discovery ───────────────────

def find_best_checkpoint(variant_dir: str) -> str:
    """在 variant_dir/checkpoints/ 中寻找最佳 checkpoint"""
    ckpt_dir = os.path.join(variant_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoints dir: {ckpt_dir}")

    # 1) Top-k checkpoint（按 test_mean_score 降序）
    topk = sorted(
        glob.glob(os.path.join(ckpt_dir, "epoch=*-test_mean_score=*.ckpt")),
        key=lambda p: float(p.rsplit("test_mean_score=", 1)[-1].replace(".ckpt", "")),
        reverse=True,
    )
    if topk:
        return topk[0]

    # 2) latest.ckpt
    latest = os.path.join(ckpt_dir, "latest.ckpt")
    if os.path.exists(latest):
        return latest

    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def discover_runs(ablation_dir: str, variant_filter: str = None):
    """
    发现所有 {ablation_dir}/{variant_key}_seed*/  目录。
    返回 {variant_key: [run_dir, ...]}.
    """
    result = defaultdict(list)
    if not os.path.isdir(ablation_dir):
        return result

    for name in sorted(os.listdir(ablation_dir)):
        full = os.path.join(ablation_dir, name)
        if not os.path.isdir(full):
            continue
        # 解析 variant_key（去掉 _seed*）
        for vk in VARIANT_KEYS:
            if name.startswith(vk):
                if variant_filter and vk != variant_filter:
                    continue
                result[vk].append(full)
                break
    return dict(result)


# ─────────────────── Evaluation ───────────────────

def evaluate_checkpoint(ckpt_path: str, n_episodes: int, device: str, output_dir: str):
    """加载 checkpoint 并评估，返回 (eval_log, inference_stats)"""
    from eval_ada_bridger import load_ada_bridger_policy, evaluate_policy

    policy, cfg = load_ada_bridger_policy(ckpt_path, device=device)

    # 准备 normalizer
    src_norm = policy.source_policy.normalizer
    ref_norm = policy.refinement_policy.normalizer
    try:
        _ = src_norm['obs']
        normalizer = src_norm
    except (AttributeError, KeyError):
        normalizer = ref_norm
        policy.source_policy.normalizer = ref_norm
    policy.set_normalizer(normalizer)
    policy.scheduler_deterministic = True

    eval_log, inference_stats = evaluate_policy(
        policy=policy,
        cfg=cfg,
        n_episodes=n_episodes,
        output_dir=output_dir,
        device=device,
    )
    return eval_log, inference_stats


# ─────────────────── Main ───────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation runs and output TABLE I")
    parser.add_argument("--ablation_dir", type=str, default=ABLATION_DIR)
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of eval episodes per checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--variant", type=str, default=None,
                        choices=VARIANT_KEYS,
                        help="Only evaluate this variant")
    args = parser.parse_args()

    runs = discover_runs(args.ablation_dir, args.variant)
    if not runs:
        print(f"[ERROR] No ablation runs found in {args.ablation_dir}/")
        print("  Expected directories like:  full_seed42/  no_pegrad_seed42/  no_pretrain_seed42/")
        sys.exit(1)

    # 评估每个 run
    raw_results = defaultdict(list)  # variant_key -> [{success, nfe}, ...]

    for vk in VARIANT_KEYS:
        if vk not in runs:
            continue
        for run_dir in runs[vk]:
            try:
                ckpt = find_best_checkpoint(run_dir)
            except FileNotFoundError as e:
                print(f"[SKIP] {run_dir}: {e}")
                continue

            run_name = os.path.basename(run_dir)
            eval_dir = os.path.join(args.ablation_dir, f"eval_{run_name}")

            print(f"\n{'=' * 65}")
            print(f"  Variant : {DISPLAY_NAMES.get(vk, vk)}")
            print(f"  Run     : {run_name}")
            print(f"  Ckpt    : {ckpt}")
            print(f"{'=' * 65}")

            eval_log, inference_stats = evaluate_checkpoint(
                ckpt_path=ckpt,
                n_episodes=args.n_episodes,
                device=args.device,
                output_dir=eval_dir,
            )

            success_pct = eval_log.get("test/mean_score", 0.0) * 100
            avg_nfe = inference_stats.get("avg_steps", 0.0)

            raw_results[vk].append({
                "run": run_name,
                "checkpoint": ckpt,
                "success_pct": success_pct,
                "avg_nfe": avg_nfe,
            })

            print(f"  => Success: {success_pct:.1f}%  |  Avg NFE: {avg_nfe:.2f}")

    # ────────────────── 汇总 ──────────────────
    print("\n\n")
    print("=" * 65)
    print("  TABLE I: ABLATION STUDY ON PUSH-T")
    print("=" * 65)

    header = f"{'Variant':<40} {'Success(%)↑':>12} {'Avg NFE↓':>10}"
    print(header)
    print("-" * 65)

    summary = {}
    for vk in VARIANT_KEYS:
        if vk not in raw_results:
            continue
        entries = raw_results[vk]
        successes = [e["success_pct"] for e in entries]
        nfes = [e["avg_nfe"] for e in entries]

        if len(entries) > 1:
            # 多种子：均值 ± 标准差
            s_mean, s_std = np.mean(successes), np.std(successes)
            n_mean, n_std = np.mean(nfes), np.std(nfes)
            s_str = f"{s_mean:.1f}±{s_std:.1f}"
            n_str = f"{n_mean:.1f}±{n_std:.1f}"
        else:
            s_str = f"{successes[0]:.1f}"
            n_str = f"{nfes[0]:.1f}"

        display = DISPLAY_NAMES.get(vk, vk)
        print(f"{display:<40} {s_str:>12} {n_str:>10}")

        summary[vk] = {
            "display_name": display,
            "runs": entries,
            "mean_success": float(np.mean(successes)),
            "mean_nfe": float(np.mean(nfes)),
        }

    print("=" * 65)

    # 输出 LaTeX 片段
    print("\n% LaTeX (copy-paste into paper):")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Variant & Success (\%) $\uparrow$ & Avg NFE $\downarrow$ \\")
    print(r"\midrule")
    for vk in VARIANT_KEYS:
        if vk not in summary:
            continue
        s = summary[vk]
        entries = raw_results[vk]
        if len(entries) > 1:
            successes = [e["success_pct"] for e in entries]
            nfes = [e["avg_nfe"] for e in entries]
            s_str = f"${np.mean(successes):.1f} \\pm {np.std(successes):.1f}$"
            n_str = f"${np.mean(nfes):.1f} \\pm {np.std(nfes):.1f}$"
        else:
            s_str = f"{entries[0]['success_pct']:.1f}"
            n_str = f"{entries[0]['avg_nfe']:.1f}"

        display = s["display_name"]
        if vk == "full":
            display = r"\textbf{" + display + "}"
            s_str = r"\textbf{" + s_str + "}"
        print(f"{display} & {s_str} & {n_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # 保存 JSON
    out_json = os.path.join(args.ablation_dir, "ablation_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
