#!/usr/bin/env python
"""
Ada-BRIDGER 评估脚本

评估自适应精炼扩散策略的:
1. 任务成功率 (Task Success Rate)
2. 平均控制频率 (Control Frequency)
3. 精炼步数分布 (Step Distribution)
4. 推理延迟统计 (Inference Latency)

用法:
    python eval_ada_bridger.py --checkpoint <path_to_checkpoint> --output_dir <output_dir>
"""

import os
import sys
import pathlib
import argparse
import json
import time
import copy
from collections import defaultdict

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
import dill

# 添加项目根目录到路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)


def load_ada_bridger_policy(checkpoint_path: str, device: str = 'cuda:0',
                            source_checkpoint: str = None,
                            refine_checkpoint: str = None):
    """
    加载 Ada-BRIDGER 策略
    
    Args:
        source_checkpoint: 覆盖 cfg 中的 VAE checkpoint 路径（可选）
        refine_checkpoint: 覆盖 cfg 中的 Diffusion checkpoint 路径（可选）
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # dill 不支持 weights_only 参数
    payload = torch.load(
        open(checkpoint_path, 'rb'), 
        pickle_module=dill,
    )
    
    cfg = payload['cfg']
    
    # 允许用户覆盖预训练模型路径（checkpoint 中保存的路径可能与当前环境不同）
    OmegaConf.set_struct(cfg, False)
    if source_checkpoint is not None:
        cfg.source_policy_checkpoint = source_checkpoint
        print(f"  Overriding source checkpoint: {source_checkpoint}")
    if refine_checkpoint is not None:
        cfg.refinement_policy_checkpoint = refine_checkpoint
        print(f"  Overriding refine checkpoint: {refine_checkpoint}")
    
    # 创建 workspace 并加载权重
    from diffusion_policy.workspace.train_ada_bridger_workspace import TrainAdaBridgerWorkspace
    
    workspace = TrainAdaBridgerWorkspace(cfg)
    workspace.load_payload(payload)
    
    policy = workspace.policy
    policy.to(device)
    policy.eval()
    
    return policy, cfg


def evaluate_policy(
    policy,
    cfg,
    n_episodes: int = 50,
    max_steps: int = 300,
    output_dir: str = None,
    device: str = 'cuda:0',
):
    """
    评估策略性能
    """
    # 确保输出目录存在
    output_dir = output_dir or 'data/eval_ada_bridger'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'media'), exist_ok=True)
    
    # 创建环境 runner
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        n_test=n_episodes,
        n_test_vis=min(6, n_episodes),
    )
    
    # 设置 normalizer
    # VAE checkpoint 可能未保存 normalizer（训练时未调用 set_normalizer），
    # 此时从 refinement_policy.normalizer 补填，两者共享同一套归一化统计量
    src_normalizer = policy.source_policy.normalizer
    ref_normalizer = policy.refinement_policy.normalizer
    try:
        # 检查 source_policy normalizer 是否有效
        _ = src_normalizer['obs']
        normalizer = src_normalizer
    except (AttributeError, KeyError):
        print("  [Info] source_policy normalizer is empty, borrowing from refinement_policy")
        normalizer = ref_normalizer
        # 直接拷贝 normalizer 对象（load_state_dict 对空 LinearNormalizer 可能不可靠）
        policy.source_policy.normalizer = copy.deepcopy(normalizer)
    policy.set_normalizer(normalizer)
    
    # 运行评估
    print(f"\nEvaluating on {n_episodes} episodes...")
    policy.reset()
    
    # 设置为确定性模式评估
    policy.scheduler_deterministic = True
    
    eval_log = env_runner.run(policy)
    
    # 获取推理统计
    inference_stats = policy.get_inference_stats()
    
    return eval_log, inference_stats


def print_results(eval_log: dict, inference_stats: dict):
    """
    打印评估结果
    """
    print("\n" + "=" * 60)
    print("Ada-BRIDGER Evaluation Results")
    print("=" * 60)
    
    # 任务性能
    print("\n📊 Task Performance:")
    print(f"  • Test Mean Score: {eval_log.get('test/mean_score', 0):.4f}")
    print(f"  • Train Mean Score: {eval_log.get('train/mean_score', 0):.4f}")
    
    # 效率统计
    print("\n⚡ Efficiency Statistics:")
    avg_steps = inference_stats.get('avg_steps', 0)
    print(f"  • Average Refinement Steps: {avg_steps:.2f}")
    
    if 'avg_total_time' in inference_stats:
        avg_time_ms = inference_stats['avg_total_time'] * 1000
        control_freq = 1000 / avg_time_ms if avg_time_ms > 0 else 0
        print(f"  • Average Inference Time: {avg_time_ms:.2f} ms")
        print(f"  • Control Frequency: {control_freq:.1f} Hz")
    
    # 各组件时间
    if 'avg_source_time' in inference_stats:
        print(f"  • Source Policy Time: {inference_stats['avg_source_time']*1000:.2f} ms")
    if 'avg_scheduler_time' in inference_stats:
        print(f"  • Scheduler Time: {inference_stats['avg_scheduler_time']*1000:.2f} ms")
    if 'avg_refinement_time' in inference_stats:
        print(f"  • Refinement Time: {inference_stats['avg_refinement_time']*1000:.2f} ms")
    
    # 步数分布
    print("\n📈 Step Distribution:")
    step_dist = inference_stats.get('step_distribution', {})
    for k in sorted(step_dist.keys()):
        v = step_dist[k]
        bar = '█' * int(v * 40)
        print(f"  • k={k:2d}: {v*100:5.1f}% {bar}")
    
    # 总调用次数
    print(f"\n  Total inference calls: {inference_stats.get('total_calls', 0)}")
    
    print("\n" + "=" * 60)


def save_results(
    eval_log: dict, 
    inference_stats: dict, 
    output_dir: str,
    args: argparse.Namespace,
):
    """
    保存评估结果到 JSON 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'args': vars(args),
        'eval_log': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) 
                     for k, v in eval_log.items() if not k.endswith('video')},
        'inference_stats': inference_stats,
        'summary': {
            'test_mean_score': eval_log.get('test/mean_score', 0),
            'avg_refinement_steps': inference_stats.get('avg_steps', 0),
            'avg_inference_time_ms': inference_stats.get('avg_total_time', 0) * 1000,
            'step_distribution': inference_stats.get('step_distribution', {}),
        }
    }
    
    output_path = os.path.join(output_dir, 'ada_bridger_eval_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


def compare_with_baseline(
    ada_bridger_stats: dict,
    baseline_steps: int = 100,
    baseline_time_ms: float = 50.0,
):
    """
    与基线方法对比
    """
    print("\n" + "=" * 60)
    print("Comparison with Baseline (Standard Diffusion Policy)")
    print("=" * 60)
    
    ada_avg_steps = ada_bridger_stats.get('avg_steps', 0)
    ada_avg_time = ada_bridger_stats.get('avg_total_time', 0) * 1000
    
    step_reduction = (baseline_steps - ada_avg_steps) / baseline_steps * 100
    time_reduction = (baseline_time_ms - ada_avg_time) / baseline_time_ms * 100 if ada_avg_time > 0 else 0
    speedup = baseline_time_ms / ada_avg_time if ada_avg_time > 0 else 0
    
    print(f"\n📉 Computational Savings:")
    print(f"  • Step Reduction: {step_reduction:.1f}% ({baseline_steps} → {ada_avg_steps:.1f})")
    print(f"  • Time Reduction: {time_reduction:.1f}% ({baseline_time_ms:.1f}ms → {ada_avg_time:.1f}ms)")
    print(f"  • Speedup: {speedup:.1f}x")
    
    # 计算理论控制频率提升
    baseline_freq = 1000 / baseline_time_ms
    ada_freq = 1000 / ada_avg_time if ada_avg_time > 0 else 0
    freq_improvement = ada_freq / baseline_freq if baseline_freq > 0 else 0
    
    print(f"\n🚀 Control Frequency Improvement:")
    print(f"  • Baseline: {baseline_freq:.1f} Hz")
    print(f"  • Ada-BRIDGER: {ada_freq:.1f} Hz")
    print(f"  • Improvement: {freq_improvement:.1f}x")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Ada-BRIDGER Policy')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to Ada-BRIDGER checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='data/eval_ada_bridger',
                        help='Output directory for results')
    parser.add_argument('--n_episodes', '-n', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run evaluation on')
    parser.add_argument('--baseline_steps', type=int, default=100,
                        help='Baseline diffusion steps for comparison')
    parser.add_argument('--baseline_time_ms', type=float, default=50.0,
                        help='Baseline inference time in ms for comparison')
    parser.add_argument('--source_checkpoint', type=str, default=None,
                        help='Override VAE source policy checkpoint path')
    parser.add_argument('--refine_checkpoint', type=str, default=None,
                        help='Override Diffusion refinement policy checkpoint path')
    
    args = parser.parse_args()
    
    # 加载策略
    policy, cfg = load_ada_bridger_policy(
        args.checkpoint, args.device,
        source_checkpoint=args.source_checkpoint,
        refine_checkpoint=args.refine_checkpoint,
    )
    
    # 运行评估
    eval_log, inference_stats = evaluate_policy(
        policy=policy,
        cfg=cfg,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # 打印结果
    print_results(eval_log, inference_stats)
    
    # 与基线对比
    compare_with_baseline(
        inference_stats,
        baseline_steps=args.baseline_steps,
        baseline_time_ms=args.baseline_time_ms,
    )
    
    # 保存结果
    save_results(eval_log, inference_stats, args.output_dir, args)


if __name__ == '__main__':
    main()
