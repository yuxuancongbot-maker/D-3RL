"""
Scheduler 监督预训练脚本

使用 Oracle 标签数据监督预训练 Scheduler
让 Scheduler 学会基本的判断能力

用法：
python pretrain_scheduler.py \
    --oracle_data data/oracle_labels.pt \
    --source_ckpt <vae_checkpoint> \
    --refine_ckpt <diffusion_checkpoint> \
    --output_dir data/outputs/scheduler_pretrain \
    --epochs 50
"""

import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
import dill
from omegaconf import OmegaConf
from tqdm import tqdm
import json

from diffusion_policy.model.ada_bridger.ada_scheduler import AdaScheduler


class OracleDataset(Dataset):
    """Oracle 标签数据集，兼容旧 samples 列表和新 tensor 字典格式。"""

    def __init__(self, samples=None, data=None, indices=None):
        self.samples = samples
        self.data = data
        self.indices = indices

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return len(self.indices)

    def __getitem__(self, idx):
        if self.samples is not None:
            sample = self.samples[idx]
            init_action = sample.get('init_action', sample.get('a_init'))
            return {
                'obs': sample['obs'].float() if isinstance(sample['obs'], torch.Tensor) else torch.as_tensor(sample['obs']).float(),
                'init_action': init_action.float() if isinstance(init_action, torch.Tensor) else torch.as_tensor(init_action).float(),
                'label': torch.tensor(sample['label'], dtype=torch.long),
            }
        real_idx = int(self.indices[idx])
        init_key = 'init_action' if 'init_action' in self.data else 'a_init'
        return {
            'obs': self.data['obs'][real_idx].float(),
            'init_action': self.data[init_key][real_idx].float(),
            'label': self.data['label'][real_idx].long(),
        }


def load_policy_config(ckpt_path: str):
    """加载策略配置"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    return payload['cfg']


def create_scheduler(
    cfg,
    device,
    obs_dim=None,
    action_dim=None,
    horizon=None,
    n_obs_steps=None,
    refinement_steps=None,
):
    """创建 Scheduler"""
    policy_cfg = cfg.policy if cfg is not None and 'policy' in cfg else cfg
    scheduler = AdaScheduler(
        obs_dim=obs_dim or policy_cfg.obs_dim,
        action_dim=action_dim or policy_cfg.action_dim,
        action_horizon=horizon or policy_cfg.horizon,
        n_obs_steps=n_obs_steps or policy_cfg.n_obs_steps,
        hidden_dim=256,
        num_layers=2,
        refinement_steps=refinement_steps or [0, 1, 2, 5],
    )
    scheduler.to(device)
    return scheduler


def train_epoch(scheduler, dataloader, optimizer, device, epoch):
    """训练一个 epoch"""
    scheduler.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        obs = batch['obs'].to(device)
        init_action = batch['init_action'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        logits, _ = scheduler(obs, init_action)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.2%}'
        })
    
    return total_loss / len(dataloader), correct / total


def evaluate(scheduler, dataloader, device):
    """评估"""
    scheduler.eval()
    total_loss = 0
    correct = 0
    total = 0

    # 每个类别的统计
    class_correct = {i: 0 for i in range(scheduler.num_actions)}
    class_total = {i: 0 for i in range(scheduler.num_actions)}
    
    with torch.no_grad():
        for batch in dataloader:
            obs = batch['obs'].to(device)
            init_action = batch['init_action'].to(device)
            labels = batch['label'].to(device)
            
            logits, _ = scheduler(obs, init_action)
            loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # 每个类别的统计
            for i in range(scheduler.num_actions):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += ((pred == labels) & mask).sum().item()

    # 计算每个类别的准确率
    class_acc = {}
    for i in range(scheduler.num_actions):
        if class_total[i] > 0:
            class_acc[i] = class_correct[i] / class_total[i]
        else:
            class_acc[i] = 0.0
    
    return total_loss / len(dataloader), correct / total, class_acc


def main():
    parser = argparse.ArgumentParser(description='Scheduler 监督预训练')
    parser.add_argument('--oracle_data', type=str, required=True,
                        help='Oracle 标签数据路径')
    parser.add_argument('--source_ckpt', type=str, required=True,
                        help='VAE source policy checkpoint (用于获取配置)')
    parser.add_argument('--refine_ckpt', type=str, required=True,
                        help='Diffusion refinement policy checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/outputs/scheduler_pretrain',
                        help='输出目录')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--obs_dim', type=int, default=None,
                        help='Scheduler obs feature dim; inferred from tensor oracle obs when omitted')
    parser.add_argument('--action_dim', type=int, default=None,
                        help='Action dim; inferred from oracle init_action when omitted')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Action horizon; inferred from oracle init_action when omitted')
    parser.add_argument('--n_obs_steps', type=int, default=None,
                        help='Observation steps; inferred from oracle obs when omitted')
    parser.add_argument('--obs_mode', type=str, default=None,
                        choices=['lowdim', 'image_feature'],
                        help='Metadata marker saved in scheduler checkpoint; inferred from oracle data when omitted')
    args = parser.parse_args()
    
    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 Oracle 数据
    print(f"加载 Oracle 数据: {args.oracle_data}")
    data = torch.load(args.oracle_data)
    if 'samples' in data:
        samples = data['samples']
        n_total = len(samples)
        tensor_data = None
        print(f"样本数量: {n_total} (samples 格式)")
    else:
        required = {'obs', 'label'}
        init_key = 'init_action' if 'init_action' in data else 'a_init'
        required.add(init_key)
        missing = required - set(data.keys())
        if missing:
            raise KeyError(f"Oracle tensor data missing keys: {sorted(missing)}")
        samples = None
        tensor_data = data
        n_total = len(data['label'])
        print(f"样本数量: {n_total} (tensor 格式)")

    config = data.get('config', {})
    refinement_steps = list(config.get('refinement_steps', [0, 1, 2, 5]))
    checkpoint_obs_mode = args.obs_mode or config.get('obs_mode', 'lowdim')
    print(f"精炼步数选项: {refinement_steps}")
    print(f"obs_mode: {checkpoint_obs_mode}")

    # 分割训练/验证集
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val

    # 随机打乱
    indices = np.random.permutation(n_total)

    print(f"训练集: {n_train}, 验证集: {n_val}")

    # 创建数据集和加载器
    if samples is not None:
        train_samples = [samples[i] for i in indices[:n_train]]
        val_samples = [samples[i] for i in indices[n_train:]]
        train_dataset = OracleDataset(samples=train_samples)
        val_dataset = OracleDataset(samples=val_samples)
    else:
        train_dataset = OracleDataset(data=tensor_data, indices=indices[:n_train])
        val_dataset = OracleDataset(data=tensor_data, indices=indices[n_train:])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # 加载配置并创建 Scheduler
    print("创建 Scheduler...")
    cfg = load_policy_config(args.refine_ckpt)
    sample0 = train_dataset[0]
    obs_dim = args.obs_dim or int(sample0['obs'].shape[-1])
    n_obs_steps = args.n_obs_steps or int(sample0['obs'].shape[0])
    horizon = args.horizon or int(sample0['init_action'].shape[0])
    action_dim = args.action_dim or int(sample0['init_action'].shape[-1])
    scheduler = create_scheduler(
        cfg, device,
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        refinement_steps=refinement_steps,
    )
    
    # 重置初始偏置（移除原来的偏向 k=0 的设置）
    with torch.no_grad():
        scheduler.policy_head[-1].bias.zero_()
    
    print(f"Scheduler 参数量: {sum(p.numel() for p in scheduler.parameters()):,}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        scheduler.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    
    # 学习率调度
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )
    
    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'class_acc': [],
    }
    
    best_val_acc = 0
    best_epoch = 0
    
    print("\n开始监督预训练...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            scheduler, train_loader, optimizer, device, epoch
        )
        
        # 评估
        val_loss, val_acc, class_acc = evaluate(scheduler, val_loader, device)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['class_acc'].append(class_acc)
        
        # 打印
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
        class_acc_text = ", ".join(
            f"k={k}: {class_acc.get(i, 0):.2%}"
            for i, k in enumerate(refinement_steps)
        )
        print(f"  Class Acc: {class_acc_text}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': {
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'horizon': horizon,
                    'n_obs_steps': n_obs_steps,
                    'obs_mode': checkpoint_obs_mode,
                    'refinement_steps': refinement_steps,
                }
            }, os.path.join(args.output_dir, 'scheduler_best.pt'))
            print(f"  [保存最佳模型: val_acc={val_acc:.2%}]")
    
    # 保存最终模型
    torch.save({
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'val_acc': val_acc,
        'config': {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'horizon': horizon,
            'n_obs_steps': n_obs_steps,
            'obs_mode': checkpoint_obs_mode,
            'refinement_steps': refinement_steps,
        }
    }, os.path.join(args.output_dir, 'scheduler_final.pt'))
    
    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳 Val Acc: {best_val_acc:.2%} (Epoch {best_epoch})")
    print(f"模型保存至: {args.output_dir}")
    
    # 打印使用说明
    print("\n" + "=" * 60)
    print("下一步: 在 Ada-BRIDGER 训练中加载预训练的 Scheduler")
    print(f"  scheduler_pretrain_ckpt: {os.path.join(args.output_dir, 'scheduler_best.pt')}")


if __name__ == '__main__':
    main()
