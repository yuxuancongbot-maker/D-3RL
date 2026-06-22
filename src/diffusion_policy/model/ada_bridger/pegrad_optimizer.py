"""
PEGrad: Pareto-Efficient Gradient Projection for Multi-Objective Optimization

基于梯度投影的无冲突多目标优化:
- 任务奖励梯度 g_task: 最大化任务成功率
- 效率惩罚梯度 g_cost: 最小化推理步数
- 投影策略: 将 g_cost 投影到 g_task 的正交平面

效果:
- 系统自动在"确保成功"的前提下追求"最快速度"
- 消除手动调节奖励权重的痛苦
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer


def project_gradient(
    grad_main: torch.Tensor,
    grad_aux: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    将辅助梯度投影到主梯度的正交平面
    
    Args:
        grad_main: 主梯度 (任务奖励梯度)
        grad_aux: 辅助梯度 (效率惩罚梯度)
        eps: 数值稳定性
        
    Returns:
        projected_grad: 投影后的辅助梯度
    """
    # 计算投影: g_aux_proj = g_aux - (g_aux · g_main / ||g_main||^2) * g_main
    dot_product = torch.sum(grad_aux * grad_main)
    main_norm_sq = torch.sum(grad_main * grad_main) + eps
    
    # 只有当辅助梯度与主梯度冲突时才投影
    if dot_product < 0:
        # 投影到正交平面
        projection = (dot_product / main_norm_sq) * grad_main
        projected_grad = grad_aux - projection
    else:
        # 不冲突，保持原样
        projected_grad = grad_aux
    
    return projected_grad


def compute_conflict_ratio(
    grad_main: torch.Tensor,
    grad_aux: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    计算梯度冲突比例 (用于监控)
    
    Returns:
        conflict_ratio: 冲突程度 (负值表示冲突)
    """
    dot_product = torch.sum(grad_aux * grad_main)
    main_norm = torch.sqrt(torch.sum(grad_main * grad_main) + eps)
    aux_norm = torch.sqrt(torch.sum(grad_aux * grad_aux) + eps)
    
    cosine = dot_product / (main_norm * aux_norm + eps)
    return cosine.item()


class PEGradOptimizer:
    """
    PEGrad 优化器包装器
    
    在 PPO 更新时应用 Pareto-Efficient 梯度投影
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        task_weight: float = 1.0,
        cost_weight: float = 0.1,
        project_cost_to_task: bool = True,
        clip_grad_norm: Optional[float] = 0.5,
    ):
        """
        Args:
            optimizer: 基础优化器 (Adam, AdamW 等)
            task_weight: 任务奖励权重
            cost_weight: 效率惩罚权重
            project_cost_to_task: 是否将效率梯度投影到任务梯度
            clip_grad_norm: 梯度裁剪范数
        """
        self.optimizer = optimizer
        self.task_weight = task_weight
        self.cost_weight = cost_weight
        self.project_cost_to_task = project_cost_to_task
        self.clip_grad_norm = clip_grad_norm
        
        # 监控指标
        self._last_conflict_ratio = 0.0
        self._last_task_grad_norm = 0.0
        self._last_cost_grad_norm = 0.0
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step_with_pegrad(
        self,
        task_loss: torch.Tensor,
        cost_loss: torch.Tensor,
        parameters: List[torch.nn.Parameter],
        retain_graph: bool = False,
    ) -> Dict[str, float]:
        """
        使用 PEGrad 进行优化步骤
        
        Args:
            task_loss: 任务损失 (需要最小化以最大化奖励)
            cost_loss: 效率损失 (需要最小化以减少步数)
            parameters: 模型参数列表
            retain_graph: 是否保留计算图
            
        Returns:
            info: 优化信息字典
        """
        # 1. 计算任务梯度
        self.optimizer.zero_grad()
        task_loss.backward(retain_graph=True)
        task_grads = self._collect_grads(parameters)
        
        # 2. 计算效率梯度
        self.optimizer.zero_grad()
        cost_loss.backward(retain_graph=retain_graph)
        cost_grads = self._collect_grads(parameters)
        
        # 3. 展平梯度
        task_grad_flat = self._flatten_grads(task_grads)
        cost_grad_flat = self._flatten_grads(cost_grads)
        
        # 4. 计算冲突比例
        conflict_ratio = compute_conflict_ratio(task_grad_flat, cost_grad_flat)
        self._last_conflict_ratio = conflict_ratio
        
        # 5. 投影效率梯度
        if self.project_cost_to_task:
            cost_grad_projected = project_gradient(task_grad_flat, cost_grad_flat)
        else:
            cost_grad_projected = cost_grad_flat
        
        # 6. 组合梯度
        combined_grad = (
            self.task_weight * task_grad_flat + 
            self.cost_weight * cost_grad_projected
        )
        
        # 7. 梯度裁剪
        if self.clip_grad_norm is not None:
            grad_norm = torch.norm(combined_grad)
            if grad_norm > self.clip_grad_norm:
                combined_grad = combined_grad * (self.clip_grad_norm / grad_norm)
        
        # 8. 将组合梯度写回参数
        self._set_grads(parameters, combined_grad)
        
        # 9. 优化器步骤
        self.optimizer.step()
        
        # 记录指标
        self._last_task_grad_norm = torch.norm(task_grad_flat).item()
        self._last_cost_grad_norm = torch.norm(cost_grad_flat).item()
        
        return {
            'conflict_ratio': conflict_ratio,
            'task_grad_norm': self._last_task_grad_norm,
            'cost_grad_norm': self._last_cost_grad_norm,
            'combined_grad_norm': torch.norm(combined_grad).item(),
        }
    
    def step(self):
        """标准优化步骤（不使用PEGrad）"""
        if self.clip_grad_norm is not None:
            params = []
            for group in self.optimizer.param_groups:
                params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.optimizer.step()
    
    def _collect_grads(self, parameters: List[torch.nn.Parameter]) -> List[torch.Tensor]:
        """收集参数梯度"""
        grads = []
        for p in parameters:
            if p.grad is not None:
                grads.append(p.grad.clone())
            else:
                grads.append(torch.zeros_like(p))
        return grads
    
    def _flatten_grads(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """展平梯度列表"""
        return torch.cat([g.flatten() for g in grads])
    
    def _set_grads(self, parameters: List[torch.nn.Parameter], flat_grad: torch.Tensor):
        """将展平的梯度设置回参数"""
        offset = 0
        for p in parameters:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].view_as(p)
            offset += numel
    
    def get_last_info(self) -> Dict[str, float]:
        """获取上一次优化的信息"""
        return {
            'conflict_ratio': self._last_conflict_ratio,
            'task_grad_norm': self._last_task_grad_norm,
            'cost_grad_norm': self._last_cost_grad_norm,
        }
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups


class PPOWithPEGrad:
    """
    结合 PEGrad 的 PPO 算法
    
    奖励函数设计:
    R = R_task + λ * (1 - k/K_max)
    
    其中:
    - R_task: 任务完成奖励
    - λ: 效率奖励系数
    - k: 精炼步数
    - K_max: 最大精炼步数
    """
    
    def __init__(
        self,
        policy: nn.Module,
        optimizer: PEGradOptimizer,
        # PPO 超参数
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        # 奖励配置
        efficiency_coef: float = 0.1,
        max_refinement_steps: int = 5,
        # GAE
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # 训练配置
        n_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.policy = policy
        self.optimizer = optimizer
        
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.efficiency_coef = efficiency_coef
        self.max_refinement_steps = max_refinement_steps
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
    
    def compute_reward(
        self,
        task_reward: torch.Tensor,
        refinement_steps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算组合奖励
        
        Returns:
            total_reward: 总奖励
            efficiency_reward: 效率奖励
        """
        # 效率奖励: 步数越少越高
        efficiency_reward = 1.0 - refinement_steps.float() / self.max_refinement_steps
        
        # 总奖励
        total_reward = task_reward + self.efficiency_coef * efficiency_reward
        
        return total_reward, efficiency_reward
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: [T] 奖励序列
            values: [T] 价值估计序列
            dones: [T] 终止标志
            next_value: 下一状态价值
            
        Returns:
            advantages: [T] 优势估计
            returns: [T] 回报
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        obs: torch.Tensor,
        init_actions: torch.Tensor,
        actions: torch.Tensor,  # scheduler action indices
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        refinement_steps: torch.Tensor,
    ) -> Dict[str, float]:
        """
        PPO 更新步骤（使用 PEGrad 双损失分离）

        task_loss  = PPO clip loss on task advantages (from GAE on env reward)
        cost_loss  = PPO clip loss on cost advantages (immediate efficiency signal)
        PEGrad 将 cost 梯度投影到 task 梯度正交面，避免效率目标损害任务性能。
        当 project_cost_to_task=False 时退化为 naive scalarization（消融用）。

        Returns:
            info: 训练信息
        """
        # ---- 任务优势（来自环境奖励的 GAE）----
        task_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 效率优势（即时信号，无需 GAE）----
        # cost = k / K_max ∈ [0,1]；越低越好 → advantage = -cost
        cost_signal = -(refinement_steps.float() / self.max_refinement_steps)
        if cost_signal.std() > 1e-8:
            cost_advantages = (cost_signal - cost_signal.mean()) / (cost_signal.std() + 1e-8)
        else:
            cost_advantages = torch.zeros_like(cost_signal)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_conflict_ratio = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_init_actions = init_actions[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_task_adv = task_advantages[batch_indices]
                batch_cost_adv = cost_advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估当前策略
                new_log_probs, entropy, values = self.policy.evaluate_scheduler_actions(
                    batch_obs, batch_init_actions, batch_actions
                )

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # === 任务损失 (PPO clip on task advantage) ===
                surr1 = ratio * batch_task_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_task_adv
                task_policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()
                task_loss = task_policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # === 效率损失 (PPO clip on cost advantage) ===
                surr1_c = ratio * batch_cost_adv
                surr2_c = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_cost_adv
                cost_loss = -torch.min(surr1_c, surr2_c).mean()

                # === PEGrad：投影 cost 梯度到 task 梯度正交面 ===
                params = list(self.policy.scheduler.parameters())
                info = self.optimizer.step_with_pegrad(
                    task_loss=task_loss,
                    cost_loss=cost_loss,
                    parameters=params,
                    retain_graph=False,
                )

                total_policy_loss += task_policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_conflict_ratio += info['conflict_ratio']
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'conflict_ratio': total_conflict_ratio / max(n_updates, 1),
        }
