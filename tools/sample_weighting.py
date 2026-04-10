# tools/sample_weighting.py
"""
样本级自适应加权模块

支持三种策略：
1. Loss-based: 基于损失的难例挖掘
2. Focal-style: Focal Loss 风格加权  
3. Curriculum: 课程学习式加权
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List

__all__ = [
    'SampleAdaptiveWeighting',
    'FocalSampleWeighting', 
    'CurriculumWeighting',
    'create_sample_weighting'
]


class SampleAdaptiveWeighting:
    """
    基于损失的样本自适应加权
    
    原理：维护每个样本的历史损失，损失高的样本获得更高权重
    """
    def __init__(
        self, 
        num_samples: int,
        temperature: float = 1.0,
        momentum: float = 0.9,
        focus_mode: str = 'hard',  # 'hard', 'easy', 'balanced'
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        """
        Args:
            num_samples: 数据集中样本总数
            temperature: softmax 温度，越大权重越均匀
            momentum: 历史损失的平滑动量
            focus_mode: 'hard'=关注难例, 'easy'=关注简单样本, 'balanced'=平衡
            min_weight: 最小权重
            max_weight: 最大权重
        """
        self.num_samples = num_samples
        self.temperature = temperature
        self.momentum = momentum
        self.focus_mode = focus_mode
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 初始化每个样本的历史损失（初始为1.0，表示均匀）
        self.sample_losses = torch.ones(num_samples)
        self.update_count = torch.zeros(num_samples)
        
        print(f"[SampleAdaptiveWeighting] num_samples={num_samples}, "
              f"temp={temperature}, focus={focus_mode}")
    
    def update_losses(self, sample_indices: torch.Tensor, losses: torch.Tensor):
        """
        更新样本的历史损失
        
        Args:
            sample_indices: 样本索引 [B]
            losses: 每个样本的损失 [B]
        """
        for idx, loss in zip(sample_indices.cpu(), losses.detach().cpu()):
            idx = int(idx)
            if self.update_count[idx] == 0:
                self.sample_losses[idx] = loss
            else:
                self.sample_losses[idx] = (
                    self.momentum * self.sample_losses[idx] + 
                    (1 - self.momentum) * loss
                )
            self.update_count[idx] += 1
    
    def get_weights(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """
        获取指定样本的权重
        
        Args:
            sample_indices: 样本索引 [B]
            
        Returns:
            weights: 样本权重 [B]
        """
        losses = self.sample_losses[sample_indices.cpu()]
        
        if self.focus_mode == 'hard':
            # 损失高 → 权重高
            weights = F.softmax(losses / self.temperature, dim=0) * len(losses)
        elif self.focus_mode == 'easy':
            # 损失低 → 权重高
            weights = F.softmax(-losses / self.temperature, dim=0) * len(losses)
        else:  # balanced
            # 适中损失权重最高（接近中位数）
            median_loss = losses.median()
            diff = (losses - median_loss).abs()
            weights = F.softmax(-diff / self.temperature, dim=0) * len(losses)
        
        # 限制权重范围
        weights = weights.clamp(self.min_weight, self.max_weight)
        
        return weights.to(sample_indices.device)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        valid_mask = self.update_count > 0
        valid_losses = self.sample_losses[valid_mask]
        
        return {
            'mean_loss': valid_losses.mean().item() if len(valid_losses) > 0 else 0,
            'std_loss': valid_losses.std().item() if len(valid_losses) > 0 else 0,
            'min_loss': valid_losses.min().item() if len(valid_losses) > 0 else 0,
            'max_loss': valid_losses.max().item() if len(valid_losses) > 0 else 0,
            'updated_samples': valid_mask.sum().item(),
        }


class FocalSampleWeighting:
    """
    Focal Loss 风格的样本加权
    
    原理：低置信度（高损失）样本获得更高权重
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = 'none'
    ):
        """
        Args:
            gamma: focusing parameter，越大越关注难例
            alpha: 平衡因子
        """
        self.gamma = gamma
        self.alpha = alpha
        
        print(f"[FocalSampleWeighting] gamma={gamma}, alpha={alpha}")
    
    def compute_weights(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        根据预测和目标计算样本权重
        
        Args:
            logits: 模型输出 [B, C, H, W]
            targets: 目标 [B, C, H, W]
            
        Returns:
            weights: 每个样本的权重 [B]
        """
        probs = torch.sigmoid(logits)
        
        # 计算每个样本的平均预测概率
        # 对于正确预测，p_t 高；对于错误预测，p_t 低
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 每个样本的平均 p_t
        p_t_mean = p_t.mean(dim=[1, 2, 3])  # [B]
        
        # Focal 权重：(1 - p_t)^gamma
        focal_weight = (1 - p_t_mean) ** self.gamma
        
        # 归一化使得平均权重为 1
        weights = focal_weight / focal_weight.mean() if focal_weight.mean() > 0 else focal_weight
        
        return weights


class CurriculumWeighting:
    """
    课程学习式样本加权
    
    原理：训练早期关注简单样本，后期逐步增加困难样本
    """
    def __init__(
        self,
        num_samples: int,
        total_epochs: int,
        warmup_epochs: int = 10,
        temperature: float = 1.0,
        momentum: float = 0.9
    ):
        """
        Args:
            num_samples: 样本总数
            total_epochs: 总训练轮数
            warmup_epochs: 预热轮数（简单样本优先）
            temperature: softmax 温度
            momentum: 损失平滑动量
        """
        self.num_samples = num_samples
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.temperature = temperature
        self.momentum = momentum
        
        self.sample_losses = torch.ones(num_samples)
        self.current_epoch = 0
        
        print(f"[CurriculumWeighting] num_samples={num_samples}, "
              f"warmup={warmup_epochs}/{total_epochs}")
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch"""
        self.current_epoch = epoch
    
    def update_losses(self, sample_indices: torch.Tensor, losses: torch.Tensor):
        """更新样本损失"""
        for idx, loss in zip(sample_indices.cpu(), losses.detach().cpu()):
            idx = int(idx)
            self.sample_losses[idx] = (
                self.momentum * self.sample_losses[idx] + 
                (1 - self.momentum) * loss
            )
    
    def get_weights(self, sample_indices: torch.Tensor) -> torch.Tensor:
        """
        获取样本权重，根据当前 epoch 调整策略
        """
        losses = self.sample_losses[sample_indices.cpu()]
        
        # 计算课程进度：0 → 1
        progress = min(1.0, self.current_epoch / max(1, self.warmup_epochs))
        
        # 早期：简单样本权重高；后期：困难样本权重高
        # sign = -1 + 2 * progress  → 从 -1 渐变到 1
        sign = -1.0 + 2.0 * progress
        
        weights = F.softmax(sign * losses / self.temperature, dim=0) * len(losses)
        
        return weights.to(sample_indices.device)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'current_epoch': self.current_epoch,
            'curriculum_progress': min(1.0, self.current_epoch / max(1, self.warmup_epochs)),
            'mean_loss': self.sample_losses.mean().item(),
            'focus': 'easy' if self.current_epoch < self.warmup_epochs else 'hard',
        }


def create_sample_weighting(
    strategy: str,
    num_samples: int,
    **kwargs
):
    """
    创建样本加权策略
    
    Args:
        strategy: 'loss_based', 'focal', 'curriculum', 'none'
        num_samples: 样本总数
        **kwargs: 策略特定参数
    
    Returns:
        weighting 对象
    """
    if strategy == 'loss_based':
        return SampleAdaptiveWeighting(
            num_samples=num_samples,
            temperature=kwargs.get('temperature', 1.0),
            momentum=kwargs.get('momentum', 0.9),
            focus_mode=kwargs.get('focus_mode', 'hard'),
        )
    elif strategy == 'focal':
        return FocalSampleWeighting(
            gamma=kwargs.get('gamma', 2.0),
            alpha=kwargs.get('alpha', 0.25),
        )
    elif strategy == 'curriculum':
        return CurriculumWeighting(
            num_samples=num_samples,
            total_epochs=kwargs.get('total_epochs', 50),
            warmup_epochs=kwargs.get('warmup_epochs', 10),
            temperature=kwargs.get('temperature', 1.0),
        )
    elif strategy == 'none':
        return None
    else:
        raise ValueError(f"Unknown sample weighting strategy: {strategy}")
