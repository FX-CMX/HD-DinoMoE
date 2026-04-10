# tools/class_aware_weighting.py
"""
类别感知样本加权模块 (Class-Aware Sample Weighting)

核心创新：维护 [num_samples, num_classes] 的历史损失矩阵，
使不同类别的难度被独立评估和加权。

可选：利用 CA-MoE 门控熵作为不确定性信号，
对路由器不确定的样本-类别组合施加额外关注。

权重公式:
    W[i,c] = softmax(L[i,c] / τ) × N × (1 + λ × H_gate[c,i])

其中:
    L[i,c]: 样本 i 对类别 c 的历史损失（EMA 平滑）
    τ:      温度参数
    H_gate: 门控熵（可选）
    λ:      门控熵调制系数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


__all__ = [
    'ClassAwareSampleWeighting',
    'compute_per_class_loss',
]


class ClassAwareSampleWeighting:
    """
    类别感知样本加权
    
    与传统 SampleAdaptiveWeighting 的区别:
    - 传统方案: 维护 [num_samples] 标量损失 → 不区分类别
    - 本方案:   维护 [num_samples, num_classes] 矩阵 → 逐类别独立评估
    
    典型场景: 对于一张图片, SAT 分割已很好(dice=0.95), 
    但 LVD 分割很差(dice=0.3)。传统方案将其标记为"中等难度",
    而本方案能识别出它对 LVD 是强难例, 应被大幅加权。
    """
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        temperature: float = 1.0,
        momentum: float = 0.9,
        focus_mode: str = 'hard',
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        gate_entropy_lambda: float = 0.0
    ):
        """
        Args:
            num_samples: 数据集中样本总数
            num_classes: 类别数
            temperature: softmax 温度，越大权重越均匀
            momentum: 历史损失的 EMA 平滑动量
            focus_mode: 'hard'=关注难例, 'easy'=关注简单, 'balanced'=平衡
            min_weight: 最小权重
            max_weight: 最大权重
            gate_entropy_lambda: 门控熵调制系数（0 表示不使用）
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        self.focus_mode = focus_mode
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.gate_entropy_lambda = gate_entropy_lambda
        
        # 核心：[num_samples, num_classes] 的历史损失矩阵
        self.sample_losses = torch.ones(num_samples, num_classes)
        self.update_count = torch.zeros(num_samples)
        
        print(f"[ClassAwareSampleWeighting] num_samples={num_samples}, "
              f"num_classes={num_classes}, temp={temperature}, "
              f"focus={focus_mode}, gate_lambda={gate_entropy_lambda}")
    
    def update_losses(self, sample_indices: torch.Tensor, per_class_losses: torch.Tensor):
        """
        更新样本的逐类别历史损失（EMA 平滑）
        
        Args:
            sample_indices: 样本索引 [B]
            per_class_losses: 每个样本每个类别的损失 [B, C]
        """
        for b, idx in enumerate(sample_indices.cpu()):
            idx = int(idx)
            losses_c = per_class_losses[b].detach().cpu()  # [C]
            if self.update_count[idx] == 0:
                self.sample_losses[idx] = losses_c
            else:
                self.sample_losses[idx] = (
                    self.momentum * self.sample_losses[idx] +
                    (1 - self.momentum) * losses_c
                )
            self.update_count[idx] += 1
    
    def get_weights(
        self, 
        sample_indices: torch.Tensor,
        gate_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取逐类别的样本权重
        
        Args:
            sample_indices: 样本索引 [B]
            gate_weights: CA-MoE 门控权重 [num_classes, B, num_experts]（可选）
        
        Returns:
            weights: 每个样本每个类别的权重 [B, C]
        """
        # 获取这批样本的历史损失 [B, C]
        losses = self.sample_losses[sample_indices.cpu()]  # [B, C]
        B, C = losses.shape
        
        # 逐类别计算权重
        weights = torch.zeros_like(losses)
        for c in range(C):
            losses_c = losses[:, c]  # [B]
            if self.focus_mode == 'hard':
                w = F.softmax(losses_c / self.temperature, dim=0) * B
            elif self.focus_mode == 'easy':
                w = F.softmax(-losses_c / self.temperature, dim=0) * B
            else:  # balanced
                median = losses_c.median()
                diff = (losses_c - median).abs()
                w = F.softmax(-diff / self.temperature, dim=0) * B
            weights[:, c] = w
        
        # 门控熵调制
        if self.gate_entropy_lambda > 0 and gate_weights is not None:
            # gate_weights: [num_classes, B, num_experts]
            # 门控熵: H = -Σ p·log(p), 熵高 → 路由器不确定 → 需要更多关注
            gate_entropy = -torch.sum(
                gate_weights * torch.log(gate_weights + 1e-8), dim=-1
            )  # [num_classes, B]
            gate_entropy = gate_entropy.permute(1, 0)  # [B, num_classes]
            
            # 归一化熵到 [0, 1]
            max_entropy = np.log(gate_weights.shape[-1])  # log(num_experts)
            gate_entropy_norm = gate_entropy / max_entropy
            
            # 调制: W *= (1 + λ * H)
            weights = weights * (1.0 + self.gate_entropy_lambda * gate_entropy_norm.detach().cpu())
        
        # 限制权重范围
        weights = weights.clamp(self.min_weight, self.max_weight)
        
        return weights.to(sample_indices.device)
    
    def get_statistics(self) -> Dict:
        """获取逐类别统计信息"""
        valid_mask = self.update_count > 0
        stats = {
            'updated_samples': valid_mask.sum().item(),
        }
        valid_losses = self.sample_losses[valid_mask]  # [N_valid, C]
        if len(valid_losses) > 0:
            for c in range(self.num_classes):
                col = valid_losses[:, c]
                stats[f'class{c}_mean_loss'] = col.mean().item()
                stats[f'class{c}_std_loss'] = col.std().item()
        return stats


def compute_per_class_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    计算逐类别的 BCE+Dice 损失
    
    Args:
        logits: 模型输出 [B, C, H, W]
        targets: 目标 [B, C, H, W]
        smooth: Dice 计算的平滑因子
    
    Returns:
        per_class_loss: 每个样本每个类别的损失 [B, C]
    """
    B, C, H, W = logits.shape
    pred_prob = torch.sigmoid(logits)
    
    # BCE per class: [B, C]
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    ).mean(dim=[2, 3])  # [B, C]
    
    # Dice per class: [B, C]
    p = pred_prob.reshape(B, C, -1)  # [B, C, H*W]
    t = targets.reshape(B, C, -1)
    intersection = (p * t).sum(dim=2)
    union = p.sum(dim=2) + t.sum(dim=2)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice  # [B, C]
    
    return 0.5 * bce + 0.5 * dice_loss
