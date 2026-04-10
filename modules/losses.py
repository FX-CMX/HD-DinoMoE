# modules/losses.py
"""
多标签分割损失函数
支持: BCE, Dice, Focal, 组合损失等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'DiceLoss', 'FocalLoss', 
    'BCEDiceLoss', 'FocalDiceLoss', 'MultiLabelLoss'
]


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-label segmentation
    支持每个通道独立计算Dice Loss
    """
    def __init__(self, smooth=1e-6, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw logits
            targets: (B, C, H, W) - binary masks
        Returns:
            dice_loss: scalar
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), probs.size(1), -1)  # (B, C, H*W)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Compute dice per channel
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice  # (B, C)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label segmentation
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) - raw logits
            targets: (B, C, H, W) - binary masks
        """
        probs = torch.sigmoid(logits)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weight and alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    Total = bce_weight * BCE + dice_weight * Dice
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss
    适用于类别不平衡的多标签分割
    """
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 alpha=0.25, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class MultiLabelLoss(nn.Module):
    """
    灵活的多标签分割损失函数
    支持为每个类别设置不同的权重
    """
    def __init__(self, loss_type='bce_dice', class_weights=None, 
                 bce_weight=0.5, dice_weight=0.5, 
                 focal_alpha=0.25, focal_gamma=2.0):
        """
        Args:
            loss_type: 'bce', 'dice', 'focal', 'bce_dice', 'focal_dice'
            class_weights: (C,) tensor - 每个类别的权重
        """
        super().__init__()
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        elif loss_type == 'dice':
            self.loss_fn = DiceLoss(reduction='none')
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
        elif loss_type == 'bce_dice':
            self.loss_fn = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
        elif loss_type == 'focal_dice':
            self.loss_fn = FocalDiceLoss(focal_weight=bce_weight, dice_weight=dice_weight,
                                          alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits, targets):
        if self.loss_type in ['bce_dice', 'focal_dice']:
            # 组合损失直接返回
            loss = self.loss_fn(logits, targets)
        else:
            # 单独损失，需要应用类别权重
            loss = self.loss_fn(logits, targets)
            
            if self.class_weights is not None:
                # loss shape: (B, C, H, W) or (B, C)
                weights = self.class_weights.to(logits.device)
                if loss.dim() == 4:
                    weights = weights.view(1, -1, 1, 1)
                elif loss.dim() == 2:
                    weights = weights.view(1, -1)
                loss = loss * weights
            
            loss = loss.mean()
        
        return loss
