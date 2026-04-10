# modules/metrics.py
"""
多标签分割评估指标
支持: Dice, IoU, HD95, ASD, Sensitivity, Specificity, Precision, F1等
每个类别独立计算
"""
import cv2
import numpy as np
import torch

__all__ = [
    'dice_score', 'iou_score', 'sensitivity_score', 'specificity_score',
    'precision_score', 'f1_score', 'hd95_score', 'asd_score', 'bf1_score',
    'compute_all_metrics', 'MetricsCalculator'
]


# ==================== PyTorch版本（用于训练时快速计算） ====================

def dice_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """
    计算Dice系数
    Args:
        pred_logits: (B, C, H, W) - raw logits
        target: (B, C, H, W) - binary masks
    Returns:
        dice: (B, C) - Dice score for each class
    """
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice  # (B, C)


def iou_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """计算IoU (Jaccard Index)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou  # (B, C)


def sensitivity_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """Sensitivity (Recall) = TP / (TP + FN)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    
    TP = (pred * target).sum(dim=(2, 3))
    FN = ((1 - pred) * target).sum(dim=(2, 3))
    sens = (TP + eps) / (TP + FN + eps)
    return sens  # (B, C)


def specificity_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """Specificity = TN / (TN + FP)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    
    TN = ((1 - pred) * (1 - target)).sum(dim=(2, 3))
    FP = (pred * (1 - target)).sum(dim=(2, 3))
    spec = (TN + eps) / (TN + FP + eps)
    return spec  # (B, C)


def precision_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """Precision = TP / (TP + FP)"""
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    
    TP = (pred * target).sum(dim=(2, 3))
    FP = (pred * (1 - target)).sum(dim=(2, 3))
    prec = (TP + eps) / (TP + FP + eps)
    return prec  # (B, C)


def f1_score(pred_logits, target, eps=1e-6, thresh=0.5):
    """F1 Score = 2 * Precision * Recall / (Precision + Recall)"""
    prec = precision_score(pred_logits, target, eps, thresh)
    sens = sensitivity_score(pred_logits, target, eps, thresh)
    f1 = (2 * prec * sens + eps) / (prec + sens + eps)
    return f1  # (B, C)


# ==================== NumPy版本（用于验证时精确计算距离指标） ====================

def _binary_boundary(mask_bin: np.ndarray) -> np.ndarray:
    """提取二值mask的边界"""
    mask_u8 = (mask_bin.astype(np.uint8) * 255)
    if mask_u8.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = np.zeros_like(mask_u8)
    for cnt in contours:
        cv2.drawContours(boundary, [cnt], -1, color=1, thickness=1)
    return boundary.astype(np.uint8)


def _get_surface_distances(pred_bin: np.ndarray, tgt_bin: np.ndarray):
    """计算表面距离"""
    pred_has = pred_bin.any()
    tgt_has = tgt_bin.any()
    
    if (not pred_has) and (not tgt_has):
        return np.array([0.0]), np.array([0.0])
    if (pred_has and (not tgt_has)) or ((not pred_has) and tgt_has):
        return np.array([float("inf")]), np.array([float("inf")])
    
    Pb = _binary_boundary(pred_bin)
    Tb = _binary_boundary(tgt_bin)
    if Pb.max() == 0:
        Pb = pred_bin.astype(np.uint8)
    if Tb.max() == 0:
        Tb = tgt_bin.astype(np.uint8)
    
    def dist_to_border(border01):
        inv = np.where(border01 > 0, 0, 1).astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    
    dist_to_T = dist_to_border(Tb)
    dist_to_P = dist_to_border(Pb)
    Py, Px = np.where(Pb > 0)
    Ty, Tx = np.where(Tb > 0)
    
    if len(Py) == 0 or len(Ty) == 0:
        return np.array([float("inf")]), np.array([float("inf")])
    return dist_to_T[Py, Px], dist_to_P[Ty, Tx]


def hd95_score(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    """计算HD95（Hausdorff Distance 95%）"""
    d_PT, d_TP = _get_surface_distances(pred_bin, tgt_bin)
    if not np.isfinite(d_PT).all() or not np.isfinite(d_TP).all():
        return float("inf")
    d_all = np.concatenate([d_PT, d_TP])
    return float(np.percentile(d_all, 95)) if d_all.size > 0 else float("inf")


def asd_score(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    """计算ASD（Average Surface Distance）"""
    d_PT, d_TP = _get_surface_distances(pred_bin, tgt_bin)
    if not np.isfinite(d_PT).all() or not np.isfinite(d_TP).all():
        return float("inf")
    return float((np.mean(d_PT) + np.mean(d_TP)) / 2.0)


def bf1_score(pred_bin: np.ndarray, tgt_bin: np.ndarray, threshold: float = 2.0) -> float:
    """
    计算 BF1 (Boundary F1) 分数
    Args:
        pred_bin: 二值化预测图 (H, W)
        tgt_bin: 二值化真值图 (H, W)
        threshold: 距离阈值 (默认 2 像素)
    Returns:
        bf1: 边界 F1 分数
    """
    pred_has = pred_bin.any()
    tgt_has = tgt_bin.any()
    
    if (not pred_has) and (not tgt_has):
        return 1.0
    if (pred_has and (not tgt_has)) or ((not pred_has) and tgt_has):
        return 0.0
    
    Pb = _binary_boundary(pred_bin)
    Tb = _binary_boundary(tgt_bin)
    
    if Pb.max() == 0 or Tb.max() == 0:
        return 0.0
        
    def dist_to_border(border01):
        inv = np.where(border01 > 0, 0, 1).astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    
    dist_to_T = dist_to_border(Tb)
    dist_to_P = dist_to_border(Pb)
    
    Py, Px = np.where(Pb > 0)
    Ty, Tx = np.where(Tb > 0)
    
    # Precision: 预测边界在真值边界一定范围内的比例
    precision = np.mean(dist_to_T[Py, Px] <= threshold)
    # Recall: 真值边界在预测边界一定范围内的比例
    recall = np.mean(dist_to_P[Ty, Tx] <= threshold)
    
    if precision + recall == 0:
        return 0.0
    
    bf1 = 2 * precision * recall / (precision + recall)
    return float(bf1)


# ==================== 综合指标计算器 ====================

def compute_all_metrics(pred_logits, targets, thresh=0.5, compute_distance=True):
    """
    计算所有指标
    Args:
        pred_logits: (B, C, H, W) tensor
        targets: (B, C, H, W) tensor
        thresh: 二值化阈值
        compute_distance: 是否计算距离指标(HD95, ASD)
    Returns:
        dict: 每个类别和平均值的所有指标
    """
    B, C, H, W = pred_logits.shape
    
    # PyTorch指标
    dice = dice_score(pred_logits, targets, thresh=thresh)  # (B, C)
    iou = iou_score(pred_logits, targets, thresh=thresh)
    sens = sensitivity_score(pred_logits, targets, thresh=thresh)
    spec = specificity_score(pred_logits, targets, thresh=thresh)
    prec = precision_score(pred_logits, targets, thresh=thresh)
    f1 = f1_score(pred_logits, targets, thresh=thresh)
    
    results = {}
    
    # 每个类别的指标
    for c in range(C):
        results[f'dice_class{c}'] = dice[:, c].mean().item()
        results[f'iou_class{c}'] = iou[:, c].mean().item()
        results[f'sensitivity_class{c}'] = sens[:, c].mean().item()
        results[f'specificity_class{c}'] = spec[:, c].mean().item()
        results[f'precision_class{c}'] = prec[:, c].mean().item()
        results[f'f1_class{c}'] = f1[:, c].mean().item()
    
    # 平均值
    results['dice_mean'] = dice.mean().item()
    results['iou_mean'] = iou.mean().item()
    results['sensitivity_mean'] = sens.mean().item()
    results['specificity_mean'] = spec.mean().item()
    results['precision_mean'] = prec.mean().item()
    results['f1_mean'] = f1.mean().item()
    
    # 距离指标（需要逐样本逐类别计算）
    if compute_distance:
        probs = torch.sigmoid(pred_logits)
        preds = (probs > thresh).float()
        
        hd95_per_class = {c: [] for c in range(C)}
        asd_per_class = {c: [] for c in range(C)}
        bf1_per_class = {c: [] for c in range(C)}
        
        for b in range(B):
            for c in range(C):
                gt_c = targets[b, c].cpu().numpy() > 0.5
                pr_c = preds[b, c].cpu().numpy() > 0.5
                hd95_per_class[c].append(hd95_score(pr_c, gt_c))
                asd_per_class[c].append(asd_score(pr_c, gt_c))
                bf1_per_class[c].append(bf1_score(pr_c, gt_c))
        
        for c in range(C):
            finite_hd = [x for x in hd95_per_class[c] if np.isfinite(x)]
            finite_asd = [x for x in asd_per_class[c] if np.isfinite(x)]
            results[f'hd95_class{c}'] = np.mean(finite_hd) if finite_hd else float('inf')
            results[f'asd_class{c}'] = np.mean(finite_asd) if finite_asd else float('inf')
            results[f'bf1_class{c}'] = np.mean(bf1_per_class[c])
        
        # 距离平均值
        hd95_all = [results[f'hd95_class{c}'] for c in range(C) if np.isfinite(results[f'hd95_class{c}'])]
        asd_all = [results[f'asd_class{c}'] for c in range(C) if np.isfinite(results[f'asd_class{c}'])]
        results['hd95_mean'] = np.mean(hd95_all) if hd95_all else float('inf')
        results['asd_mean'] = np.mean(asd_all) if asd_all else float('inf')
        results['bf1_mean'] = np.mean([results[f'bf1_class{c}'] for c in range(C)])
    
    return results


class MetricsCalculator:
    """
    指标计算器，用于训练过程中累积和计算指标
    """
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """重置所有累积值"""
        self.metrics_sum = {}
        self.count = 0
    
    def update(self, pred_logits, targets, thresh=0.5):
        """更新累积指标"""
        metrics = compute_all_metrics(pred_logits, targets, thresh, compute_distance=False)
        
        for k, v in metrics.items():
            if k not in self.metrics_sum:
                self.metrics_sum[k] = 0.0
            self.metrics_sum[k] += v
        
        self.count += 1
    
    def compute(self):
        """计算平均指标"""
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.metrics_sum.items()}
    
    def get_summary_string(self):
        """获取汇总字符串"""
        metrics = self.compute()
        if not metrics:
            return "No metrics computed"
        
        lines = []
        lines.append(f"Mean: Dice={metrics.get('dice_mean', 0):.4f}  "
                     f"IoU={metrics.get('iou_mean', 0):.4f}  "
                     f"F1={metrics.get('f1_mean', 0):.4f}")
        
        for c in range(self.num_classes):
            lines.append(f"  {self.class_names[c]}: "
                         f"Dice={metrics.get(f'dice_class{c}', 0):.4f}  "
                         f"IoU={metrics.get(f'iou_class{c}', 0):.4f}")
        
        return '\n'.join(lines)
