#!/usr/bin/env python3
# tools/train_hd_moe_joint.py
"""
层级双MoE (HD-MoE) 单阶段联合训练脚本

与三阶段版本 (train_hd_moe_staged_v2.py) 的核心区别：
- 直接构建完整 HD-MoE 模型，端到端联合训练
- 双 Backbone 不冻结（默认），所有参数一起训练
- 支持差异化学习率：Backbone 低学习率，Gate/Decoder 标准学习率
- 无阶段切换，单一训练循环

保留功能：
1. 反光抑制渐进式损失 (GlareProgressiveLoss)
2. 样本级自适应加权 (loss_based / focal / curriculum / class_aware)
3. 混合精度训练 (fp16 / bf16)
4. 门控权重记录与可视化
5. Resume 断点续训
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hd_moe_model import build_hd_moe_model
from dataset import MultiLabelDataset, ResizeAndNormalize
from modules.visualize import save_train_visuals, plot_training_curves, save_training_log, init_history
from modules.metrics import dice_score, iou_score
from modules.losses import BCEDiceLoss, FocalDiceLoss

# 反光数据集(可选)
try:
    from dataset_glare_v2 import MultiLabelGlareDatasetV2, ResizeAndNormalize as GlareResizeAndNormalize
    GLARE_DATASET_AVAILABLE = True
except ImportError:
    GLARE_DATASET_AVAILABLE = False
    GlareResizeAndNormalize = None

# 样本加权模块(可选)
try:
    from sample_weighting import create_sample_weighting, SampleAdaptiveWeighting, CurriculumWeighting, FocalSampleWeighting
    SAMPLE_WEIGHTING_AVAILABLE = True
except ImportError:
    SAMPLE_WEIGHTING_AVAILABLE = False
    SampleAdaptiveWeighting = None
    CurriculumWeighting = None
    FocalSampleWeighting = None

# 类别感知样本加权(可选)
try:
    from class_aware_weighting import ClassAwareSampleWeighting, compute_per_class_loss
    CLASS_AWARE_WEIGHTING_AVAILABLE = True
except ImportError:
    CLASS_AWARE_WEIGHTING_AVAILABLE = False
    ClassAwareSampleWeighting = None
    compute_per_class_loss = None


# ============== 全局配置 ==============
AUTO_CLEANUP = False

# 默认专家
DEFAULT_EXPERTS = ['sam', 'd2s', 'dpt', 'linear_attn']


class IndexedDataset(torch.utils.data.Dataset):
    """为数据集添加全局索引，用于样本加权跟踪"""

    def __init__(self, dataset, start_idx=0):
        self.dataset = dataset
        self.start_idx = start_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        global_idx = self.start_idx + idx
        return (*item, global_idx)


class GlareProgressiveLoss(nn.Module):
    """
    渐进式置信度惩罚损失

    在反光区域：
    - 惩罚强度与预测置信度成正比
    - weight = 1.0 + (penalty_factor - 1) * pred_prob^gamma
    """

    def __init__(self, penalty_factor=3.0, gamma=1.0, smooth=1e-5):
        super().__init__()
        self.penalty_factor = penalty_factor
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target, glare_mask=None):
        pred_prob = torch.sigmoid(pred)
        B, C, H, W = pred.shape

        if glare_mask is None:
            return self._compute_bce_dice(pred, pred_prob, target, weight=None)

        if glare_mask.shape[-2:] != pred.shape[-2:]:
            glare_mask = F.interpolate(
                glare_mask, pred.shape[-2:], mode='nearest')

        if glare_mask.shape[1] == 1:
            glare_mask = glare_mask.expand(-1, C, -1, -1)

        progressive_penalty = (pred_prob.detach() **
                               self.gamma) * (self.penalty_factor - 1.0)
        weight = 1.0 + glare_mask * progressive_penalty

        return self._compute_bce_dice(pred, pred_prob, target, weight=weight)

    def _compute_bce_dice(self, pred, pred_prob, target, weight=None):
        B, C, H, W = pred.shape

        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        if weight is not None:
            bce = bce * weight
        bce_loss = bce.mean()

        dice_loss = 0.0
        for c in range(C):
            p = pred_prob[:, c].reshape(B, -1)
            t = target[:, c].reshape(B, -1)

            if weight is not None:
                w = weight[:, c].reshape(B, -1)
                intersection = (p * t * w).sum(dim=1)
                union = (p * w).sum(dim=1) + (t * w).sum(dim=1)
            else:
                intersection = (p * t).sum(dim=1)
                union = p.sum(dim=1) + t.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1.0 - dice).mean()

        dice_loss /= C

        return 0.5 * bce_loss + 0.5 * dice_loss


def compute_per_sample_loss(logits, targets, criterion, glare_mask=None, use_glare_loss=False):
    """计算每个样本的独立损失（不 reduce）"""
    B = logits.size(0)
    losses = []
    for b in range(B):
        if use_glare_loss and glare_mask is not None:
            loss = criterion(logits[b:b+1], targets[b:b+1],
                             glare_mask=glare_mask[b:b+1])
        else:
            loss = criterion(logits[b:b+1], targets[b:b+1])
        losses.append(loss)
    return torch.stack(losses)


def parse_pos_weight(pos_weight_str, num_classes):
    """解析命令行传入的 pos_weight"""
    if pos_weight_str is None:
        return None

    parts = [p.strip() for p in str(pos_weight_str).split(',') if p.strip()]
    if len(parts) == 0:
        return None

    values = [float(p) for p in parts]

    if len(values) == 1:
        return torch.tensor(values, dtype=torch.float32)

    if len(values) != num_classes:
        raise ValueError(
            f"--pos_weight 需要提供 1 个值或与 num_classes={num_classes} 相同数量的值，"
            f"当前收到 {len(values)} 个: {values}"
        )

    return torch.tensor(values, dtype=torch.float32)


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=None, mixed_precision="no", num_classes=3,
                    thresh=0.5, vis_dir=None, epoch=0,
                    expert_types=None, return_gate_weights=True,
                    use_glare_loss=False, sample_weighting=None):
    """训练一个 epoch (联合训练版)"""
    model.train()
    total_loss = 0.0
    total_weighted_loss = 0.0
    first_batch_logged = False

    metrics = {}
    for c in range(num_classes):
        metrics[f'dice_class{c}'] = []
        metrics[f'iou_class{c}'] = []

    backbone_weights_sum = None
    decoder_weights_sum = None
    gate_count = 0

    all_sample_weights = []
    all_per_sample_losses = []

    pbar = tqdm(train_loader, desc=f"[Joint Train E{epoch}]")
    for step, batch in enumerate(pbar):
        # 解析 batch
        sample_indices = None

        if sample_weighting is not None:
            sample_indices = batch[-1]
            if use_glare_loss:
                inputs = batch[0]
                targets = batch[1]
                glare_mask = batch[2].to(device)
            else:
                inputs = batch[0]
                targets = batch[1]
                glare_mask = None
        else:
            if use_glare_loss and len(batch) >= 4:
                inputs, targets, glare_mask = batch[0], batch[1], batch[2]
                glare_mask = glare_mask.to(device)
            else:
                inputs, targets = batch[0], batch[1]
                glare_mask = None

        inputs = inputs.to(device)
        targets = targets.to(device)
        if sample_indices is not None:
            sample_indices = sample_indices.to(device)

        optimizer.zero_grad()

        can_return_weights = return_gate_weights and (
            hasattr(model, 'backbone_gate') or hasattr(model, 'decoder_moe') or hasattr(model, 'decoder_gates')
        )

        # === Forward + Loss ===
        if mixed_precision != "no":
            amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                if can_return_weights:
                    logits, weights = model(inputs, return_weights=True)
                else:
                    logits = model(inputs)
                    weights = None

                loss = _compute_loss(
                    logits, targets, criterion, glare_mask, use_glare_loss,
                    sample_weighting, sample_indices, weights,
                    all_sample_weights, all_per_sample_losses
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            if can_return_weights:
                logits, weights = model(inputs, return_weights=True)
            else:
                logits = model(inputs)
                weights = None

            loss = _compute_loss(
                logits, targets, criterion, glare_mask, use_glare_loss,
                sample_weighting, sample_indices, weights,
                all_sample_weights, all_per_sample_losses
            )

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if sample_weighting is not None:
            total_weighted_loss += loss.item()

        # 累计门控权重
        if weights is not None:
            if isinstance(weights, dict) and 'backbone' in weights:
                bb_w = weights['backbone'].detach()
                if backbone_weights_sum is None:
                    backbone_weights_sum = bb_w.sum(dim=1) if bb_w.dim() == 3 else bb_w.sum(dim=0)
                else:
                    backbone_weights_sum += (bb_w.sum(dim=1) if bb_w.dim() == 3 else bb_w.sum(dim=0))

                if isinstance(weights['decoder'], dict):
                    if decoder_weights_sum is None:
                        decoder_weights_sum = {}
                    for c in weights['decoder']:
                        if weights['decoder'][c] is not None:
                            if c in decoder_weights_sum:
                                decoder_weights_sum[c] += weights['decoder'][c].detach().sum(dim=0)
                            else:
                                decoder_weights_sum[c] = weights['decoder'][c].detach().sum(dim=0)
                elif weights['decoder'] is not None:
                    if decoder_weights_sum is None:
                        decoder_weights_sum = weights['decoder'].detach().sum(dim=0)
                    else:
                        decoder_weights_sum += weights['decoder'].detach().sum(dim=0)
            gate_count += inputs.size(0)

        with torch.no_grad():
            dice_vals = dice_score(logits, targets, thresh=thresh)
            iou_vals = iou_score(logits, targets, thresh=thresh)
            for c in range(num_classes):
                metrics[f'dice_class{c}'].append(dice_vals[:, c].mean().item())
                metrics[f'iou_class{c}'].append(iou_vals[:, c].mean().item())

        avg_dice = np.mean(
            [metrics[f'dice_class{c}'][-1] for c in range(num_classes)])
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{avg_dice:.4f}")

        if (not first_batch_logged) and vis_dir is not None:
            save_train_visuals(epoch, inputs, logits, targets,
                               out_dir=vis_dir, max_save=8, thr=thresh)
            first_batch_logged = True

    # 汇总 metrics
    avg_metrics = {}
    for k, v in metrics.items():
        avg_metrics[k] = float(np.mean(v))

    avg_metrics['dice_mean'] = np.mean(
        [avg_metrics[f'dice_class{c}'] for c in range(num_classes)])
    avg_metrics['iou_mean'] = np.mean(
        [avg_metrics[f'iou_class{c}'] for c in range(num_classes)])
    avg_metrics['loss'] = total_loss / max(1, len(train_loader))

    # 样本权重统计
    if sample_weighting is not None and len(all_sample_weights) > 0:
        avg_metrics['weighted_loss'] = total_weighted_loss / max(1, len(train_loader))
        avg_metrics['weight_mean'] = float(np.mean(all_sample_weights))
        avg_metrics['weight_std'] = float(np.std(all_sample_weights))
        avg_metrics['weight_min'] = float(np.min(all_sample_weights))
        avg_metrics['weight_max'] = float(np.max(all_sample_weights))
        print(f"[Train E{epoch}] loss={avg_metrics['loss']:.4f}  weighted_loss={avg_metrics['weighted_loss']:.4f}  "
              f"dice={avg_metrics['dice_mean']:.4f}")
        print(f"    [Weights] mean={avg_metrics['weight_mean']:.3f}, std={avg_metrics['weight_std']:.3f}, "
              f"range=[{avg_metrics['weight_min']:.3f}, {avg_metrics['weight_max']:.3f}]")
    else:
        print(f"[Train E{epoch}] loss={avg_metrics['loss']:.4f}  "
              f"dice={avg_metrics['dice_mean']:.4f}  iou={avg_metrics['iou_mean']:.4f}")

    # 打印门控权重
    if backbone_weights_sum is not None and gate_count > 0:
        avg_backbone = backbone_weights_sum / gate_count
        if avg_backbone.dim() == 2:
            for c in range(avg_backbone.shape[0]):
                print(
                    f"[Train E{epoch}] Class{c} backbone: SAT={avg_backbone[c, 0].item():.3f} LVD={avg_backbone[c, 1].item():.3f}")
        else:
            print(
                f"[Train E{epoch}] Backbone gate: SAT={avg_backbone[0].item():.3f} LVD={avg_backbone[1].item():.3f}")

        if isinstance(decoder_weights_sum, dict) and decoder_weights_sum:
            for c in decoder_weights_sum:
                avg_dec = decoder_weights_sum[c] / gate_count
                w_str = " ".join(
                    [f"{exp}={avg_dec[i].item():.3f}" for i, exp in enumerate(expert_types)])
                print(f"[Train E{epoch}] Class{c} decoder: {w_str}")
        elif decoder_weights_sum is not None and not isinstance(decoder_weights_sum, dict):
            avg_dec = decoder_weights_sum / gate_count
            w_str = " ".join(
                [f"{exp}={avg_dec[i].item():.3f}" for i, exp in enumerate(expert_types)])
            print(f"[Train E{epoch}] Decoder gate: {w_str}")

    gate_weights = None
    if gate_count > 0:
        gate_weights = {
            'backbone': backbone_weights_sum / gate_count if backbone_weights_sum is not None else None,
            'decoder': {c: decoder_weights_sum[c] / gate_count for c in decoder_weights_sum}
            if isinstance(decoder_weights_sum, dict) else
            (decoder_weights_sum / gate_count if decoder_weights_sum is not None else None)
        }

    return avg_metrics, gate_weights


def _compute_loss(logits, targets, criterion, glare_mask, use_glare_loss,
                  sample_weighting, sample_indices, weights,
                  all_sample_weights, all_per_sample_losses):
    """统一的损失计算逻辑（避免 AMP / 非 AMP 代码重复）"""
    if sample_weighting is not None:
        if isinstance(sample_weighting, ClassAwareSampleWeighting):
            per_class_loss = compute_per_class_loss(logits, targets)
            per_sample_loss = per_class_loss.mean(dim=1)
            gate_w = weights.get('backbone', None) if isinstance(weights, dict) else None
            sw = sample_weighting.get_weights(sample_indices, gate_weights=gate_w)
            sample_weighting.update_losses(sample_indices, per_class_loss)
            loss = (per_class_loss * sw).mean()
            all_sample_weights.extend(sw.mean(dim=1).detach().cpu().tolist())
        else:
            per_sample_loss = compute_per_sample_loss(
                logits, targets, criterion, glare_mask, use_glare_loss)
            if hasattr(sample_weighting, 'get_weights') and sample_indices is not None:
                sw = sample_weighting.get_weights(sample_indices)
                sample_weighting.update_losses(sample_indices, per_sample_loss)
            elif hasattr(sample_weighting, 'compute_weights'):
                sw = sample_weighting.compute_weights(logits, targets)
            else:
                sw = torch.ones_like(per_sample_loss)
            loss = (per_sample_loss * sw).mean()
            all_sample_weights.extend(sw.detach().cpu().tolist())
        all_per_sample_losses.extend(per_sample_loss.detach().cpu().tolist())
    else:
        if use_glare_loss:
            loss = criterion(logits, targets, glare_mask=glare_mask)
        else:
            loss = criterion(logits, targets)
    return loss


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, num_classes, thresh=0.5,
             expert_types=None, return_gate_weights=True, use_glare_loss=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0

    metrics = {}
    for c in range(num_classes):
        metrics[f'dice_class{c}'] = []
        metrics[f'iou_class{c}'] = []

    pbar = tqdm(val_loader, desc="[Eval]")
    for batch in pbar:
        if use_glare_loss and len(batch) >= 4:
            inputs, targets, glare_mask = batch[0], batch[1], batch[2]
            glare_mask = glare_mask.to(device)
        else:
            inputs, targets = batch[0], batch[1]
            glare_mask = None
        inputs = inputs.to(device)
        targets = targets.to(device)

        if return_gate_weights and hasattr(model, 'backbone_gate'):
            logits, _ = model(inputs, return_weights=True)
        else:
            logits = model(inputs)

        if use_glare_loss:
            loss = criterion(logits, targets, glare_mask=glare_mask)
        else:
            loss = criterion(logits, targets)
        total_loss += loss.item()

        dice_vals = dice_score(logits, targets, thresh=thresh)
        iou_vals = iou_score(logits, targets, thresh=thresh)

        B = inputs.size(0)
        for b in range(B):
            for c in range(num_classes):
                metrics[f'dice_class{c}'].append(dice_vals[b, c].item())
                metrics[f'iou_class{c}'].append(iou_vals[b, c].item())

        avg_dice = np.mean([dice_vals[:, c].mean().item()
                           for c in range(num_classes)])
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{avg_dice:.4f}")

    avg_metrics = {}
    for c in range(num_classes):
        avg_metrics[f'dice_class{c}'] = float(np.mean(metrics[f'dice_class{c}']))
        avg_metrics[f'iou_class{c}'] = float(np.mean(metrics[f'iou_class{c}']))

    avg_metrics['dice_mean'] = np.mean(
        [avg_metrics[f'dice_class{c}'] for c in range(num_classes)])
    avg_metrics['iou_mean'] = np.mean(
        [avg_metrics[f'iou_class{c}'] for c in range(num_classes)])
    avg_metrics['loss'] = total_loss / max(1, len(val_loader))

    print(f"[Eval] loss={avg_metrics['loss']:.4f}  "
          f"dice={avg_metrics['dice_mean']:.4f}  iou={avg_metrics['iou_mean']:.4f}")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='HD-MoE 单阶段联合训练')

    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--num_classes", type=int, required=True, help="类别数")
    parser.add_argument("--input_h", type=int, default=1024, help="输入高度")
    parser.add_argument("--input_w", type=int, default=1024, help="输入宽度")

    # 模型参数
    parser.add_argument("--dino_ckpt_sat", type=str, required=True, help="SAT预训练权重")
    parser.add_argument("--dino_ckpt_lvd", type=str, required=True, help="LVD预训练权重")
    parser.add_argument("--dino_size", type=str, default="l", choices=["s", "b", "l"])
    parser.add_argument("--repo_dir", type=str, default="../dinov3")
    parser.add_argument("--channels", type=int, default=256, help="解码器通道数")

    # HD-MoE 参数
    parser.add_argument("--decoder_mode", type=str, default="shared_moe",
                        choices=["shared_moe", "multi_moe", "single"],
                        help="解码器模式: shared_moe=共享MoE, multi_moe=独立MoE, single=单解码器(消融用)")
    parser.add_argument("--separate_projects", action="store_true", default=False,
                        help="multi_moe模式下每类使用独立投影层")
    parser.add_argument("--single_decoder_type", type=str, default="dpt",
                        choices=["sam", "d2s", "dpt", "linear_attn"],
                        help="single模式下使用的解码器类型 (默认dpt)")
    parser.add_argument("--use_checkpoint", action="store_true", default=False,
                        help="是否使用 gradient checkpointing (降低显存)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gate/Decoder 学习率")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Backbone 学习率 (默认 lr * 0.1)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="冻结双 Backbone (仅训练 Gate + Decoder)")

    # 损失函数
    parser.add_argument("--loss_type", type=str, default="bce_dice",
                        choices=["bce", "bce_dice", "focal_dice", "glare_progressive"],
                        help="损失函数类型")
    parser.add_argument("--pos_weight", type=str, default=None,
                        help="BCE 正类权重")

    # 反光损失参数
    parser.add_argument("--glare_data_dir", type=str, default=None,
                        help="反光数据目录(使用glare_progressive损失时必须)")
    parser.add_argument("--glare_penalty", type=float, default=3.0,
                        help="反光区域最大惩罚倍数")
    parser.add_argument("--glare_gamma", type=float, default=1.0,
                        help="惩罚曲线陡峭程度")

    # 样本级加权参数
    parser.add_argument("--sample_weighting", type=str, default="none",
                        choices=["none", "loss_based", "focal", "curriculum", "class_aware"],
                        help="样本级加权策略")
    parser.add_argument("--focus_mode", type=str, default="hard",
                        choices=["hard", "easy", "balanced"])
    parser.add_argument("--sample_temp", type=float, default=1.0)
    parser.add_argument("--sample_warmup_epochs", type=int, default=10)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--gate_entropy_lambda", type=float, default=0.0,
                        help="[class_aware] 门控熵调制系数")

    # 保存参数
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--exp_name", type=str, default=None)

    # 混合精度
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])

    # Resume
    parser.add_argument("--resume", action="store_true", help="恢复训练")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Backbone 学习率默认为 lr * 0.1
    if args.backbone_lr is None:
        args.backbone_lr = args.lr * 0.1

    # 创建保存目录
    date_suffix = datetime.now().strftime("%m%d_%H%M")
    exp_name = args.exp_name or f"hd_moe_joint_{args.decoder_mode}_{date_suffix}"
    args.save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"[Config] HD-MoE 单阶段联合训练")
    print(f"[Config] Epochs: {args.epochs}")
    print(f"[Config] Decoder mode: {args.decoder_mode}")
    print(f"[Config] Separate projects: {args.separate_projects}")
    print(f"[Config] Gradient checkpointing: {args.use_checkpoint}")
    print(f"[Config] LR (Gate/Decoder): {args.lr}")
    print(f"[Config] LR (Backbone): {args.backbone_lr}")
    print(f"[Config] Freeze backbone: {args.freeze_backbone}")
    print(f"[Config] Save directory: {args.save_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = parse_pos_weight(args.pos_weight, args.num_classes)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        print(f"[Config] Using pos_weight: {pos_weight.detach().cpu().tolist()}")

    use_glare_loss = (args.loss_type == 'glare_progressive')

    # ========== 数据加载 ==========
    if use_glare_loss:
        if not GLARE_DATASET_AVAILABLE:
            raise ImportError("无法导入 MultiLabelGlareDatasetV2")
        if args.glare_data_dir is None:
            raise ValueError("使用 glare_progressive 损失时必须指定 --glare_data_dir")
        train_transform = GlareResizeAndNormalize(size=(args.input_h, args.input_w))
        val_transform = GlareResizeAndNormalize(size=(args.input_h, args.input_w))
        train_dataset = MultiLabelGlareDatasetV2(
            args.data_dir, args.glare_data_dir,
            split="train", transform=train_transform, num_classes=args.num_classes
        )
        val_dataset = MultiLabelGlareDatasetV2(
            args.data_dir, args.glare_data_dir,
            split="test", transform=val_transform, num_classes=args.num_classes
        )
        print(f"[Data] 使用反光数据集: glare_dir={args.glare_data_dir}")
    else:
        train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
        val_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
        train_dataset = MultiLabelDataset(
            args.data_dir, split="train", transform=train_transform)
        val_dataset = MultiLabelDataset(
            args.data_dir, split="test", transform=val_transform)

    # 样本加权 - IndexedDataset 包装
    use_sample_weighting = (args.sample_weighting != 'none')
    if use_sample_weighting:
        train_dataset = IndexedDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ========== 损失函数 ==========
    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args.loss_type == 'bce_dice':
        criterion = BCEDiceLoss()
    elif args.loss_type == 'focal_dice':
        criterion = FocalDiceLoss()
    elif args.loss_type == 'glare_progressive':
        criterion = GlareProgressiveLoss(
            penalty_factor=args.glare_penalty,
            gamma=args.glare_gamma
        )
    else:
        criterion = BCEDiceLoss()

    # ========== 构建 HD-MoE 模型 ==========
    expert_types = DEFAULT_EXPERTS

    model = build_hd_moe_model(
        num_classes=args.num_classes,
        dino_size=args.dino_size,
        dino_ckpt_1=args.dino_ckpt_sat,
        dino_ckpt_2=args.dino_ckpt_lvd,
        repo_dir=args.repo_dir,
        channels=args.channels,
        expert_types=expert_types,
        decoder_mode=args.decoder_mode,
        separate_projects=args.separate_projects,
        use_checkpoint=args.use_checkpoint,
        single_decoder_type=args.single_decoder_type,
    )

    # 可选冻结 Backbone
    if args.freeze_backbone:
        print("[Config] 冻结双 Backbone...")
        model.lock_backbone(expert_id=None)

    model = model.to(device)

    # 统计参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Trainable params: {trainable_params:,} / {total_params:,}")

    # ========== 差异化学习率优化器 ==========
    if not args.freeze_backbone:
        backbone_params = list(model.backbone_1.parameters()) + list(model.backbone_2.parameters())
        backbone_param_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids and p.requires_grad]

        param_groups = [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": args.backbone_lr, "name": "backbone"},
            {"params": other_params,
             "lr": args.lr, "name": "gate_decoder"},
        ]
        print(f"[Optim] Backbone params: {sum(p.numel() for p in backbone_params if p.requires_grad):,} (lr={args.backbone_lr})")
        print(f"[Optim] Gate/Decoder params: {sum(p.numel() for p in other_params):,} (lr={args.lr})")
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": args.lr},
        ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # 混合精度
    scaler = None
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    # ========== 样本加权 ==========
    sample_weighting = None
    if use_sample_weighting:
        total_samples = len(train_dataset)
        if args.sample_weighting == 'class_aware':
            if not CLASS_AWARE_WEIGHTING_AVAILABLE or ClassAwareSampleWeighting is None:
                raise ImportError("class_aware_weighting.py 不可用")
            sample_weighting = ClassAwareSampleWeighting(
                num_samples=total_samples,
                num_classes=args.num_classes,
                temperature=args.sample_temp,
                focus_mode=args.focus_mode,
                gate_entropy_lambda=args.gate_entropy_lambda,
            )
        else:
            if not SAMPLE_WEIGHTING_AVAILABLE:
                raise ImportError("sample_weighting.py 不可用")
            sample_weighting = create_sample_weighting(
                strategy=args.sample_weighting,
                num_samples=total_samples,
                temperature=args.sample_temp,
                focus_mode=args.focus_mode,
                total_epochs=args.epochs,
                warmup_epochs=args.sample_warmup_epochs,
                gamma=args.focal_gamma,
            )
        print(f"[Config] 样本加权: {args.sample_weighting}")

    # ========== 训练历史 ==========
    train_vis_dir = os.path.join(args.save_dir, "train_vis")
    ckpt_dir = os.path.join(args.save_dir, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    history = init_history(args.num_classes)
    # 门控权重历史
    for c in range(args.num_classes):
        history[f'gate_class{c}_sat'] = []
        history[f'gate_class{c}_lvd'] = []
    for exp in expert_types:
        history[f'gate_decoder_{exp}'] = []

    best_val_dice = -1.0
    prev_best_path = None
    start_epoch = 1

    # Resume
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    if args.resume and os.path.exists(latest_ckpt):
        print(f"[Resume] Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if 'history' in ckpt:
            history = ckpt['history']
        if 'best_val_dice' in ckpt:
            best_val_dice = ckpt['best_val_dice']
        print(f"[Resume] Resuming from epoch {start_epoch}")

    # ========== 训练循环 ==========
    multi_moe = args.decoder_mode == 'multi_moe'

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n>>> Epoch {epoch}/{args.epochs} (HD-MoE Joint)")

        # 课程学习
        if sample_weighting is not None and hasattr(sample_weighting, 'set_epoch'):
            sample_weighting.set_epoch(epoch)

        train_metrics, gate_weights = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, mixed_precision=args.mixed_precision,
            num_classes=args.num_classes, thresh=0.5,
            vis_dir=train_vis_dir, epoch=epoch,
            expert_types=expert_types, return_gate_weights=True,
            use_glare_loss=use_glare_loss,
            sample_weighting=sample_weighting
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            args.num_classes, thresh=0.5,
            expert_types=expert_types, return_gate_weights=True,
            use_glare_loss=use_glare_loss
        )

        # 记录历史
        history['epochs'].append(epoch)
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice_mean'].append(train_metrics['dice_mean'])
        history['val_dice_mean'].append(val_metrics['dice_mean'])
        history['train_iou_mean'].append(train_metrics['iou_mean'])
        history['val_iou_mean'].append(val_metrics['iou_mean'])

        for c in range(args.num_classes):
            history[f'train_dice_class{c}'].append(train_metrics[f'dice_class{c}'])
            history[f'train_iou_class{c}'].append(train_metrics[f'iou_class{c}'])
            history[f'val_dice_class{c}'].append(val_metrics[f'dice_class{c}'])
            history[f'val_iou_class{c}'].append(val_metrics[f'iou_class{c}'])

        # 记录门控权重
        if gate_weights is not None and gate_weights['backbone'] is not None:
            bb = gate_weights['backbone']
            if bb.dim() == 2:
                for c in range(bb.shape[0]):
                    history[f'gate_class{c}_sat'].append(float(bb[c, 0].item()))
                    history[f'gate_class{c}_lvd'].append(float(bb[c, 1].item()))
            else:
                if 'gate_backbone_sat' not in history:
                    history['gate_backbone_sat'] = []
                    history['gate_backbone_lvd'] = []
                history['gate_backbone_sat'].append(float(bb[0].item()))
                history['gate_backbone_lvd'].append(float(bb[1].item()))

            if not multi_moe and gate_weights['decoder'] is not None:
                dec = gate_weights['decoder']
                if isinstance(dec, torch.Tensor):
                    for i, exp in enumerate(expert_types):
                        history[f'gate_decoder_{exp}'].append(float(dec[i].item()))

        # 保存检查点
        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_val_dice": best_val_dice,
        }, latest_path)

        if val_metrics['dice_mean'] > best_val_dice:
            best_val_dice = val_metrics['dice_mean']
            if prev_best_path is not None and os.path.exists(prev_best_path):
                os.remove(prev_best_path)
            best_path = os.path.join(
                ckpt_dir, f"best_ep{epoch:03d}_dice{val_metrics['dice_mean']:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"[Save] New best: {best_path}")
            prev_best_path = best_path

        plot_training_curves(history, args.save_dir, args.num_classes)
        save_training_log(history, args.save_dir, args.num_classes,
                          [f'Class{i}' for i in range(args.num_classes)], vars(args))

    # 自动清理
    if AUTO_CLEANUP:
        print("[Cleanup] 正在清理 ckpts 和 train_vis 文件夹...")
        for root, dirs, files in os.walk(args.save_dir):
            for d in list(dirs):
                if d in ["ckpts", "train_vis"]:
                    dir_path = os.path.join(root, d)
                    try:
                        shutil.rmtree(dir_path)
                        print(f"  [Deleted] {dir_path}")
                    except Exception as e:
                        print(f"  [Error] Failed to delete {dir_path}: {e}")
        print("[Cleanup] 清理完成")

    print("\n" + "=" * 70)
    print(f"[Summary] Best Val Dice: {best_val_dice:.4f}")
    print(f"[Summary] Results saved to: {args.save_dir}")
    print("=" * 70)
    print("Training completed successfully")


if __name__ == "__main__":
    main()
