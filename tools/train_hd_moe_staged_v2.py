#!/usr/bin/env python3
# tools/train_hd_moe_staged_v2.py
"""
层级双MoE (HD-MoE) 三阶段训练脚本 V2

融合功能：
1. 反光抑制渐进式损失 (GlareProgressiveLoss)
2. 样本级自适应加权 (loss_based / focal / curriculum)

阶段 1: SAT 单分支微调 (Backbone + 单解码器)
阶段 2: LVD 单分支微调 (Backbone + 单解码器)
阶段 3: HD-MoE 联合训练 (冻结双Backbone, 训练门控+解码器MoE)
"""
from model_multienc_v2 import build_shared_moe_model, build_multi_moe_model
from dataset import MultiLabelDataset, ResizeAndNormalize
from modules.visualize import save_train_visuals, plot_training_curves, save_training_log, init_history
from modules.metrics import dice_score, iou_score
from modules.losses import BCEDiceLoss, FocalDiceLoss
import torch.nn.functional as F
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from datetime import datetime
import shutil  # Added by user

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 反光数据集(可选) - 需要使用其专用的 ResizeAndNormalize
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
        # 返回原始item加上全局索引
        return (*item, global_idx)


# ============== 全局配置 ==============
# 是否在训练完成后自动删除 ckpts 和 train_vis 文件夹 (节省空间)
AUTO_CLEANUP = False


# ============== 渐进式置信度惩罚损失 ==============
class GlareProgressiveLoss(nn.Module):
    """
    渐进式置信度惩罚损失

    在反光区域：
    - 惩罚强度与预测置信度成正比
    - weight = 1.0 + (penalty_factor - 1) * pred_prob^gamma

    Args:
        penalty_factor: 最大惩罚倍数 (默认 3.0)
        gamma: 惩罚曲线的陡峭程度 (默认 1.0, 线性; >1 更陡, <1 更平缓)
        smooth: Dice 计算的平滑因子
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

        # 渐进式权重：
        # 非反光区域: weight = 1.0
        # 反光区域: weight = 1.0 + (penalty_factor - 1) * pred_prob^gamma
        progressive_penalty = (pred_prob.detach() **
                               self.gamma) * (self.penalty_factor - 1.0)
        weight = 1.0 + glare_mask * progressive_penalty

        return self._compute_bce_dice(pred, pred_prob, target, weight=weight)

    def _compute_bce_dice(self, pred, pred_prob, target, weight=None):
        B, C, H, W = pred.shape

        # BCE Loss (逐像素加权)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        if weight is not None:
            bce = bce * weight
        bce_loss = bce.mean()

        # Dice Loss (加权版本)
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


# 默认专家
DEFAULT_EXPERTS = ['sam', 'd2s', 'dpt', 'linear_attn']


def print_stage_banner(stage, total_epochs, description):
    """打印阶段横幅"""
    print("\n" + "=" * 70)
    print(f" 阶段 {stage}: {description}")
    print(f" Epochs: {total_epochs}")
    print("=" * 70 + "\n")


def compute_per_sample_loss(logits, targets, criterion, glare_mask=None, use_glare_loss=False):
    """
    计算每个样本的独立损失（不 reduce）

    Args:
        logits: 模型输出 [B, C, H, W]
        targets: 目标 [B, C, H, W]
        criterion: 损失函数
        glare_mask: 反光掩码 [B, 1, H, W] 或 None
        use_glare_loss: 是否使用反光损失

    Returns:
        per_sample_loss: 每个样本的损失 [B]
    """
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


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=None, mixed_precision="no", num_classes=3,
                    thresh=0.5, vis_dir=None, epoch=0, stage=1,
                    multi_moe=False, expert_types=None, return_gate_weights=False,
                    use_glare_loss=False, sample_weighting=None):
    """
    训练一个 epoch

    Args:
        sample_weighting: 样本加权策略对象，为 None 时不使用加权
    """
    model.train()
    total_loss = 0.0
    total_weighted_loss = 0.0
    first_batch_logged = False

    metrics = {}
    for c in range(num_classes):
        metrics[f'dice_class{c}'] = []
        metrics[f'iou_class{c}'] = []

    # 门控权重累计
    backbone_weights_sum = None
    decoder_weights_sum = None
    gate_count = 0

    # 样本权重统计
    all_sample_weights = []
    all_per_sample_losses = []

    pbar = tqdm(train_loader, desc=f"[Stage{stage} Train E{epoch}]")
    for step, batch in enumerate(pbar):
        # 根据数据集格式解析 batch
        # 格式可能是: (img, mask, name) 或 (img, mask, glare, name) 或 ... + global_idx
        batch_len = len(batch)
        sample_indices = None

        # 解析batch - 支持 IndexedDataset
        if sample_weighting is not None:
            # IndexedDataset 会将 index 放在最后
            sample_indices = batch[-1]

            if use_glare_loss:
                # (img, mask, glare, name, ..., idx)
                if len(batch) < 4:
                    # 防御性检查
                    print(
                        f"[Warning] Batch length {len(batch)} < 4 with glare loss, unexpected!")

                inputs = batch[0]
                targets = batch[1]
                glare_mask = batch[2]
                glare_mask = glare_mask.to(device)
            else:
                # (img, mask, name, ..., idx)
                inputs = batch[0]
                targets = batch[1]
                glare_mask = None
        else:
            if use_glare_loss and batch_len >= 4:
                inputs, targets, glare_mask, _ = batch[0], batch[1], batch[2], batch[3]
                glare_mask = glare_mask.to(device)
            else:
                inputs, targets = batch[0], batch[1]
                glare_mask = None

        inputs = inputs.to(device)
        targets = targets.to(device)
        if sample_indices is not None:
            sample_indices = sample_indices.to(device)

        optimizer.zero_grad()

        # 判断是否有return_weights方法
        can_return_weights = return_gate_weights and hasattr(model, 'forward') and (
            hasattr(model, 'backbone_gate') or hasattr(
                model, 'decoder_moe') or hasattr(model, 'decoder_gates')
        )

        if mixed_precision != "no":
            amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                if can_return_weights:
                    logits, weights = model(inputs, return_weights=True)
                else:
                    logits = model(inputs)
                    weights = None

                # 样本加权损失计算
                if sample_weighting is not None:
                    if isinstance(sample_weighting, ClassAwareSampleWeighting):
                        # 类别感知加权：逐类别损失 + 逐类别权重
                        per_class_loss = compute_per_class_loss(
                            logits, targets)  # [B, C]
                        per_sample_loss = per_class_loss.mean(dim=1)  # [B]
                        gate_w = weights.get('backbone', None) if isinstance(
                            weights, dict) else None
                        sw = sample_weighting.get_weights(
                            sample_indices, gate_weights=gate_w)  # [B, C]
                        sample_weighting.update_losses(
                            sample_indices, per_class_loss)
                        loss = (per_class_loss * sw).mean()
                        all_sample_weights.extend(
                            sw.mean(dim=1).detach().cpu().tolist())
                    else:
                        per_sample_loss = compute_per_sample_loss(
                            logits, targets, criterion, glare_mask, use_glare_loss)
                        if hasattr(sample_weighting, 'get_weights') and sample_indices is not None:
                            sw = sample_weighting.get_weights(sample_indices)
                            sample_weighting.update_losses(
                                sample_indices, per_sample_loss)
                        elif hasattr(sample_weighting, 'compute_weights'):
                            sw = sample_weighting.compute_weights(
                                logits, targets)
                        else:
                            sw = torch.ones_like(per_sample_loss)
                        loss = (per_sample_loss * sw).mean()
                        all_sample_weights.extend(sw.detach().cpu().tolist())
                    all_per_sample_losses.extend(
                        per_sample_loss.detach().cpu().tolist())
                else:
                    if use_glare_loss:
                        loss = criterion(
                            logits, targets, glare_mask=glare_mask)
                    else:
                        loss = criterion(logits, targets)

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

            # 样本加权损失计算
            if sample_weighting is not None:
                if isinstance(sample_weighting, ClassAwareSampleWeighting):
                    # 类别感知加权：逐类别损失 + 逐类别权重
                    per_class_loss = compute_per_class_loss(
                        logits, targets)  # [B, C]
                    per_sample_loss = per_class_loss.mean(dim=1)  # [B]
                    gate_w = weights.get('backbone', None) if isinstance(
                        weights, dict) else None
                    sw = sample_weighting.get_weights(
                        sample_indices, gate_weights=gate_w)  # [B, C]
                    sample_weighting.update_losses(
                        sample_indices, per_class_loss)
                    loss = (per_class_loss * sw).mean()
                    all_sample_weights.extend(
                        sw.mean(dim=1).detach().cpu().tolist())
                else:
                    per_sample_loss = compute_per_sample_loss(
                        logits, targets, criterion, glare_mask, use_glare_loss)
                    if hasattr(sample_weighting, 'get_weights') and sample_indices is not None:
                        sw = sample_weighting.get_weights(sample_indices)
                        sample_weighting.update_losses(
                            sample_indices, per_sample_loss)
                    elif hasattr(sample_weighting, 'compute_weights'):
                        sw = sample_weighting.compute_weights(logits, targets)
                    else:
                        sw = torch.ones_like(per_sample_loss)
                    loss = (per_sample_loss * sw).mean()
                    all_sample_weights.extend(sw.detach().cpu().tolist())
                all_per_sample_losses.extend(
                    per_sample_loss.detach().cpu().tolist())
            else:
                if use_glare_loss:
                    loss = criterion(logits, targets, glare_mask=glare_mask)
                else:
                    loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if sample_weighting is not None:
            total_weighted_loss += loss.item()

        # 累计门控权重
        if weights is not None:
            # HD-MoE模型返回 {'backbone': ..., 'decoder': ...}
            # 单Backbone模型返回解码器权重张量
            if isinstance(weights, dict) and 'backbone' in weights:
                # HD-MoE模型
                if backbone_weights_sum is None:
                    # backbone权重形状: [num_classes, B, 2] (类别感知) 或 [B, 2] (全局)
                    bb_w = weights['backbone'].detach()
                    if bb_w.dim() == 3:
                        # 类别感知: [num_classes, B, 2] -> 对B求和 -> [num_classes, 2]
                        backbone_weights_sum = bb_w.sum(dim=1)
                    else:
                        # 全局: [B, 2] -> 对B求和 -> [2]
                        backbone_weights_sum = bb_w.sum(dim=0)
                    if isinstance(weights['decoder'], dict):
                        # 过滤掉None值（single模式下decoder权重为None）
                        decoder_weights_sum = {c: weights['decoder'][c].detach().sum(dim=0)
                                               for c in weights['decoder'] if weights['decoder'][c] is not None}
                    elif weights['decoder'] is not None:
                        decoder_weights_sum = weights['decoder'].detach().sum(
                            dim=0)
                else:
                    bb_w = weights['backbone'].detach()
                    if bb_w.dim() == 3:
                        backbone_weights_sum += bb_w.sum(dim=1)
                    else:
                        backbone_weights_sum += bb_w.sum(dim=0)
                    if isinstance(weights['decoder'], dict):
                        for c in weights['decoder']:
                            # 跳过None值
                            if weights['decoder'][c] is not None:
                                if c in decoder_weights_sum:
                                    decoder_weights_sum[c] += weights['decoder'][c].detach().sum(
                                        dim=0)
                                else:
                                    decoder_weights_sum[c] = weights['decoder'][c].detach().sum(
                                        dim=0)
                    elif weights['decoder'] is not None:
                        decoder_weights_sum += weights['decoder'].detach().sum(
                            dim=0)
            else:
                # 单Backbone模型（SharedDecoderMoE/MultiDecoderMoE）
                if decoder_weights_sum is None:
                    if isinstance(weights, dict):
                        decoder_weights_sum = {
                            c: weights[c].detach().sum(dim=0) for c in weights}
                    else:
                        decoder_weights_sum = weights.detach().sum(dim=0)
                else:
                    if isinstance(weights, dict):
                        for c in weights:
                            decoder_weights_sum[c] += weights[c].detach().sum(dim=0)
                    else:
                        decoder_weights_sum += weights.detach().sum(dim=0)
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
        avg_metrics['weighted_loss'] = total_weighted_loss / \
            max(1, len(train_loader))
        avg_metrics['weight_mean'] = float(np.mean(all_sample_weights))
        avg_metrics['weight_std'] = float(np.std(all_sample_weights))
        avg_metrics['weight_min'] = float(np.min(all_sample_weights))
        avg_metrics['weight_max'] = float(np.max(all_sample_weights))
        print(f"[Stage{stage} Train E{epoch}] loss={avg_metrics['loss']:.4f}  weighted_loss={avg_metrics['weighted_loss']:.4f}  "
              f"dice={avg_metrics['dice_mean']:.4f}")
        print(f"    [Weights] mean={avg_metrics['weight_mean']:.3f}, std={avg_metrics['weight_std']:.3f}, "
              f"range=[{avg_metrics['weight_min']:.3f}, {avg_metrics['weight_max']:.3f}]")
    else:
        print(f"[Stage{stage} Train E{epoch}] loss={avg_metrics['loss']:.4f}  "
              f"dice={avg_metrics['dice_mean']:.4f}  iou={avg_metrics['iou_mean']:.4f}")

    # 打印门控权重
    if backbone_weights_sum is not None and gate_count > 0:
        avg_backbone = backbone_weights_sum / gate_count
        # 检查形状：[num_classes, 2] (类别感知) 或 [2] (全局)
        if avg_backbone.dim() == 2:
            # 类别感知：分别打印每个类别的门控权重
            num_cls = avg_backbone.shape[0]
            for c in range(num_cls):
                print(
                    f"[Stage{stage} Train E{epoch}] Class{c} backbone: SAT={avg_backbone[c, 0].item():.3f} LVD={avg_backbone[c, 1].item():.3f}")
        else:
            # 全局门控
            print(
                f"[Stage{stage} Train E{epoch}] Backbone gate: SAT={avg_backbone[0].item():.3f} LVD={avg_backbone[1].item():.3f}")

        if isinstance(decoder_weights_sum, dict) and decoder_weights_sum:
            for c in decoder_weights_sum:
                avg_dec = decoder_weights_sum[c] / gate_count
                w_str = " ".join(
                    [f"{exp}={avg_dec[i].item():.3f}" for i, exp in enumerate(expert_types)])
                print(
                    f"[Stage{stage} Train E{epoch}] Class{c} decoder: {w_str}")
        elif decoder_weights_sum is not None and not isinstance(decoder_weights_sum, dict):
            avg_dec = decoder_weights_sum / gate_count
            w_str = " ".join(
                [f"{exp}={avg_dec[i].item():.3f}" for i, exp in enumerate(expert_types)])
            print(f"[Stage{stage} Train E{epoch}] Decoder gate: {w_str}")

    gate_weights = None
    if gate_count > 0:
        gate_weights = {
            'backbone': backbone_weights_sum / gate_count if backbone_weights_sum is not None else None,
            'decoder': {c: decoder_weights_sum[c] / gate_count for c in decoder_weights_sum}
            if isinstance(decoder_weights_sum, dict) else
            (decoder_weights_sum / gate_count if decoder_weights_sum is not None else None)
        }

    return avg_metrics, gate_weights


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, num_classes, thresh=0.5,
             stage=1, multi_moe=False, expert_types=None, return_gate_weights=False,
             use_glare_loss=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0

    metrics = {}
    for c in range(num_classes):
        metrics[f'dice_class{c}'] = []
        metrics[f'iou_class{c}'] = []

    pbar = tqdm(val_loader, desc=f"[Stage{stage} Eval]")
    for batch in pbar:
        # 根据数据集格式解析 batch
        if use_glare_loss and len(batch) >= 4:
            inputs, targets, glare_mask, _ = batch[0], batch[1], batch[2], batch[3]
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

        # 根据损失类型调用
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
        avg_metrics[f'dice_class{c}'] = float(
            np.mean(metrics[f'dice_class{c}']))
        avg_metrics[f'iou_class{c}'] = float(np.mean(metrics[f'iou_class{c}']))

    avg_metrics['dice_mean'] = np.mean(
        [avg_metrics[f'dice_class{c}'] for c in range(num_classes)])
    avg_metrics['iou_mean'] = np.mean(
        [avg_metrics[f'iou_class{c}'] for c in range(num_classes)])
    avg_metrics['loss'] = total_loss / max(1, len(val_loader))

    print(f"[Stage{stage} Eval] loss={avg_metrics['loss']:.4f}  "
          f"dice={avg_metrics['dice_mean']:.4f}  iou={avg_metrics['iou_mean']:.4f}")

    return avg_metrics


def run_stage_1_2(args, stage, dino_ckpt, save_name, device, train_loader, val_loader, criterion, use_glare_loss=False, sample_weighting=None):
    """运行阶段 1 或 2: 单分支微调"""
    desc = "SAT 分支微调" if stage == 1 else "LVD 分支微调"
    epochs = args.epochs_stage1 if stage == 1 else args.epochs_stage2
    print_stage_banner(stage, epochs, desc)

    # 构建单Backbone + 解码器模型
    if args.decoder_mode == 'multi_moe':
        model = build_multi_moe_model(
            args.num_classes,
            args.dino_size,
            dino_ckpt,
            args.repo_dir,
            channels=args.channels,
            separate_projects=args.separate_projects
        )
    elif args.decoder_mode == 'shared_moe':
        model = build_shared_moe_model(
            args.num_classes,
            args.dino_size,
            dino_ckpt,
            args.repo_dir,
            channels=args.channels
        )
    else:  # single 模式 - 使用标准 DPT 解码器
        from model import build_model
        model = build_model(
            num_classes=args.num_classes,
            dino_size=args.dino_size,
            dino_ckpt=dino_ckpt,
            repo_dir=args.repo_dir,
            use_checkpoint=False
        )
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 混合精度
    scaler = None
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    # 保存目录
    stage_dir = os.path.join(args.save_dir, f"stage{stage}_{save_name}")
    os.makedirs(stage_dir, exist_ok=True)
    train_vis_dir = os.path.join(stage_dir, "train_vis")
    ckpt_dir = os.path.join(stage_dir, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    history = init_history(args.num_classes)
    # 门控权重历史
    for exp in DEFAULT_EXPERTS:
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

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n>>> 阶段 {stage} - Epoch {epoch}/{epochs}")

        # 课程学习更新 epoch
        if sample_weighting is not None and hasattr(sample_weighting, 'set_epoch'):
            sample_weighting.set_epoch(epoch)

        multi_moe = args.decoder_mode == 'multi_moe'
        train_metrics, gate_weights = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, mixed_precision=args.mixed_precision,
            num_classes=args.num_classes, thresh=0.5,
            vis_dir=train_vis_dir, epoch=epoch, stage=stage,
            multi_moe=multi_moe, expert_types=DEFAULT_EXPERTS,
            return_gate_weights=True,
            use_glare_loss=use_glare_loss,
            sample_weighting=sample_weighting
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            args.num_classes, thresh=0.5, stage=stage,
            multi_moe=multi_moe, expert_types=DEFAULT_EXPERTS,
            return_gate_weights=True,
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
            history[f'train_dice_class{c}'].append(
                train_metrics[f'dice_class{c}'])
            history[f'train_iou_class{c}'].append(
                train_metrics[f'iou_class{c}'])
            history[f'val_dice_class{c}'].append(val_metrics[f'dice_class{c}'])
            history[f'val_iou_class{c}'].append(val_metrics[f'iou_class{c}'])

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

        plot_training_curves(history, stage_dir, args.num_classes)
        save_training_log(history, stage_dir, args.num_classes,
                          [f'Class{i}' for i in range(args.num_classes)], vars(args))

    # 保存 backbone 权重
    backbone_path = os.path.join(ckpt_dir, "backbone_final.pth")
    # SharedDecoderMoE/MultiDecoderMoE 的 backbone 属性名
    if hasattr(model, 'backbone'):
        torch.save(model.backbone.state_dict(), backbone_path)
    else:
        # 需要查找正确的backbone属性
        print(f"[Warning] Cannot find backbone attribute, saving full model")
        torch.save(model.state_dict(), backbone_path)
    print(f"[Stage{stage}] Backbone saved to: {backbone_path}")

    return backbone_path, best_val_dice


def run_stage_3(args, device, train_loader, val_loader, criterion,
                sat_backbone_path, lvd_backbone_path, use_glare_loss=False, sample_weighting=None):
    """运行阶段 3: HD-MoE 联合训练"""
    from modules.hd_moe_model import build_hd_moe_model

    print_stage_banner(3, args.epochs_stage3, "HD-MoE 联合训练 (双Backbone冻结)")

    expert_types = ['sam', 'd2s', 'dpt', 'linear_attn']
    multi_moe = args.decoder_mode == 'multi_moe'

    # 构建 HD-MoE 模型
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
        use_checkpoint=False,
        single_decoder_type=args.single_decoder_type
    )

    # 加载微调后的 backbone 权重
    print(
        f"[Stage 3] Loading finetuned SAT backbone from: {sat_backbone_path}")
    sat_state = torch.load(sat_backbone_path, map_location='cpu')
    model.backbone_1.load_state_dict(sat_state)

    print(
        f"[Stage 3] Loading finetuned LVD backbone from: {lvd_backbone_path}")
    lvd_state = torch.load(lvd_backbone_path, map_location='cpu')
    model.backbone_2.load_state_dict(lvd_state)

    # 冻结两个 backbone
    print("[Stage 3] Freezing both backbones...")
    model.lock_backbone(expert_id=None)

    model = model.to(device)

    # 统计参数
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Stage 3] Trainable params: {trainable_params:,} / {total_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # 混合精度
    scaler = None
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    # 保存目录
    stage_dir = os.path.join(args.save_dir, "stage3_hd_moe")
    os.makedirs(stage_dir, exist_ok=True)
    train_vis_dir = os.path.join(stage_dir, "train_vis")
    ckpt_dir = os.path.join(stage_dir, "ckpts")
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    history = init_history(args.num_classes)
    # 门控权重历史 (类别感知)
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

    for epoch in range(start_epoch, args.epochs_stage3 + 1):
        print(f"\n>>> 阶段 3 - Epoch {epoch}/{args.epochs_stage3} (HD-MoE)")

        # 课程学习更新 epoch
        if sample_weighting is not None and hasattr(sample_weighting, 'set_epoch'):
            sample_weighting.set_epoch(epoch)

        train_metrics, gate_weights = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler, mixed_precision=args.mixed_precision,
            num_classes=args.num_classes, thresh=0.5,
            vis_dir=train_vis_dir, epoch=epoch, stage=3,
            multi_moe=multi_moe, expert_types=expert_types,
            return_gate_weights=True,
            use_glare_loss=use_glare_loss,
            sample_weighting=sample_weighting
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device,
            args.num_classes, thresh=0.5, stage=3,
            multi_moe=multi_moe, expert_types=expert_types,
            return_gate_weights=True,
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
            history[f'train_dice_class{c}'].append(
                train_metrics[f'dice_class{c}'])
            history[f'train_iou_class{c}'].append(
                train_metrics[f'iou_class{c}'])
            history[f'val_dice_class{c}'].append(val_metrics[f'dice_class{c}'])
            history[f'val_iou_class{c}'].append(val_metrics[f'iou_class{c}'])

        # 记录门控权重
        if gate_weights is not None and gate_weights['backbone'] is not None:
            bb = gate_weights['backbone']
            if bb.dim() == 2:
                # 类别感知: [num_classes, 2]
                for c in range(bb.shape[0]):
                    history[f'gate_class{c}_sat'].append(
                        float(bb[c, 0].item()))
                    history[f'gate_class{c}_lvd'].append(
                        float(bb[c, 1].item()))
            else:
                # 全局: [2] (兼容旧模型)
                if 'gate_backbone_sat' not in history:
                    history['gate_backbone_sat'] = []
                    history['gate_backbone_lvd'] = []
                history['gate_backbone_sat'].append(float(bb[0].item()))
                history['gate_backbone_lvd'].append(float(bb[1].item()))

            if not multi_moe and gate_weights['decoder'] is not None:
                dec = gate_weights['decoder']
                # 仅当 decoder 权重是张量时记录（全局 shared_moe 模式）
                # 类别感知模式下 decoder 是 dict，跳过此处记录
                if isinstance(dec, torch.Tensor):
                    for i, exp in enumerate(expert_types):
                        history[f'gate_decoder_{exp}'].append(
                            float(dec[i].item()))

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

        plot_training_curves(history, stage_dir, args.num_classes)
        save_training_log(history, stage_dir, args.num_classes,
                          [f'Class{i}' for i in range(args.num_classes)], vars(args))

    return best_val_dice


def parse_pos_weight(pos_weight_str, num_classes):
    """解析命令行传入的 pos_weight，单类任务支持一个浮点数。"""
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


def main():
    parser = argparse.ArgumentParser(description='HD-MoE 三阶段训练')

    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="数据集目录")
    parser.add_argument("--num_classes", type=int, required=True, help="类别数")
    parser.add_argument("--input_h", type=int, default=1024, help="输入高度")
    parser.add_argument("--input_w", type=int, default=1024, help="输入宽度")

    # 模型参数
    parser.add_argument("--dino_ckpt_sat", type=str,
                        default=None, help="SAT预训练权重 (单分支模式可选)")
    parser.add_argument("--dino_ckpt_lvd", type=str,
                        default=None, help="LVD预训练权重 (单分支模式可选)")
    parser.add_argument("--dino_size", type=str,
                        default="l", choices=["s", "b", "l"])
    parser.add_argument("--repo_dir", type=str, default="../dinov3")
    parser.add_argument("--channels", type=int, default=256, help="解码器通道数")
    parser.add_argument("--decoder_type", type=str,
                        default="dpt", help="Stage1/2解码器类型")

    # HD-MoE 参数（Stage 3）
    parser.add_argument("--decoder_mode", type=str, default="shared_moe",
                        choices=["shared_moe", "multi_moe", "single"],
                        help="解码器模式: shared_moe=独立MoE, multi_moe=多头MoE, single=单解码器(消融用)")
    parser.add_argument("--separate_projects", action="store_true", default=False,
                        help="multi_moe模式下每类使用独立投影层")
    parser.add_argument("--single_decoder_type", type=str, default="dpt",
                        choices=["sam", "d2s", "dpt", "linear_attn"],
                        help="single模式下使用的解码器类型 (默认dpt)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=None,
                        help="全局训练轮数 (设置后将覆盖所有阶段的epochs)")
    parser.add_argument("--epochs_stage1", type=int,
                        default=30, help="阶段1训练轮数")
    parser.add_argument("--epochs_stage2", type=int,
                        default=30, help="阶段2训练轮数")
    parser.add_argument("--epochs_stage3", type=int,
                        default=50, help="阶段3训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # 损失函数
    parser.add_argument("--loss_type", type=str, default="bce_dice",
                        choices=["bce", "bce_dice",
                                 "focal_dice", "glare_progressive"],
                        help="损失函数类型 (选择glare_progressive需要提供--glare_data_dir)")
    parser.add_argument("--pos_weight", type=str, default=None,
                        help="BCE 正类权重，单类任务传单个值即可，如 --pos_weight 3.0")

    # 反光损失参数
    parser.add_argument("--glare_data_dir", type=str, default=None,
                        help="反光数据目录(使用glare_progressive损失时必须)")
    parser.add_argument("--glare_penalty", type=float, default=3.0,
                        help="反光区域最大惩罚倍数")
    parser.add_argument("--glare_gamma", type=float, default=1.0,
                        help="惩罚曲线陡峭程度 (1.0=线性, >1更陡, <1更平缓)")
    parser.add_argument("--glare_loss_stages", type=str, default="auto",
                        help="在哪些阶段使用反光损失 (如: '3' 或 '1,2,3' 或 'auto' 自动匹配运行阶段)")

    # 样本级加权参数
    parser.add_argument("--sample_weighting", type=str, default="none",
                        choices=["none", "loss_based", "focal",
                                 "curriculum", "class_aware"],
                        help="样本级加权策略: none/loss_based/focal/curriculum/class_aware")
    parser.add_argument("--sample_weight_stages", type=str, default="auto",
                        help="在哪些阶段使用样本加权 (如: '3' 或 '1,2,3' 或 'auto' 自动匹配运行阶段)")
    parser.add_argument("--focus_mode", type=str, default="hard",
                        choices=["hard", "easy", "balanced"],
                        help="样本关注模式: hard=难例优先, easy=简单优先, balanced=平衡")
    parser.add_argument("--sample_temp", type=float, default=1.0,
                        help="样本权重温度参数 (越大权重越均匀)")
    parser.add_argument("--sample_warmup_epochs", type=int, default=10,
                        help="课程学习预热轮数")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal样本加权的gamma参数")
    parser.add_argument("--gate_entropy_lambda", type=float, default=0.0,
                        help="[class_aware] 门控熵调制系数 (0=不使用, >0时利用MoE门控不确定性)")

    # 保存参数
    parser.add_argument("--save_dir", type=str, default="./runs")
    parser.add_argument("--exp_name", type=str, default=None)

    # 混合精度
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])

    # 阶段控制
    parser.add_argument("--stage", type=int, default=0,
                        help="运行指定阶段(1/2/3)，0表示全部")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--sat_backbone_path", type=str, default=None,
                        help="直接指定SAT backbone路径（跳过Stage1）")
    parser.add_argument("--lvd_backbone_path", type=str, default=None,
                        help="直接指定LVD backbone路径（跳过Stage2）")

    args = parser.parse_args()

    # 权重校验
    if args.dino_ckpt_sat is None and args.dino_ckpt_lvd is None:
        parser.error("必须至少提供 --dino_ckpt_sat 或 --dino_ckpt_lvd 其中之一")

    # 自动调整运行阶段
    run_stages = []
    if args.stage == 0:
        if args.dino_ckpt_sat is not None:
            run_stages.append(1)
        if args.dino_ckpt_lvd is not None:
            run_stages.append(2)
        if args.dino_ckpt_sat is not None and args.dino_ckpt_lvd is not None:
            run_stages.append(3)
        print(f"[Config] 自动检测运行阶段: {run_stages}")
    else:
        run_stages = [args.stage]
        # 检查依赖
        if args.stage == 1 and args.dino_ckpt_sat is None:
            parser.error("运行 Stage 1 需要 --dino_ckpt_sat")
        if args.stage == 2 and args.dino_ckpt_lvd is None:
            parser.error("运行 Stage 2 需要 --dino_ckpt_lvd")
        if args.stage == 3 and (args.dino_ckpt_sat is None or args.dino_ckpt_lvd is None):
            # Stage 3 需要双流，除非只做推理但这里是训练脚本
            parser.error("运行 Stage 3 (HD-MoE) 需要同时提供 SAT 和 LVD 权重")

    # 处理全局 Epochs 覆盖
    if args.epochs is not None:
        print(f"[Config] 使用全局 Epochs: {args.epochs}, 将覆盖各阶段独立设置")
        args.epochs_stage1 = args.epochs
        args.epochs_stage2 = args.epochs
        args.epochs_stage3 = args.epochs

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建保存目录
    date_suffix = datetime.now().strftime("%m%d_%H%M")
    exp_name = args.exp_name or f"hd_moe_{args.decoder_mode}_{date_suffix}"
    args.save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"[Config] HD-MoE 三阶段训练")
    print(f"[Config] Stage 1 epochs: {args.epochs_stage1}")
    print(f"[Config] Stage 2 epochs: {args.epochs_stage2}")
    print(f"[Config] Stage 3 epochs: {args.epochs_stage3}")
    print(f"[Config] Decoder mode: {args.decoder_mode}")
    print(f"[Config] Separate projects: {args.separate_projects}")
    print(f"[Config] Save directory: {args.save_dir}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_weight = parse_pos_weight(args.pos_weight, args.num_classes)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        print(
            f"[Config] Using pos_weight: {pos_weight.detach().cpu().tolist()}")

    # 是否使用反光损失
    use_glare_loss = (args.loss_type == 'glare_progressive')

    # 解析哪些阶段使用样本加权
    sample_weight_stages = set()
    if args.sample_weighting != 'none':
        if args.sample_weight_stages == 'auto':
            sample_weight_stages = set(run_stages)
            print(f"[Config] 样本加权阶段自动设为: {sorted(sample_weight_stages)}")
        else:
            user_sample_stages = set(int(s.strip())
                                     for s in args.sample_weight_stages.split(','))
            sample_weight_stages = user_sample_stages & set(run_stages)
            if sample_weight_stages != user_sample_stages:
                print(
                    f"[Config] 样本加权阶段已自动调整: {sorted(user_sample_stages)} -> {sorted(sample_weight_stages)}")
            else:
                print(f"[Config] 样本加权启用阶段: {sorted(sample_weight_stages)}")

    # 解析哪些阶段使用反光损失
    glare_loss_stages = set()
    if use_glare_loss:
        if args.glare_loss_stages == 'auto':
            glare_loss_stages = set(run_stages)
            print(f"[Config] 反光损失阶段自动设为: {sorted(glare_loss_stages)}")
        else:
            user_glare_stages = set(int(s.strip())
                                    for s in args.glare_loss_stages.split(','))
            glare_loss_stages = user_glare_stages & set(run_stages)
            if glare_loss_stages != user_glare_stages:
                print(
                    f"[Config] 反光损失阶段已自动调整: {sorted(user_glare_stages)} -> {sorted(glare_loss_stages)}")
            else:
                print(f"[Config] 反光损失启用阶段: {sorted(glare_loss_stages)}")

    # 数据加载 - 反光数据集使用专用变换
    if use_glare_loss:
        if not GLARE_DATASET_AVAILABLE:
            raise ImportError(
                "无法导入 MultiLabelGlareDatasetV2，请检查 dataset_glare_v2.py 是否存在")
        if args.glare_data_dir is None:
            raise ValueError("使用 glare_progressive 损失时必须指定 --glare_data_dir")
        # 使用支持 glare_mask 的变换
        train_transform = GlareResizeAndNormalize(
            size=(args.input_h, args.input_w))
        val_transform = GlareResizeAndNormalize(
            size=(args.input_h, args.input_w))
        train_dataset = MultiLabelGlareDatasetV2(
            args.data_dir, args.glare_data_dir,
            split="train", transform=train_transform, num_classes=args.num_classes
        )
        val_dataset = MultiLabelGlareDatasetV2(
            args.data_dir, args.glare_data_dir,
            split="test", transform=val_transform, num_classes=args.num_classes
        )
        print(f"[Data] 使用反光数据集: glare_dir={args.glare_data_dir}")
        print(
            f"[Config] Glare penalty: {args.glare_penalty}, gamma: {args.glare_gamma}")
    else:
        train_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
        val_transform = ResizeAndNormalize(size=(args.input_h, args.input_w))
        train_dataset = MultiLabelDataset(
            args.data_dir, split="train", transform=train_transform)
        val_dataset = MultiLabelDataset(
            args.data_dir, split="test", transform=val_transform)

    # 如果有任何阶段使用了样本加权，使用 IndexedDataset 包装
    if len(sample_weight_stages) > 0:
        train_dataset = IndexedDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 损失函数
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

    # 为不同阶段创建损失函数（非反光阶段使用标准损失）
    criterion_standard = criterion
    criterion_glare = None
    if use_glare_loss:
        criterion_glare = GlareProgressiveLoss(
            penalty_factor=args.glare_penalty,
            gamma=args.glare_gamma
        )

    def create_stage_sample_weighting(stage, total_epochs):
        if args.sample_weighting == 'none':
            return None

        if stage not in sample_weight_stages:
            return None

        total_samples = len(train_dataset)

        if args.sample_weighting == 'class_aware':
            if (not CLASS_AWARE_WEIGHTING_AVAILABLE) or (ClassAwareSampleWeighting is None):
                raise ImportError(
                    "class_aware_weighting.py 不可用，无法启用 class_aware 样本加权")

            return ClassAwareSampleWeighting(
                num_samples=total_samples,
                num_classes=args.num_classes,
                temperature=args.sample_temp,
                focus_mode=args.focus_mode,
                gate_entropy_lambda=args.gate_entropy_lambda,
            )

        if not SAMPLE_WEIGHTING_AVAILABLE:
            raise ImportError("sample_weighting.py 不可用，无法启用样本加权")

        return create_sample_weighting(
            strategy=args.sample_weighting,
            num_samples=total_samples,
            temperature=args.sample_temp,
            focus_mode=args.focus_mode,
            total_epochs=total_epochs,
            warmup_epochs=args.sample_warmup_epochs,
            gamma=args.focal_gamma,
        )

    sat_backbone_path = args.sat_backbone_path
    lvd_backbone_path = args.lvd_backbone_path

    # 运行阶段
    def is_stage_completed(save_dir, stage, name):
        """检查阶段是否已完成 (backbone_final.pth 是否存在)"""
        final_path = os.path.join(
            save_dir, f"stage{stage}_{name}", "ckpts", "backbone_final.pth")
        return os.path.exists(final_path)

    # Stage 1
    if 1 in run_stages:
        if sat_backbone_path is not None:
            pass  # 用户指定路径，跳过
        elif is_stage_completed(args.save_dir, 1, "sat") and not args.resume:
            # 如果不是resume模式且文件存在，说明可能是重复运行，这里我们假设如果文件存在就是完成了
            # 但为了安全，我们在非resume模式下可能还是应该询问或者覆盖??
            # 虽然脚本通常是覆盖。但这里为了配合resume逻辑：
            # 如果 resume=True, 且 completed, 则跳过
            # 如果 resume=False, 则重新跑
            pass

        # 更正逻辑：
        # 如果是 resume 且 completed -> 跳过
        # 如果不是 resume -> 重新跑
        # 如果是 resume 且未 completed -> run_stage_1_2 (内部会处理 resume)

        should_run_stage1 = True
        if args.resume and is_stage_completed(args.save_dir, 1, "sat"):
            print(f"[Resume] Stage 1 (SAT) 已完成，跳过")
            sat_backbone_path = os.path.join(
                args.save_dir, "stage1_sat", "ckpts", "backbone_final.pth")
            should_run_stage1 = False

        if should_run_stage1:
            use_glare_stage1 = (1 in glare_loss_stages)
            criterion_stage1 = criterion_glare if use_glare_stage1 else (
                criterion if not use_glare_loss else criterion_standard)
            sample_weighting_stage1 = create_stage_sample_weighting(
                1, args.epochs_stage1)
            sat_backbone_path, _ = run_stage_1_2(
                args, stage=1, dino_ckpt=args.dino_ckpt_sat,
                save_name="sat", device=device,
                train_loader=train_loader, val_loader=val_loader,
                criterion=criterion_stage1, use_glare_loss=use_glare_stage1,
                sample_weighting=sample_weighting_stage1
            )

    # Stage 2
    if 2 in run_stages:
        if lvd_backbone_path is not None:
            pass

        should_run_stage2 = True
        if args.resume and is_stage_completed(args.save_dir, 2, "lvd"):
            print(f"[Resume] Stage 2 (LVD) 已完成，跳过")
            lvd_backbone_path = os.path.join(
                args.save_dir, "stage2_lvd", "ckpts", "backbone_final.pth")
            should_run_stage2 = False

        if should_run_stage2:
            use_glare_stage2 = (2 in glare_loss_stages)
            criterion_stage2 = criterion_glare if use_glare_stage2 else (
                criterion if not use_glare_loss else criterion_standard)
            sample_weighting_stage2 = create_stage_sample_weighting(
                2, args.epochs_stage2)
            lvd_backbone_path, _ = run_stage_1_2(
                args, stage=2, dino_ckpt=args.dino_ckpt_lvd,
                save_name="lvd", device=device,
                train_loader=train_loader, val_loader=val_loader,
                criterion=criterion_stage2, use_glare_loss=use_glare_stage2,
                sample_weighting=sample_weighting_stage2
            )

    # Stage 3
    if 3 in run_stages:
        if sat_backbone_path is None:
            sat_backbone_path = os.path.join(
                args.save_dir, "stage1_sat", "ckpts", "backbone_final.pth")
        if lvd_backbone_path is None:
            lvd_backbone_path = os.path.join(
                args.save_dir, "stage2_lvd", "ckpts", "backbone_final.pth")

        if not os.path.exists(sat_backbone_path):
            print(f"[Error] SAT backbone not found: {sat_backbone_path}")
            print("[Hint] 请先运行 Stage 1 或使用 --sat_backbone_path 指定路径")
            return
        if not os.path.exists(lvd_backbone_path):
            print(f"[Error] LVD backbone not found: {lvd_backbone_path}")
            print("[Hint] 请先运行 Stage 2 或使用 --lvd_backbone_path 指定路径")
            return

        # Stage 3 也可以跳过吗？通常 Stage 3 是最后一步。
        # 如果 Stage 3 还没跑完，run_stage_3 内部会处理 resume。
        # 如果 Stage 3 已经跑完，再次运行且 resume=True...
        # run_stage_3 内部似乎没有检查 completed 的逻辑，它主要检查 latest.pth
        # 我们也可以加上 check

        should_run_stage3 = True
        # Stage 3 没有 backbone_final.pth，它保存的是 best_model.pth 和 final_checkpoint
        # 我们可以检查 best_model.pth 是否存在且 log 中是否有 "Training completed"
        # 简单起见，如果 resume 且 logs 目录下有 completed 标记或者 final_checkpoint 存在?
        # 暂时只依赖 run_stage_3 内部的 resume 逻辑 (它会加载 latest)
        # 如果已经跑完了，latest.pth 的 epoch == total_epochs，循环不会执行

        if should_run_stage3:
            use_glare_stage3 = (3 in glare_loss_stages)
            criterion_stage3 = criterion_glare if use_glare_stage3 else (
                criterion if not use_glare_loss else criterion_standard)
            sample_weighting_stage3 = create_stage_sample_weighting(
                3, args.epochs_stage3)
            best_dice = run_stage_3(
                args, device, train_loader, val_loader, criterion_stage3,
                sat_backbone_path, lvd_backbone_path, use_glare_loss=use_glare_stage3,
                sample_weighting=sample_weighting_stage3
            )

            print("\n" + "=" * 70)
            print(f"[Summary] Best Val Dice (Stage 3): {best_dice:.4f}")
            print(f"[Summary] Results saved to: {args.save_dir}")
            print("=" * 70)

            print(f"[Summary] Best Val Dice (Stage 3): {best_dice:.4f}")
            print(f"[Summary] Results saved to: {args.save_dir}")
            print("=" * 70)

    # 自动清理逻辑
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

    print("Training completed successfully")


if __name__ == "__main__":
    main()
