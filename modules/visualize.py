# modules/visualize.py
"""
多标签分割可视化工具
"""
import os
import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

__all__ = [
    'tensor_to_rgb', 'mask_to_gray', 'save_train_visuals', 
    'save_eval_visuals', 'plot_training_curves', 'save_training_log',
    'visualize_multilabel_prediction', 'init_history'
]


def tensor_to_rgb(img_t: torch.Tensor) -> np.ndarray:
    """将tensor转换为RGB图像(BGR格式用于cv2保存)"""
    img = img_t.detach().cpu().float()
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def mask_to_gray(mask_t: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    """将mask tensor转换为灰度图"""
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    elif m.ndim == 3 and m.shape[0] > 1:
        if m.max() > 1.0 or m.min() < 0.0:
            m = torch.sigmoid(m)
        m = m.max(dim=0)[0]
    elif m.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {m.shape}")
    
    if m.max() > 1.0 or m.min() < 0.0:
        m = torch.sigmoid(m)
    m_bin = (m > thr).float()
    m_img = (m_bin * 255.0).round().clamp(0, 255).byte().numpy()
    return m_img


def save_train_visuals(epoch, inputs, logits, targets, out_dir, max_save=8, thr=0.5):
    """保存训练过程的可视化结果"""
    os.makedirs(out_dir, exist_ok=True)
    b = min(inputs.size(0), max_save)
    num_classes = logits.shape[1]
    
    for i in range(b):
        img_bgr = tensor_to_rgb(inputs[i])
        base = os.path.join(out_dir, f"train_ep{epoch:03d}_idx{i:02d}")
        cv2.imwrite(base + "_img.png", img_bgr)
        
        for c in range(num_classes):
            pred_c = mask_to_gray(logits[i, c:c+1], thr)
            gt_c = mask_to_gray(targets[i, c:c+1], thr)
            cv2.imwrite(base + f"_pred_class{c}.png", pred_c)
            cv2.imwrite(base + f"_gt_class{c}.png", gt_c)


@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="val"):
    """保存验证过程的可视化结果"""
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = tensor_to_rgb(inputs)
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    cv2.imwrite(base + "_img.png", img_bgr)
    
    num_classes = logits.shape[0]
    for c in range(num_classes):
        pred_c = mask_to_gray(logits[c:c+1], thr)
        gt_c = mask_to_gray(targets[c:c+1], thr)
        cv2.imwrite(base + f"_pred_class{c}.png", pred_c)
        cv2.imwrite(base + f"_gt_class{c}.png", gt_c)


def visualize_multilabel_prediction(img, pred_masks, gt_masks=None, class_names=None, 
                                     save_path=None, show=False):
    """
    可视化多标签分割结果
    Args:
        img: (H, W, 3) numpy array (RGB)
        pred_masks: (C, H, W) numpy array - predicted binary masks
        gt_masks: (C, H, W) numpy array - ground truth masks (optional)
        class_names: list of class names
        save_path: path to save the visualization
        show: whether to display the plot
    """
    num_classes = pred_masks.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 颜色配置
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    num_cols = num_classes + 2 if gt_masks is not None else num_classes + 1
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))
    
    # 第一行：原图和各类别预测
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for c in range(num_classes):
        axes[0, c + 1].imshow(pred_masks[c], cmap='gray')
        axes[0, c + 1].set_title(f'Pred: {class_names[c]}')
        axes[0, c + 1].axis('off')
    
    # 叠加显示
    overlay = img.copy().astype(np.float32)
    for c in range(num_classes):
        color = np.array(colors[c % len(colors)]) / 255.0
        mask = pred_masks[c] > 0.5
        overlay[mask] = overlay[mask] * 0.5 + color * 127.5
    
    if gt_masks is not None:
        axes[0, -1].imshow(overlay.astype(np.uint8))
        axes[0, -1].set_title('Pred Overlay')
        axes[0, -1].axis('off')
        
        # 第二行：GT
        axes[1, 0].imshow(img)
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        for c in range(num_classes):
            axes[1, c + 1].imshow(gt_masks[c], cmap='gray')
            axes[1, c + 1].set_title(f'GT: {class_names[c]}')
            axes[1, c + 1].axis('off')
        
        # GT叠加显示
        gt_overlay = img.copy().astype(np.float32)
        for c in range(num_classes):
            color = np.array(colors[c % len(colors)]) / 255.0
            mask = gt_masks[c] > 0.5
            gt_overlay[mask] = gt_overlay[mask] * 0.5 + color * 127.5
        
        axes[1, -1].imshow(gt_overlay.astype(np.uint8))
        axes[1, -1].set_title('GT Overlay')
        axes[1, -1].axis('off')
    else:
        axes[0, -1].imshow(overlay.astype(np.uint8))
        axes[0, -1].set_title('Prediction Overlay')
        axes[0, -1].axis('off')
        for ax in axes[1]:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def init_history(num_classes):
    """初始化history字典，包含每个类别的指标"""
    history = {'epochs': [], 'train_loss': [], 'val_loss': []}
    
    # 训练集指标
    for c in range(num_classes):
        history[f'train_dice_class{c}'] = []
        history[f'train_iou_class{c}'] = []
        history[f'train_sensitivity_class{c}'] = []
        history[f'train_specificity_class{c}'] = []
        history[f'train_precision_class{c}'] = []
    history['train_dice_mean'] = []
    history['train_iou_mean'] = []
    history['train_sensitivity_mean'] = []
    history['train_specificity_mean'] = []
    history['train_precision_mean'] = []
    
    # 验证集指标
    for c in range(num_classes):
        history[f'val_dice_class{c}'] = []
        history[f'val_iou_class{c}'] = []
        history[f'val_hd95_class{c}'] = []
        history[f'val_asd_class{c}'] = []
        history[f'val_sensitivity_class{c}'] = []
        history[f'val_specificity_class{c}'] = []
        history[f'val_precision_class{c}'] = []
    history['val_dice_mean'] = []
    history['val_iou_mean'] = []
    history['val_hd95_mean'] = []
    history['val_asd_mean'] = []
    history['val_sensitivity_mean'] = []
    history['val_specificity_mean'] = []
    history['val_precision_mean'] = []
    
    return history


def plot_training_curves(history, save_dir, num_classes, class_names=None):
    """绘制完整的训练曲线"""
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    if class_names is None:
        class_names = [f'Class{i}' for i in range(num_classes)]
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Training Curves - Multi-label Segmentation', fontsize=16, fontweight='bold')
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 2. Dice
    ax = axes[0, 1]
    for c in range(num_classes):
        color = class_colors[c % len(class_colors)]
        ax.plot(epochs, history[f'train_dice_class{c}'], color=color, linestyle='-', 
                label=f'Train {class_names[c]}', linewidth=1.5, alpha=0.7)
        ax.plot(epochs, history[f'val_dice_class{c}'], color=color, linestyle='--', 
                label=f'Val {class_names[c]}', linewidth=1.5)
    ax.plot(epochs, history['train_dice_mean'], 'b-', label='Train Mean', linewidth=2.5)
    ax.plot(epochs, history['val_dice_mean'], 'r-', label='Val Mean', linewidth=2.5)
    best_dice = max(history['val_dice_mean'])
    ax.axhline(y=best_dice, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice')
    ax.set_title(f'Dice (Best Mean: {best_dice:.4f})')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    
    # 3. IoU
    ax = axes[0, 2]
    for c in range(num_classes):
        color = class_colors[c % len(class_colors)]
        ax.plot(epochs, history[f'val_iou_class{c}'], color=color, linestyle='--', 
                label=f'{class_names[c]}', linewidth=1.5)
    ax.plot(epochs, history['val_iou_mean'], 'r-', label='Mean', linewidth=2.5)
    best_iou = max(history['val_iou_mean'])
    ax.set_xlabel('Epoch'); ax.set_ylabel('IoU')
    ax.set_title(f'IoU (Best Mean: {best_iou:.4f})')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 4. HD95 (可选)
    ax = axes[1, 0]
    has_hd95 = 'val_hd95_mean' in history and len(history.get('val_hd95_mean', [])) > 0
    if has_hd95:
        for c in range(num_classes):
            if f'val_hd95_class{c}' in history:
                val_hd95 = [x if np.isfinite(x) else None for x in history[f'val_hd95_class{c}']]
                valid_epochs = [e for e, v in zip(epochs, val_hd95) if v is not None]
                valid_vals = [v for v in val_hd95 if v is not None]
                if valid_vals:
                    color = class_colors[c % len(class_colors)]
                    ax.plot(valid_epochs, valid_vals, color=color, linestyle='--', 
                            label=f'{class_names[c]}', linewidth=1.5)
        val_hd95_mean = [x if np.isfinite(x) else None for x in history['val_hd95_mean']]
        valid_epochs_mean = [e for e, v in zip(epochs, val_hd95_mean) if v is not None]
        valid_mean = [v for v in val_hd95_mean if v is not None]
        if valid_mean:
            ax.plot(valid_epochs_mean, valid_mean, 'r-', label='Mean', linewidth=2.5)
            ax.set_title(f'HD95 (Best: {min(valid_mean):.2f})')
        else:
            ax.set_title('HD95')
    else:
        ax.set_title('HD95 (N/A)')
        ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch'); ax.set_ylabel('HD95')
    ax.grid(True, alpha=0.3)
    
    # 5. ASD (可选)
    ax = axes[1, 1]
    has_asd = 'val_asd_mean' in history and len(history.get('val_asd_mean', [])) > 0
    if has_asd:
        for c in range(num_classes):
            if f'val_asd_class{c}' in history:
                val_asd = [x if np.isfinite(x) else None for x in history[f'val_asd_class{c}']]
                valid_epochs = [e for e, v in zip(epochs, val_asd) if v is not None]
                valid_vals = [v for v in val_asd if v is not None]
                if valid_vals:
                    color = class_colors[c % len(class_colors)]
                    ax.plot(valid_epochs, valid_vals, color=color, linestyle='--', 
                            label=f'{class_names[c]}', linewidth=1.5)
        val_asd_mean = [x if np.isfinite(x) else None for x in history['val_asd_mean']]
        valid_epochs_mean = [e for e, v in zip(epochs, val_asd_mean) if v is not None]
        valid_mean = [v for v in val_asd_mean if v is not None]
        if valid_mean:
            ax.plot(valid_epochs_mean, valid_mean, 'r-', label='Mean', linewidth=2.5)
            ax.set_title(f'ASD (Best: {min(valid_mean):.2f})')
        else:
            ax.set_title('ASD')
    else:
        ax.set_title('ASD (N/A)')
        ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch'); ax.set_ylabel('ASD')
    ax.grid(True, alpha=0.3)
    
    # 6. Sensitivity (可选)
    ax = axes[1, 2]
    has_sens = 'val_sensitivity_mean' in history and len(history.get('val_sensitivity_mean', [])) > 0
    if has_sens:
        for c in range(num_classes):
            if f'val_sensitivity_class{c}' in history and len(history[f'val_sensitivity_class{c}']) > 0:
                color = class_colors[c % len(class_colors)]
                ax.plot(epochs, history[f'val_sensitivity_class{c}'], color=color, linestyle='--', 
                        label=f'{class_names[c]}', linewidth=1.5)
        ax.plot(epochs, history['val_sensitivity_mean'], 'r-', label='Mean', linewidth=2.5)
        ax.set_title(f'Sensitivity (Best: {max(history["val_sensitivity_mean"]):.4f})')
    else:
        ax.set_title('Sensitivity (N/A)')
        ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # 7. Specificity (可选)
    ax = axes[2, 0]
    has_spec = 'val_specificity_mean' in history and len(history.get('val_specificity_mean', [])) > 0
    if has_spec:
        for c in range(num_classes):
            if f'val_specificity_class{c}' in history and len(history[f'val_specificity_class{c}']) > 0:
                color = class_colors[c % len(class_colors)]
                ax.plot(epochs, history[f'val_specificity_class{c}'], color=color, linestyle='--', 
                        label=f'{class_names[c]}', linewidth=1.5)
        ax.plot(epochs, history['val_specificity_mean'], 'r-', label='Mean', linewidth=2.5)
        ax.set_title(f'Specificity (Best: {max(history["val_specificity_mean"]):.4f})')
    else:
        ax.set_title('Specificity (N/A)')
        ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Specificity')
    ax.grid(True, alpha=0.3)
    
    # 8. Precision (可选)
    ax = axes[2, 1]
    has_prec = 'val_precision_mean' in history and len(history.get('val_precision_mean', [])) > 0
    if has_prec:
        for c in range(num_classes):
            if f'val_precision_class{c}' in history and len(history[f'val_precision_class{c}']) > 0:
                color = class_colors[c % len(class_colors)]
                ax.plot(epochs, history[f'val_precision_class{c}'], color=color, linestyle='--', 
                        label=f'{class_names[c]}', linewidth=1.5)
        ax.plot(epochs, history['val_precision_mean'], 'r-', label='Mean', linewidth=2.5)
        ax.set_title(f'Precision (Best: {max(history["val_precision_mean"]):.4f})')
    else:
        ax.set_title('Precision (N/A)')
        ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.3)
    
    # 9. 综合对比
    ax = axes[2, 2]
    ax.plot(epochs, history['val_dice_mean'], 'g-', label='Dice', linewidth=2)
    ax.plot(epochs, history['val_iou_mean'], 'm-', label='IoU', linewidth=2)
    if has_sens:
        ax.plot(epochs, history['val_sensitivity_mean'], 'c-', label='Sensitivity', linewidth=2)
    if has_prec:
        ax.plot(epochs, history['val_precision_mean'], 'y-', label='Precision', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
    ax.set_title('Validation Metrics Comparison')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Save] Training curves saved to: {curve_path}")


def save_training_log(history, save_dir, num_classes, class_names=None, args_dict=None):
    """保存训练日志到JSON和CSV，包含完整的评估指标（兼容缺失指标）"""
    if class_names is None:
        class_names = [f'Class{i}' for i in range(num_classes)]
    
    # 检查是否有 hd95/asd 指标
    has_hd95 = 'val_hd95_mean' in history and len(history['val_hd95_mean']) > 0
    has_asd = 'val_asd_mean' in history and len(history['val_asd_mean']) > 0
    
    # 找最佳值 - 包含所有指标
    best_info = {
        'best_val_dice_mean': max(history['val_dice_mean']),
        'best_val_dice_epoch': history['val_dice_mean'].index(max(history['val_dice_mean'])) + 1,
        'best_val_iou_mean': max(history['val_iou_mean']),
    }
    
    # 可选指标
    if 'val_sensitivity_mean' in history and len(history['val_sensitivity_mean']) > 0:
        best_info['best_val_sensitivity_mean'] = max(history['val_sensitivity_mean'])
    if 'val_specificity_mean' in history and len(history['val_specificity_mean']) > 0:
        best_info['best_val_specificity_mean'] = max(history['val_specificity_mean'])
    if 'val_precision_mean' in history and len(history['val_precision_mean']) > 0:
        best_info['best_val_precision_mean'] = max(history['val_precision_mean'])
    
    # HD95 和 ASD 最佳值 (越小越好，需要过滤 inf)
    if has_hd95:
        finite_hd95 = [x for x in history['val_hd95_mean'] if np.isfinite(x)]
        best_info['best_val_hd95_mean'] = min(finite_hd95) if finite_hd95 else float('inf')
    if has_asd:
        finite_asd = [x for x in history['val_asd_mean'] if np.isfinite(x)]
        best_info['best_val_asd_mean'] = min(finite_asd) if finite_asd else float('inf')
    
    # 每个类别的最佳值
    for c in range(num_classes):
        if f'val_dice_class{c}' in history and len(history[f'val_dice_class{c}']) > 0:
            best_info[f'best_val_dice_{class_names[c]}'] = max(history[f'val_dice_class{c}'])
        if f'val_iou_class{c}' in history and len(history[f'val_iou_class{c}']) > 0:
            best_info[f'best_val_iou_{class_names[c]}'] = max(history[f'val_iou_class{c}'])
        if f'val_sensitivity_class{c}' in history and len(history[f'val_sensitivity_class{c}']) > 0:
            best_info[f'best_val_sens_{class_names[c]}'] = max(history[f'val_sensitivity_class{c}'])
        if f'val_specificity_class{c}' in history and len(history[f'val_specificity_class{c}']) > 0:
            best_info[f'best_val_spec_{class_names[c]}'] = max(history[f'val_specificity_class{c}'])
        if f'val_precision_class{c}' in history and len(history[f'val_precision_class{c}']) > 0:
            best_info[f'best_val_prec_{class_names[c]}'] = max(history[f'val_precision_class{c}'])
        
        if has_hd95 and f'val_hd95_class{c}' in history:
            finite_hd = [x for x in history[f'val_hd95_class{c}'] if np.isfinite(x)]
            best_info[f'best_val_hd95_{class_names[c]}'] = min(finite_hd) if finite_hd else float('inf')
        if has_asd and f'val_asd_class{c}' in history:
            finite_as = [x for x in history[f'val_asd_class{c}'] if np.isfinite(x)]
            best_info[f'best_val_asd_{class_names[c]}'] = min(finite_as) if finite_as else float('inf')
    
    # 保存JSON (包含完整历史记录)
    log_data = {
        'args': args_dict if args_dict else {},
        'num_classes': num_classes,
        'class_names': class_names,
        'history': history,
        **best_info
    }
    
    json_path = os.path.join(save_dir, 'training_log.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, 
                  default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)
    print(f"[Save] Training log saved to: {json_path}")
    
    # 保存CSV - 动态构建表头和行
    csv_path = os.path.join(save_dir, 'training_log.csv')
    
    # 构建表头（只包含存在的指标）
    headers = ['epoch', 'train_loss', 'val_loss']
    
    # 每个类别的指标
    for c in range(num_classes):
        cn = class_names[c]
        if f'train_dice_class{c}' in history:
            headers.append(f'train_dice_{cn}')
        if f'train_iou_class{c}' in history:
            headers.append(f'train_iou_{cn}')
        if f'val_dice_class{c}' in history:
            headers.append(f'val_dice_{cn}')
        if f'val_iou_class{c}' in history:
            headers.append(f'val_iou_{cn}')
    
    # 均值指标
    if 'train_dice_mean' in history:
        headers.append('train_dice_mean')
    if 'train_iou_mean' in history:
        headers.append('train_iou_mean')
    if 'val_dice_mean' in history:
        headers.append('val_dice_mean')
    if 'val_iou_mean' in history:
        headers.append('val_iou_mean')
    
    # 门控权重 (MoE 专用)
    if 'gate_expert1' in history and len(history['gate_expert1']) > 0:
        headers.extend(['gate_expert1', 'gate_expert2'])
    
    # FA-MoE 专家权重
    if 'gate_wavelet' in history and len(history['gate_wavelet']) > 0:
        headers.extend(['gate_wavelet', 'gate_fourier', 'gate_spatial'])
    
    # 数据集权重 (多数据集加权训练)
    if 'dataset_weights' in history and len(history['dataset_weights']) > 0:
        first_weights = history['dataset_weights'][0]
        if isinstance(first_weights, list):
            for j in range(len(first_weights)):
                headers.append(f'weight_D{j}')
    
    # 数据集损失 (多数据集加权训练)
    if 'dataset_losses' in history and len(history['dataset_losses']) > 0:
        first_losses = history['dataset_losses'][0]
        if isinstance(first_losses, dict):
            for k in sorted(first_losses.keys()):
                headers.append(f'loss_D{k}')
    
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        
        for i in range(len(history['train_loss'])):
            # 使用 history['epochs'] 中的实际 epoch，如果存在的话
            if 'epochs' in history and i < len(history['epochs']):
                epoch = history['epochs'][i]
            else:
                epoch = i + 1
            row = [epoch, history['train_loss'][i], history['val_loss'][i]]
            
            # 每个类别的指标
            for c in range(num_classes):
                if f'train_dice_class{c}' in history and i < len(history[f'train_dice_class{c}']):
                    row.append(history[f'train_dice_class{c}'][i])
                if f'train_iou_class{c}' in history and i < len(history[f'train_iou_class{c}']):
                    row.append(history[f'train_iou_class{c}'][i])
                if f'val_dice_class{c}' in history and i < len(history[f'val_dice_class{c}']):
                    row.append(history[f'val_dice_class{c}'][i])
                if f'val_iou_class{c}' in history and i < len(history[f'val_iou_class{c}']):
                    row.append(history[f'val_iou_class{c}'][i])
            
            # 均值指标
            if 'train_dice_mean' in history and i < len(history['train_dice_mean']):
                row.append(history['train_dice_mean'][i])
            if 'train_iou_mean' in history and i < len(history['train_iou_mean']):
                row.append(history['train_iou_mean'][i])
            if 'val_dice_mean' in history and i < len(history['val_dice_mean']):
                row.append(history['val_dice_mean'][i])
            if 'val_iou_mean' in history and i < len(history['val_iou_mean']):
                row.append(history['val_iou_mean'][i])
            
            # 门控权重 (MoE)
            if 'gate_expert1' in history and i < len(history['gate_expert1']):
                row.append(history['gate_expert1'][i])
                row.append(history['gate_expert2'][i])
            
            # FA-MoE 专家权重
            if 'gate_wavelet' in history and i < len(history['gate_wavelet']):
                row.append(history['gate_wavelet'][i])
                row.append(history['gate_fourier'][i])
                row.append(history['gate_spatial'][i])
            
            # 数据集权重 (多数据集加权训练)
            if 'dataset_weights' in history and i < len(history['dataset_weights']):
                weights = history['dataset_weights'][i]
                if isinstance(weights, list):
                    for w in weights:
                        row.append(w)
            
            # 数据集损失 (多数据集加权训练)
            if 'dataset_losses' in history and i < len(history['dataset_losses']):
                losses = history['dataset_losses'][i]
                if isinstance(losses, dict):
                    for k in sorted(losses.keys()):
                        row.append(losses[k])
            
            # 格式化输出
            formatted_row = []
            for x in row:
                if isinstance(x, float):
                    if np.isfinite(x):
                        formatted_row.append(f"{x:.6f}")
                    else:
                        formatted_row.append("inf")
                else:
                    formatted_row.append(str(x))
            
            f.write(','.join(formatted_row) + '\n')
    
    print(f"[Save] Training CSV saved to: {csv_path}")
    
    # 打印最佳指标摘要
    print("=" * 60)
    print("[Best Metrics Summary]")
    print(f"  Dice Mean: {best_info['best_val_dice_mean']:.4f} (Epoch {best_info['best_val_dice_epoch']})")
    print(f"  IoU Mean:  {best_info['best_val_iou_mean']:.4f}")
    if 'best_val_hd95_mean' in best_info:
        hd_str = f"{best_info['best_val_hd95_mean']:.2f}" if np.isfinite(best_info['best_val_hd95_mean']) else "N/A"
        print(f"  HD95 Mean: {hd_str}")
    if 'best_val_asd_mean' in best_info:
        asd_str = f"{best_info['best_val_asd_mean']:.2f}" if np.isfinite(best_info['best_val_asd_mean']) else "N/A"
        print(f"  ASD Mean:  {asd_str}")
    if 'best_val_sensitivity_mean' in best_info:
        print(f"  Sensitivity (Recall) Mean: {best_info['best_val_sensitivity_mean']:.4f}")
    if 'best_val_specificity_mean' in best_info:
        print(f"  Specificity Mean: {best_info['best_val_specificity_mean']:.4f}")
    if 'best_val_precision_mean' in best_info:
        print(f"  Precision Mean: {best_info['best_val_precision_mean']:.4f}")
    print("=" * 60)


