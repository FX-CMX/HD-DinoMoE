# tools/model_multienc_v2.py
"""
多解码器 MoE 模型 V2

支持两种模式：
1. shared_moe: 共享 4分支MoE（3类共用）
2. multi_moe: 独立 4分支MoE（每类单独）

4个解码器专家：sam, d2s, dpt, linear_attn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.decoder_moe import SharedDecoderMoE, MultiDecoderMoE

__all__ = [
    'build_shared_moe_model',
    'build_multi_moe_model',
    'get_available_experts'
]


# 默认专家列表
DEFAULT_EXPERTS = ['sam', 'd2s', 'dpt', 'linear_attn']


def build_shared_moe_model(
    num_classes,
    dino_size='l',
    dino_ckpt=None,
    repo_dir='./dinov3',
    channels=256,
    expert_types=None,
    use_checkpoint=True
):
    """
    构建共享解码器 MoE 模型
    
    1个 Backbone + 1个 4分支MoE（3类共享）
    
    Args:
        num_classes: 类别数
        dino_size: DINOv3 模型大小 (s/b/l)
        dino_ckpt: 预训练权重路径
        repo_dir: DINOv3 仓库路径
        channels: 解码器通道数
        expert_types: 专家类型列表
        use_checkpoint: 是否使用 gradient checkpointing
    """
    size_map = {
        's': ('dinov3_vits16', 'small'),
        'b': ('dinov3_vitb16', 'base'),
        'l': ('dinov3_vitl16', 'large'),
    }
    
    hub_name, encoder_size = size_map[dino_size]
    print(f"[SharedMoE] Loading backbone: {hub_name}")
    
    backbone = torch.hub.load(repo_dir, hub_name, source='local', weights=dino_ckpt)
    
    model = SharedDecoderMoE(
        num_classes=num_classes,
        backbone=backbone,
        encoder_size=encoder_size,
        channels=channels,
        expert_types=expert_types or DEFAULT_EXPERTS,
        use_checkpoint=use_checkpoint
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SharedMoE] Total params: {total_params:,}")
    print(f"[SharedMoE] Trainable params: {trainable_params:,}")
    print(f"[SharedMoE] Experts: {model.decoder_moe.expert_types}")
    
    return model


def build_multi_moe_model(
    num_classes,
    dino_size='l',
    dino_ckpt=None,
    repo_dir='./dinov3',
    channels=256,
    expert_types=None,
    use_checkpoint=True,
    separate_projects=True
):
    """
    构建独立解码器 MoE 模型
    
    1个 Backbone + 每类独立的 4分支MoE（二级嵌套）
    
    Args:
        num_classes: 类别数
        dino_size: DINOv3 模型大小 (s/b/l)
        dino_ckpt: 预训练权重路径
        repo_dir: DINOv3 仓库路径
        channels: 解码器通道数
        expert_types: 专家类型列表
        use_checkpoint: 是否使用 gradient checkpointing
        separate_projects: 是否每类使用独立投影层
    """
    size_map = {
        's': ('dinov3_vits16', 'small'),
        'b': ('dinov3_vitb16', 'base'),
        'l': ('dinov3_vitl16', 'large'),
    }
    
    hub_name, encoder_size = size_map[dino_size]
    print(f"[MultiMoE] Loading backbone: {hub_name}")
    
    backbone = torch.hub.load(repo_dir, hub_name, source='local', weights=dino_ckpt)
    
    model = MultiDecoderMoE(
        num_classes=num_classes,
        backbone=backbone,
        encoder_size=encoder_size,
        channels=channels,
        expert_types=expert_types or DEFAULT_EXPERTS,
        use_checkpoint=use_checkpoint,
        separate_projects=separate_projects
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MultiMoE] Total params: {total_params:,}")
    print(f"[MultiMoE] Trainable params: {trainable_params:,}")
    print(f"[MultiMoE] Experts: {model.decoder_moes[0].expert_types}")
    print(f"[MultiMoE] Separate projects: {separate_projects}")
    
    return model


def get_available_experts():
    """获取可用的专家解码器列表"""
    return DEFAULT_EXPERTS.copy()
