# modules/decoder_moe.py
"""
多解码器混合专家 (Decoder MoE) 模块

4个解码器专家分支：sam, d2s, dpt, linear_attn
门控模块动态分配权重

支持两种模式：
1. 共享MoE：3类共用1个4分支MoE
2. 独立MoE：每类单独1个4分支MoE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import build_decoder

__all__ = ['DecoderMoE', 'DecoderGate', 'SharedDecoderMoE', 'MultiDecoderMoE']


# 默认的4个解码器专家
DEFAULT_EXPERTS = ['sam', 'd2s', 'dpt', 'linear_attn']


class DecoderGate(nn.Module):
    """
    解码器门控网络
    
    基于全局池化特征计算4个专家的权重
    """
    def __init__(self, in_channels_list, num_experts=4, hidden_dim=256):
        super().__init__()
        self.num_experts = num_experts
        
        # 总输入通道
        total_channels = sum(in_channels_list)
        
        # 门控网络：全局池化 -> MLP -> Softmax
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # 温度参数（控制 softmax 的平滑程度）
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P1, P2, P3, P4]
        Returns:
            weights: 专家权重 [B, num_experts]
        """
        # 上采样到统一尺寸并拼接
        target_size = features[0].shape[-2:]
        upsampled = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled.append(feat)
        
        # 拼接 [B, C_total, H, W]
        concat_feat = torch.cat(upsampled, dim=1)
        
        # 门控计算
        logits = self.gate(concat_feat)  # [B, num_experts]
        
        # 温度缩放的 softmax
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return weights


class DecoderMoE(nn.Module):
    """
    单个 4分支解码器 MoE
    
    4个解码器专家 + 门控融合
    
    Args:
        in_channels_list: 输入通道列表 [C1, C2, C3, C4]
        num_classes: 输出类别数
        channels: 解码器中间通道数
        expert_types: 专家类型列表，默认 ['sam', 'd2s', 'dpt', 'linear_attn']
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=256,
        expert_types=None,
    ):
        super().__init__()
        
        self.expert_types = expert_types or DEFAULT_EXPERTS
        self.num_experts = len(self.expert_types)
        self.num_classes = num_classes
        
        # 创建4个解码器专家
        self.experts = nn.ModuleDict()
        for exp_type in self.expert_types:
            self.experts[exp_type] = build_decoder(
                decoder_type=exp_type,
                in_channels_list=in_channels_list,
                num_classes=num_classes,
                channels=channels
            )
        
        # 门控网络
        self.gate = DecoderGate(in_channels_list, num_experts=self.num_experts)
    
    def forward(self, features, return_weights=False):
        """
        Args:
            features: 多尺度特征列表 [P1, P2, P3, P4]
            return_weights: 是否返回门控权重
        Returns:
            output: 融合输出 [B, num_classes, H, W]
            weights: (可选) 门控权重 [B, num_experts]
        """
        # 计算门控权重
        weights = self.gate(features)  # [B, num_experts]
        
        # 获取各专家输出
        expert_outputs = []
        for exp_type in self.expert_types:
            out = self.experts[exp_type](features)  # [B, C, H, W]
            expert_outputs.append(out)
        
        # 统一输出尺寸（不同解码器可能输出不同尺寸）
        # 找到最大尺寸
        max_h = max(out.shape[2] for out in expert_outputs)
        max_w = max(out.shape[3] for out in expert_outputs)
        target_size = (max_h, max_w)
        
        # 上采样到统一尺寸
        unified_outputs = []
        for out in expert_outputs:
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
            unified_outputs.append(out)
        
        # 加权融合
        # weights: [B, E] -> [B, E, 1, 1, 1]
        weights_expand = weights.view(weights.size(0), self.num_experts, 1, 1, 1)
        
        # expert_outputs: list of [B, C, H, W] -> [B, E, C, H, W]
        stacked_outputs = torch.stack(unified_outputs, dim=1)
        
        # 加权求和
        fused = (stacked_outputs * weights_expand).sum(dim=1)  # [B, C, H, W]
        
        if return_weights:
            return fused, weights
        return fused
    
    def get_expert_weights_dict(self, weights):
        """将权重张量转换为字典"""
        weight_dict = {}
        for i, exp_type in enumerate(self.expert_types):
            weight_dict[exp_type] = weights[:, i].mean().item()
        return weight_dict


class SharedDecoderMoE(nn.Module):
    """
    共享解码器 MoE 模型
    
    1个 Backbone + 1个 4分支MoE（输出所有类别）
    3类共享投影层和解码器
    """
    def __init__(
        self,
        num_classes,
        backbone,
        encoder_size='large',
        channels=256,
        expert_types=None,
        use_checkpoint=True,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.encoder_size = encoder_size
        self.use_checkpoint = use_checkpoint
        
        # 中间层索引
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }
        
        # 输出通道
        embed_dim = backbone.embed_dim
        self.out_channels = [embed_dim] * 4
        
        # 共享投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            for _ in range(4)
        ])
        
        # 4分支解码器 MoE
        self.decoder_moe = DecoderMoE(
            in_channels_list=self.out_channels,
            num_classes=num_classes,
            channels=channels,
            expert_types=expert_types
        )
    
    def lock_backbone(self):
        """冻结 backbone"""
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def _backbone_forward(self, x):
        return self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
    
    def forward(self, x, return_weights=False):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_weights: 是否返回门控权重
        Returns:
            logits: 分割预测 [B, num_classes, H, W]
            weights: (可选) 门控权重
        """
        B, _, H, W = x.shape
        patch_h, patch_w = H // 16, W // 16
        
        # Backbone 特征提取
        if self.use_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self._backbone_forward, x, use_reentrant=False
            )
        else:
            features = self._backbone_forward(x)
        
        # 共享投影
        projected = []
        for i, feat in enumerate(features):
            f = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
            f = self.projects[i](f)
            projected.append(f)
        
        # MoE 解码
        if return_weights:
            logits, weights = self.decoder_moe(projected, return_weights=True)
        else:
            logits = self.decoder_moe(projected)
            weights = None
        
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
        
        if return_weights:
            return logits, weights
        return logits


class MultiDecoderMoE(nn.Module):
    """
    独立解码器 MoE 模型
    
    1个 Backbone + 每类独立的 4分支MoE
    （二级嵌套：每个类别有独立的4专家门控）
    """
    def __init__(
        self,
        num_classes,
        backbone,
        encoder_size='large',
        channels=256,
        expert_types=None,
        use_checkpoint=True,
        separate_projects=True,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.encoder_size = encoder_size
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.separate_projects = separate_projects
        
        # 中间层索引
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }
        
        embed_dim = backbone.embed_dim
        self.out_channels = [embed_dim] * 4
        
        # 投影层
        if separate_projects:
            # 每类独立投影
            self.projects = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                    for _ in range(4)
                ])
                for _ in range(num_classes)
            ])
        else:
            # 共享投影
            self.projects = nn.ModuleList([
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                for _ in range(4)
            ])
        
        # 每类独立的 4分支MoE
        self.decoder_moes = nn.ModuleList([
            DecoderMoE(
                in_channels_list=self.out_channels,
                num_classes=1,  # 每个MoE输出1类
                channels=channels,
                expert_types=expert_types
            )
            for _ in range(num_classes)
        ])
    
    def lock_backbone(self):
        """冻结 backbone"""
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def _backbone_forward(self, x):
        return self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
    
    def forward(self, x, return_separate=False, return_weights=False):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_separate: 是否返回每个解码器的独立输出
            return_weights: 是否返回每个类别的门控权重
        Returns:
            logits: 融合输出 [B, num_classes, H, W]
            separate_outputs: (可选) 每类独立输出列表
            weights_dict: (可选) 每类的门控权重字典
        """
        B, _, H, W = x.shape
        patch_h, patch_w = H // 16, W // 16
        
        # Backbone 特征提取
        if self.use_checkpoint and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self._backbone_forward, x, use_reentrant=False
            )
        else:
            features = self._backbone_forward(x)
        
        # 投影特征（每类可能不同）
        separate_outputs = []
        weights_list = []
        
        for c in range(self.num_classes):
            # 投影
            if self.separate_projects:
                projected = []
                for i, feat in enumerate(features):
                    f = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
                    f = self.projects[c][i](f)
                    projected.append(f)
            else:
                projected = []
                for i, feat in enumerate(features):
                    f = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
                    f = self.projects[i](f)
                    projected.append(f)
            
            # MoE 解码
            if return_weights:
                out, weights = self.decoder_moes[c](projected, return_weights=True)
                weights_list.append(weights)
            else:
                out = self.decoder_moes[c](projected)
            
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
            separate_outputs.append(out)
        
        # 融合所有类别输出
        logits = torch.cat(separate_outputs, dim=1)  # [B, num_classes, H, W]
        
        # 返回值
        results = [logits]
        if return_separate:
            results.append(separate_outputs)
        if return_weights:
            weights_dict = {c: weights_list[c] for c in range(self.num_classes)}
            results.append(weights_dict)
        
        if len(results) == 1:
            return results[0]
        return tuple(results)
    
    def get_all_expert_weights(self, weights_dict):
        """获取所有类别的专家权重摘要"""
        summary = {}
        for c, weights in weights_dict.items():
            for i, exp_type in enumerate(self.decoder_moes[c].expert_types):
                key = f"class{c}_{exp_type}"
                summary[key] = weights[:, i].mean().item()
        return summary
