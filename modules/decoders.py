# modules/decoders.py
"""
高性能分割解码器模块

包含：
1. PPM (Pyramid Pooling Module) - 金字塔池化
2. UPerNetHead - 统一感知解析网络解码器
3. FPNDecoder - 特征金字塔网络解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PPM', 'UPerNetHead', 'FPNDecoder', 'ConvBNReLU']


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU 组合"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PPM(nn.Module):
    """
    金字塔池化模块 (Pyramid Pooling Module)
    
    来自 PSPNet，捕获多尺度上下文信息
    """
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), out_channels=512):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        
        # 每个池化分支的输出通道数
        branch_channels = in_channels // len(pool_sizes)
        self.branch_channels = branch_channels
        
        # 池化 + 1x1 卷积（不用 BN，避免 1x1 输入问题）
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(pool_size) for pool_size in pool_sizes
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, branch_channels, 1, bias=True)
            for _ in pool_sizes
        ])
        
        # 上采样后的 BN + ReLU
        self.post_bn = nn.BatchNorm2d(branch_channels * len(pool_sizes))
        self.relu = nn.ReLU(inplace=True)
        
        # 融合层：原始特征 + 所有池化分支
        self.bottleneck = ConvBNReLU(
            in_channels + branch_channels * len(pool_sizes),
            out_channels, 3, 1, 1
        )
    
    def forward(self, x):
        """
        Args:
            x: 特征图 [B, C, H, W]
        Returns:
            out: 融合后的特征 [B, out_channels, H, W]
        """
        H, W = x.shape[-2:]
        
        # 池化分支（先池化、卷积、上采样，再统一 BN）
        pooled_feats = []
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            pooled = conv(pooled)
            upsampled = F.interpolate(pooled, size=(H, W), mode='bilinear', align_corners=True)
            pooled_feats.append(upsampled)
        
        # 拼接池化分支并做 BN
        pooled_cat = torch.cat(pooled_feats, dim=1)
        pooled_cat = self.relu(self.post_bn(pooled_cat))
        
        # 拼接原始特征并融合
        out = torch.cat([x, pooled_cat], dim=1)
        out = self.bottleneck(out)
        
        return out


class FPNDecoder(nn.Module):
    """
    特征金字塔网络解码器
    
    自顶向下融合多尺度特征
    """
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: 各层输入通道数 [C1, C2, C3, C4]
            out_channels: 输出通道数
        """
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        
        # 融合卷积
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(out_channels, out_channels, 3, 1, 1)
            for _ in range(self.num_levels)
        ])
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P1, P2, P3, P4]（从小到大）
        Returns:
            fpn_outs: FPN 输出特征列表
        """
        # 横向连接
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_levels)]
        
        # 自顶向下路径
        for i in range(self.num_levels - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=True
            )
        
        # 输出卷积
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_levels)]
        
        return fpn_outs


class UPerNetHead(nn.Module):
    """
    UPerNet 解码器头
    
    结合 PPM 和 FPN 的高性能语义分割解码器
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=512,
        pool_sizes=(1, 2, 3, 6),
        dropout_ratio=0.1
    ):
        """
        Args:
            in_channels_list: 各层输入通道数 [C1, C2, C3, C4]
            num_classes: 类别数
            channels: 中间通道数
            pool_sizes: PPM 池化尺寸
            dropout_ratio: Dropout 比例
        """
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # PPM 放在最高层特征上
        self.ppm = PPM(in_channels_list[-1], pool_sizes, channels)
        
        # 横向连接（调整通道数）
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(c, channels, 1, 1, 0) for c in in_channels_list[:-1]
        ])
        
        # FPN 融合卷积
        self.fpn_convs = nn.ModuleList([
            ConvBNReLU(channels, channels, 3, 1, 1) for _ in range(self.num_levels - 1)
        ])
        
        # 最终融合
        self.bottleneck = ConvBNReLU(channels * self.num_levels, channels, 3, 1, 1)
        
        # 分类头
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features):
        """
        Args:
            features: backbone 输出的多尺度特征 [P1, P2, P3, P4]
        Returns:
            out: 分割预测 [B, num_classes, H, W]（与最大特征图同尺寸）
        """
        # PPM 处理最高层
        ppm_out = self.ppm(features[-1])
        
        # 横向连接
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_levels - 1)]
        laterals.append(ppm_out)
        
        # 自顶向下 FPN
        for i in range(self.num_levels - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=True
            )
        
        # FPN 卷积
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_levels - 1)]
        fpn_outs.append(ppm_out)
        
        # 上采样到统一尺寸并拼接
        target_size = fpn_outs[0].shape[-2:]
        fpn_outs_upsampled = [fpn_outs[0]]
        for i in range(1, self.num_levels):
            fpn_outs_upsampled.append(
                F.interpolate(fpn_outs[i], size=target_size, mode='bilinear', align_corners=True)
            )
        
        # 融合所有层级
        fused = torch.cat(fpn_outs_upsampled, dim=1)
        fused = self.bottleneck(fused)
        
        # 分类
        out = self.dropout(fused)
        out = self.cls_seg(out)
        
        return out


class ASPPModule(nn.Module):
    """
    ASPP (Atrous Spatial Pyramid Pooling) 模块
    
    来自 DeepLabV3+，使用不同空洞率捕获多尺度上下文
    """
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18)):
        super().__init__()
        
        self.aspp = nn.ModuleList([
            ConvBNReLU(in_channels, out_channels, 1, 1, 0) if d == 1 else
            ConvBNReLU(in_channels, out_channels, 3, 1, d, dilation=d)
            for d in dilations
        ])
        
        # 全局平均池化分支（不使用 BN，避免 1x1 问题）
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        
        # 上采样后的 BN（用于 GAP 分支）
        self.gap_bn = nn.BatchNorm2d(out_channels)
        self.gap_relu = nn.ReLU(inplace=True)
        
        # 融合
        self.bottleneck = ConvBNReLU(
            out_channels * (len(dilations) + 1), out_channels, 1, 1, 0
        )
    
    def forward(self, x):
        H, W = x.shape[-2:]
        
        branches = [aspp_block(x) for aspp_block in self.aspp]
        
        # GAP 分支：池化 -> 卷积 -> 上采样 -> BN -> ReLU
        gap_out = self.gap_pool(x)
        gap_out = self.gap_conv(gap_out)
        gap_out = F.interpolate(gap_out, size=(H, W), mode='bilinear', align_corners=True)
        gap_out = self.gap_relu(self.gap_bn(gap_out))
        branches.append(gap_out)
        
        out = torch.cat(branches, dim=1)
        out = self.bottleneck(out)
        
        return out


# ============================================================
# 新增解码器（2021-2025）
# ============================================================

class ASPPHead(nn.Module):
    """
    DeepLabV3+ 风格解码器
    
    ASPP + 低层特征融合
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=256,
        dilations=(1, 6, 12, 18),
        dropout_ratio=0.1
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # ASPP 在最高层
        self.aspp = ASPPModule(in_channels_list[-1], channels, dilations)
        
        # 低层特征投影
        self.low_level_proj = ConvBNReLU(in_channels_list[0], 48, 1, 1, 0)
        
        # 融合
        self.fuse = nn.Sequential(
            ConvBNReLU(channels + 48, channels, 3, 1, 1),
            ConvBNReLU(channels, channels, 3, 1, 1),
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features):
        """
        Args:
            features: [low, ..., high] 多尺度特征
        """
        low_feat = features[0]
        high_feat = features[-1]
        
        # ASPP
        aspp_out = self.aspp(high_feat)
        
        # 上采样到低层尺寸
        aspp_out = F.interpolate(aspp_out, size=low_feat.shape[-2:], 
                                  mode='bilinear', align_corners=True)
        
        # 低层投影
        low_proj = self.low_level_proj(low_feat)
        
        # 融合
        fused = torch.cat([aspp_out, low_proj], dim=1)
        fused = self.fuse(fused)
        
        out = self.dropout(fused)
        out = self.cls_seg(out)
        
        return out


class SegFormerHead(nn.Module):
    """
    SegFormer MLP 解码器 (NeurIPS 2021)
    
    轻量级 MLP 解码器，通过简单拼接多尺度特征
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        embed_dim=256,
        dropout_ratio=0.1
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # 每层的线性投影
        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        
        # 融合 MLP
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_levels, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(embed_dim, num_classes, 1)
    
    def forward(self, features):
        """轻量级多尺度融合"""
        target_size = features[0].shape[-2:]
        
        # 投影并上采样
        projected = []
        for i, feat in enumerate(features):
            proj = self.linear_projs[i](feat)
            proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=True)
            projected.append(proj)
        
        # 拼接融合
        fused = torch.cat(projected, dim=1)
        fused = self.fuse(fused)
        
        out = self.dropout(fused)
        out = self.cls_seg(out)
        
        return out


class FPNHead(nn.Module):
    """
    简单 FPN 解码器
    
    轻量级基线
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=256,
        dropout_ratio=0.1
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # 横向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, channels, 1) for c in in_channels_list
        ])
        
        # 输出卷积
        self.output_convs = nn.ModuleList([
            ConvBNReLU(channels, channels, 3, 1, 1) for _ in range(self.num_levels)
        ])
        
        # 最终融合
        self.fuse = ConvBNReLU(channels * self.num_levels, channels, 1, 1, 0)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features):
        # 横向连接
        laterals = [self.lateral_convs[i](features[i]) for i in range(self.num_levels)]
        
        # 自顶向下
        for i in range(self.num_levels - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:],
                mode='bilinear', align_corners=True
            )
        
        # 输出卷积
        outputs = [self.output_convs[i](laterals[i]) for i in range(self.num_levels)]
        
        # 上采样并拼接
        target_size = outputs[0].shape[-2:]
        upsampled = [outputs[0]]
        for i in range(1, self.num_levels):
            upsampled.append(F.interpolate(outputs[i], size=target_size, mode='bilinear', align_corners=True))
        
        fused = torch.cat(upsampled, dim=1)
        fused = self.fuse(fused)
        
        out = self.dropout(fused)
        out = self.cls_seg(out)
        
        return out


class SAMHead(nn.Module):
    """
    SAM 风格 MLP 解码器 (Meta AI 2024)
    
    轻量级两层 MLP + 上采样
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        embed_dim=256,
        num_layers=2,
        dropout_ratio=0.1
    ):
        super().__init__()
        
        # 特征融合
        total_channels = sum(in_channels_list)
        
        # MLP 层
        layers = []
        in_dim = total_channels
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else embed_dim * 2
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Linear(embed_dim, num_classes)
        
        self.in_channels_list = in_channels_list
    
    def forward(self, features):
        # 上采样到统一尺寸
        target_size = features[0].shape[-2:]
        
        upsampled = []
        for feat in features:
            up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            upsampled.append(up)
        
        # 拼接 [B, C_total, H, W]
        fused = torch.cat(upsampled, dim=1)
        B, C, H, W = fused.shape
        
        # [B, H*W, C]
        fused = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # MLP
        fused = self.mlp(fused)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        # 分类 [B, H*W, num_classes]
        out = self.cls_seg(fused)
        
        # [B, num_classes, H, W]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        
        return out


class DepthToSpaceHead(nn.Module):
    """
    Depth-to-Space 解码器 (CVPR 2025)
    
    使用 PixelShuffle 进行上采样，避免插值伪影
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=256,
        upscale_factor=4,
        dropout_ratio=0.1
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.upscale_factor = upscale_factor
        
        # 特征融合
        self.projs = nn.ModuleList([
            ConvBNReLU(c, channels, 1, 1, 0) for c in in_channels_list
        ])
        
        # 融合后处理
        self.fuse = nn.Sequential(
            ConvBNReLU(channels * self.num_levels, channels, 3, 1, 1),
            ConvBNReLU(channels, channels, 3, 1, 1),
        )
        
        # Depth-to-Space (PixelShuffle) 上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features):
        # 投影
        target_size = features[0].shape[-2:]
        
        projected = []
        for i, feat in enumerate(features):
            proj = self.projs[i](feat)
            proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=True)
            projected.append(proj)
        
        # 融合
        fused = torch.cat(projected, dim=1)
        fused = self.fuse(fused)
        
        # Depth-to-Space 上采样
        up = self.upsample(fused)
        
        out = self.dropout(up)
        out = self.cls_seg(out)
        
        return out


class LinearAttentionHead(nn.Module):
    """
    线性注意力解码器 (2024-2025)
    
    使用线性复杂度的注意力机制
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        embed_dim=256,
        num_heads=4,
        dropout_ratio=0.1
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 特征投影
        self.projs = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1) for c in in_channels_list
        ])
        
        # 线性注意力的 Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 后处理
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Linear(embed_dim, num_classes)
    
    def linear_attention(self, q, k, v):
        """线性注意力：O(N) 复杂度"""
        # 使用 ELU + 1 作为特征映射
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # k^T v 先计算
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # q @ (k^T v)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # 归一化
        k_sum = k.sum(dim=2, keepdim=True)
        normalizer = torch.einsum('bhnd,bhmd->bhn', q, k_sum).unsqueeze(-1)
        out = out / (normalizer + 1e-6)
        
        return out
    
    def forward(self, features):
        # 上采样并投影
        target_size = features[0].shape[-2:]
        
        projected = []
        for i, feat in enumerate(features):
            proj = self.projs[i](feat)
            proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=True)
            projected.append(proj)
        
        # 平均融合
        fused = sum(projected) / len(projected)  # [B, C, H, W]
        B, C, H, W = fused.shape
        
        # [B, H*W, C]
        x = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # 线性注意力
        q = self.q_proj(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_out = self.linear_attention(q, k, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_out = self.out_proj(attn_out)
        
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        x = self.dropout(x)
        out = self.cls_seg(x)
        
        # [B, num_classes, H, W]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        
        return out


# ============================================================
# DPT 解码器（原始解码器）
# ============================================================

class DPTHead(nn.Module):
    """
    DPT 解码器（原始）
    
    来自 SegDINO，轻量级多尺度融合
    """
    def __init__(
        self,
        in_channels_list,
        num_classes,
        channels=256,
        dropout_ratio=0.0
    ):
        super().__init__()
        self.num_levels = len(in_channels_list)
        
        # 特征融合
        self.layer_rns = nn.ModuleList([
            nn.Conv2d(c, channels, 3, 1, 1) for c in in_channels_list
        ])
        
        # 输出
        self.output_conv = nn.Conv2d(channels * self.num_levels, num_classes, 1)
    
    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P1, P2, P3, P4]
        """
        # 融合每层
        refined = [self.layer_rns[i](features[i]) for i in range(self.num_levels)]
        
        # 上采样到统一尺寸
        target_hw = refined[0].shape[-2:]
        upsampled = [refined[0]]
        for i in range(1, self.num_levels):
            upsampled.append(
                F.interpolate(refined[i], size=target_hw, mode='bilinear', align_corners=True)
            )
        
        # 拼接输出
        fused = torch.cat(upsampled, dim=1)
        out = self.output_conv(fused)
        
        return out


# ============================================================
# 解码器注册表
# ============================================================

DECODER_REGISTRY = {
    'dpt': DPTHead,            # 原始解码器
    'upernet': UPerNetHead,
    'aspp': ASPPHead,
    'segformer': SegFormerHead,
    'fpn': FPNHead,
    'sam': SAMHead,
    'd2s': DepthToSpaceHead,
    'linear_attn': LinearAttentionHead,
}


def build_decoder(decoder_type, in_channels_list, num_classes, channels=256, **kwargs):
    """
    构建解码器
    
    Args:
        decoder_type: 解码器类型
        in_channels_list: 输入通道列表
        num_classes: 类别数
        channels: 中间通道数
    """
    if decoder_type not in DECODER_REGISTRY:
        raise ValueError(f"Unknown decoder: {decoder_type}. Available: {list(DECODER_REGISTRY.keys())}")
    
    decoder_cls = DECODER_REGISTRY[decoder_type]
    
    # 不同解码器使用不同参数
    if decoder_type in ['segformer', 'sam', 'linear_attn']:
        return decoder_cls(in_channels_list, num_classes, embed_dim=channels, **kwargs)
    else:
        return decoder_cls(in_channels_list, num_classes, channels=channels, **kwargs)


