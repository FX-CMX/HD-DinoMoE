# tools/model.py
"""
SegDINO模型定义
DPT (Dense Prediction Transformer) 架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint

__all__ = ['DPT', 'DPTHead', 'build_model']


def _make_scratch(in_channels, out_features, groups=1, expand=False):
    """创建特征融合模块"""
    scratch = nn.Module()
    
    out_channels = [out_features] * 4
    if expand:
        out_channels = [out_features * 2 ** i for i in range(4)]
    
    scratch.layer1_rn = nn.Conv2d(in_channels[0], out_channels[0], 3, 1, 1, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_channels[1], out_channels[1], 3, 1, 1, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_channels[2], out_channels[2], 3, 1, 1, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_channels[3], out_channels[3], 3, 1, 1, groups=groups)
    
    return scratch


class DPTHead(nn.Module):
    """DPT解码器头"""
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 768],
    ):
        super(DPTHead, self).__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.output_conv = nn.Conv2d(features * 4, nclass, kernel_size=1, stride=1, padding=0)
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        target_hw = layer_1_rn.shape[-2:]
        layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
        layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
        
        fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        out = self.scratch.output_conv(fused)
        return out


class DPT(nn.Module):
    """
    DPT模型
    使用DINO作为backbone的密集预测模型
    """
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone=None,
        use_checkpoint=False
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        self.use_checkpoint = use_checkpoint
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels)
        
    def lock_backbone(self):
        """冻结backbone参数"""
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def _backbone_forward(self, x):
        """Backbone forward，用于 checkpoint 包装"""
        return self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        
        if self.use_checkpoint and self.training:
            # 使用梯度检查点，不保存中间激活值
            features = checkpoint(self._backbone_forward, x, use_reentrant=False)
        else:
            features = self.backbone.get_intermediate_layers(
                x, n=self.intermediate_layer_idx[self.encoder_size]
            )
        
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out


def build_model(num_classes, dino_size='b', dino_ckpt=None, repo_dir='../dinov3', use_checkpoint=False):
    """
    构建SegDINO模型
    Args:
        num_classes: 类别数
        dino_size: DINO模型大小 ('s', 'b', 'l')
        dino_ckpt: DINO预训练权重路径
        repo_dir: DINO仓库路径
        use_checkpoint: 是否使用梯度检查点 (省显存)
    Returns:
        model: DPT模型
    """
    size_map = {
        's': ('dinov3_vits16', 'small'),
        'b': ('dinov3_vitb16', 'base'),
        'l': ('dinov3_vitl16', 'large'),
    }
    
    if dino_size not in size_map:
        raise ValueError(f"Unknown dino_size: {dino_size}. Choose from 's', 'b', 'l'")
    
    hub_name, encoder_size = size_map[dino_size]
    
    # 加载DINO backbone
    backbone = torch.hub.load(repo_dir, hub_name, source='local', weights=dino_ckpt)
    
    # 构建模型
    model = DPT(encoder_size=encoder_size, nclass=num_classes, backbone=backbone, use_checkpoint=use_checkpoint)
    
    return model
