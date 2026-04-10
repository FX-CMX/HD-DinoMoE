# modules/hd_moe_model.py
"""
层级双MoE模型 (Hierarchical Dual-MoE, HD-MoE)

融合两个创新点：
1. 双流Backbone MoE：SAT+LVD双分支编码器，门控动态融合
2. 多专家解码器MoE：sam/d2s/dpt/linear_attn四分支解码器

支持配置项：
- decoder_mode: shared_moe（共享MoE）/ multi_moe（独立MoE）
- separate_projects: 是否使用独立投影层（multi_moe模式下每类独立）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .decoders import build_decoder

__all__ = ['HierarchicalDualMoE', 'BackboneGate', 'DecoderGate', 'build_hd_moe_model']


# 默认的4个解码器专家
DEFAULT_EXPERTS = ['sam', 'd2s', 'dpt', 'linear_attn']


class ClassAwareBackboneGate(nn.Module):
    """
    类别感知 Backbone 门控模块
    为每个类别学习独立的专家权重组合
    """
    def __init__(self, in_channels=3, hidden_dim=64, num_experts=2, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        
        # 共享的轻量级特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 每个类别独立的门控 MLP
        self.class_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_classes)
        ])
    
    def forward(self, x):
        feat = self.feature_extractor(x)  # [B, 64, 1, 1]
        feat = feat.flatten(1)  # [B, 64]
        
        class_weights = []
        for gate in self.class_gates:
            weights = gate(feat)  # [B, num_experts]
            class_weights.append(weights)
        
        # [num_classes, B, num_experts]
        return torch.stack(class_weights, dim=0)


class BackboneGate(nn.Module):
    """
    Backbone门控模块（第一级门控）
    
    基于输入图像动态分配双Backbone权重
    """
    def __init__(self, in_channels=3, hidden_dim=64, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        
        # 轻量级特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # MLP 门控网络
        self.gate_mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            gate_weights: 专家权重 [B, num_experts]
        """
        feat = self.feature_extractor(x)  # [B, 64, 1, 1]
        feat = feat.flatten(1)  # [B, 64]
        gate_weights = self.gate_mlp(feat)  # [B, num_experts]
        return gate_weights


class DecoderGate(nn.Module):
    """
    解码器门控模块（第二级门控）
    
    基于融合特征动态分配4个解码器专家权重
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
        
        # 温度参数
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
        
        concat_feat = torch.cat(upsampled, dim=1)
        logits = self.gate(concat_feat)
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return weights


class HierarchicalDualMoE(nn.Module):
    """
    层级双MoE模型
    
    双Backbone（SAT+LVD）+ 4专家解码器（sam/d2s/dpt/linear_attn）
    
    Args:
        num_classes: 类别数
        backbone_1: 第一个Backbone（如SAT）
        backbone_2: 第二个Backbone（如LVD）
        encoder_size: 编码器大小 (small/base/large)
        channels: 解码器通道数
        expert_types: 解码器专家类型列表
        decoder_mode: 解码器模式 (shared_moe/multi_moe)
        separate_projects: 是否使用独立投影层
        use_checkpoint: 是否使用gradient checkpointing
    """
    def __init__(
        self,
        num_classes,
        backbone_1,
        backbone_2,
        encoder_size='large',
        channels=256,
        expert_types=None,
        decoder_mode='shared_moe',
        separate_projects=False,
        use_checkpoint=True,
        single_decoder_type='dpt',  # single 模式下使用的解码器类型
    ):
        super().__init__()
        
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.encoder_size = encoder_size
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.decoder_mode = decoder_mode
        self.separate_projects = separate_projects
        self.expert_types = expert_types or DEFAULT_EXPERTS
        self.num_experts = len(self.expert_types)
        
        # 中间层索引
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
        }
        
        embed_dim = backbone_1.embed_dim
        self.out_channels = [embed_dim] * 4
        
        # ========== 第一级门控：Backbone级 ==========
        self.backbone_gate = ClassAwareBackboneGate(in_channels=3, hidden_dim=64, num_experts=2, num_classes=num_classes)
        
        # ========== 投影层 ==========
        if decoder_mode == 'multi_moe' and separate_projects:
            # 每类独立投影层
            self.projects = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                    for _ in range(4)
                ])
                for _ in range(num_classes)
            ])
        else:
            # 共享投影层
            self.projects = nn.ModuleList([
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
                for _ in range(4)
            ])
        
        # ========== 第二级门控：解码器级 ==========
        if decoder_mode == 'multi_moe':
            # 每类独立的解码器MoE门控
            self.decoder_gates = nn.ModuleList([
                DecoderGate(self.out_channels, num_experts=self.num_experts)
                for _ in range(num_classes)
            ])
            # 每类独立的解码器专家
            self.decoder_experts = nn.ModuleList([
                nn.ModuleDict({
                    exp_type: build_decoder(
                        decoder_type=exp_type,
                        in_channels_list=self.out_channels,
                        num_classes=1,
                        channels=channels
                    )
                    for exp_type in self.expert_types
                })
                for _ in range(num_classes)
            ])
        elif decoder_mode == 'single':
            # 单解码器模式（消融实验用）- 无MoE，只用一个指定类型的解码器
            self.single_decoder_type = single_decoder_type
            self.decoder = build_decoder(
                decoder_type=single_decoder_type,
                in_channels_list=self.out_channels,
                num_classes=num_classes,
                channels=channels
            )
        else:
            # 共享解码器MoE门控 (shared_moe)
            self.decoder_gate = DecoderGate(self.out_channels, num_experts=self.num_experts)
            # 共享解码器专家
            self.decoder_experts = nn.ModuleDict({
                exp_type: build_decoder(
                    decoder_type=exp_type,
                    in_channels_list=self.out_channels,
                    num_classes=num_classes,
                    channels=channels
                )
                for exp_type in self.expert_types
            })
    
    def lock_backbone(self, expert_id=None):
        """冻结Backbone参数"""
        if expert_id is None or expert_id == 1:
            for p in self.backbone_1.parameters():
                p.requires_grad = False
        if expert_id is None or expert_id == 2:
            for p in self.backbone_2.parameters():
                p.requires_grad = False
    
    def _backbone_forward(self, backbone, x):
        return backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
    
    def forward(self, x, return_weights=False):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            return_weights: 是否返回门控权重
        Returns:
            logits: 分割输出 [B, num_classes, H, W]
            weights_dict: (可选) 门控权重字典
        """
        B, _, H, W = x.shape
        patch_h, patch_w = H // 16, W // 16
        
        # ========== 第一级：Backbone特征提取与融合 ==========
        # 获取Backbone门控权重 [num_classes, B, 2]
        backbone_weights = self.backbone_gate(x)
        
        # Backbone 1 特征提取 (共享)
        if self.use_checkpoint and self.training:
            features_1 = checkpoint(self._backbone_forward, self.backbone_1, x, use_reentrant=False)
        else:
            features_1 = self._backbone_forward(self.backbone_1, x)
        
        # Backbone 2 特征提取 (共享)
        if self.use_checkpoint and self.training:
            features_2 = checkpoint(self._backbone_forward, self.backbone_2, x, use_reentrant=False)
        else:
            features_2 = self._backbone_forward(self.backbone_2, x)
        
        # ========== 投影与解码 (按类别逐一处理) ==========
        all_class_outputs = []
        all_decoder_weights = {} # Key: class_idx, Val: weights
        
        for c in range(self.num_classes):
            # 1. 该类别的特征融合
            # 提取该类别的门控权重
            alpha = backbone_weights[c, :, 0:1].unsqueeze(-1)  # [B, 1, 1]
            beta = backbone_weights[c, :, 1:2].unsqueeze(-1)   # [B, 1, 1]
            
            fused_features = []
            for f1, f2 in zip(features_1, features_2):
                # 逐特征层融合
                fused = alpha * f1 + beta * f2
                fused_features.append(fused)
            
            # 2. 投影层
            # 准备投影输入: [B, C, h, w]
            projected = []
            for i, feat in enumerate(fused_features):
                f = feat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)
                
                # 选择投影层
                if self.separate_projects and self.decoder_mode == 'multi_moe':
                    # 每类独立投影
                    proj_layer = self.projects[c][i]
                else:
                    # 共享投影 (单解码器 或 shared_moe 或 简单 multi_moe)
                    proj_layer = self.projects[i]
                
                projected.append(proj_layer(f))
            
            # 3. 解码器处理
            if self.decoder_mode == 'multi_moe':
                # 每类独立的解码器MoE
                gate = self.decoder_gates[c]
                experts = self.decoder_experts[c]
                
                # 门控
                dec_weights = gate(projected) # [B, num_experts]
                all_decoder_weights[c] = dec_weights
                
                # 专家输出
                expert_outputs = []
                for exp_type in self.expert_types:
                    expert_outputs.append(experts[exp_type](projected))
                
                # 统一尺寸
                max_h = max(o.shape[2] for o in expert_outputs)
                max_w = max(o.shape[3] for o in expert_outputs)
                unified = []
                for out in expert_outputs:
                    if out.shape[2:] != (max_h, max_w):
                        out = F.interpolate(out, size=(max_h, max_w), mode='bilinear', align_corners=True)
                    unified.append(out)
                
                # 加权融合
                weights_expand = dec_weights.view(B, self.num_experts, 1, 1, 1)
                stacked = torch.stack(unified, dim=1)
                class_out = (stacked * weights_expand).sum(dim=1) # [B, 1, H', W']
                
                # 上采样到原图
                class_out = F.interpolate(class_out, (H, W), mode='bilinear', align_corners=True)
                all_class_outputs.append(class_out)
                
            elif self.decoder_mode == 'single':
                # 单解码器模式
                # 使用该类别的融合特征，通过共享解码器
                logits = self.decoder(projected) # [B, num_classes, H', W']
                
                # 只取对应类别的通道
                class_out = logits[:, c:c+1]
                class_out = F.interpolate(class_out, (H, W), mode='bilinear', align_corners=True)
                all_class_outputs.append(class_out)
                
                all_decoder_weights[c] = None
                
            else: # shared_moe
                # 共享MoE模式
                # 使用共享门控和专家
                dec_weights = self.decoder_gate(projected) # [B, num_experts]
                all_decoder_weights[c] = dec_weights
                
                expert_outputs = []
                for exp_type in self.expert_types:
                    expert_outputs.append(self.decoder_experts[exp_type](projected))
                
                max_h = max(o.shape[2] for o in expert_outputs)
                max_w = max(o.shape[3] for o in expert_outputs)
                unified = []
                for out in expert_outputs:
                    if out.shape[2:] != (max_h, max_w):
                        out = F.interpolate(out, size=(max_h, max_w), mode='bilinear', align_corners=True)
                    unified.append(out)
                
                weights_expand = dec_weights.view(B, self.num_experts, 1, 1, 1)
                stacked = torch.stack(unified, dim=1)
                logits = (stacked * weights_expand).sum(dim=1) # [B, num_classes, H', W']
                
                # 只取对应类别的通道
                class_out = logits[:, c:c+1]
                class_out = F.interpolate(class_out, (H, W), mode='bilinear', align_corners=True)
                all_class_outputs.append(class_out)
        
        # 拼接所有类别的输出
        final_logits = torch.cat(all_class_outputs, dim=1) # [B, num_classes, H, W]
        
        if return_weights:
            return final_logits, {'backbone': backbone_weights, 'decoder': all_decoder_weights}
        return final_logits


def build_hd_moe_model(
    num_classes,
    dino_size='l',
    dino_ckpt_1=None,
    dino_ckpt_2=None,
    repo_dir='./dinov3',
    channels=256,
    expert_types=None,
    decoder_mode='shared_moe',
    separate_projects=False,
    use_checkpoint=True,
    single_decoder_type='dpt'
):
    """
    构建层级双MoE模型
    
    Args:
        num_classes: 类别数
        dino_size: DINOv3模型大小 (s/b/l)
        dino_ckpt_1: 第一个Backbone权重（如SAT）
        dino_ckpt_2: 第二个Backbone权重（如LVD）
        repo_dir: DINOv3仓库路径
        channels: 解码器通道数
        expert_types: 解码器专家类型
        decoder_mode: 解码器模式 (shared_moe/multi_moe/single)
        separate_projects: 是否独立投影层
        use_checkpoint: 是否使用gradient checkpointing
        single_decoder_type: single模式下使用的解码器类型
    """
    size_map = {
        's': ('dinov3_vits16', 'small'),
        'b': ('dinov3_vitb16', 'base'),
        'l': ('dinov3_vitl16', 'large'),
    }
    
    hub_name, encoder_size = size_map[dino_size]
    print(f"[HD-MoE] Loading backbone 1: {hub_name}")
    backbone_1 = torch.hub.load(repo_dir, hub_name, source='local', weights=dino_ckpt_1)
    
    print(f"[HD-MoE] Loading backbone 2: {hub_name}")
    backbone_2 = torch.hub.load(repo_dir, hub_name, source='local', weights=dino_ckpt_2)
    
    model = HierarchicalDualMoE(
        num_classes=num_classes,
        backbone_1=backbone_1,
        backbone_2=backbone_2,
        encoder_size=encoder_size,
        channels=channels,
        expert_types=expert_types or DEFAULT_EXPERTS,
        decoder_mode=decoder_mode,
        separate_projects=separate_projects,
        use_checkpoint=use_checkpoint,
        single_decoder_type=single_decoder_type
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HD-MoE] Total params: {total_params:,}")
    print(f"[HD-MoE] Trainable params: {trainable_params:,}")
    print(f"[HD-MoE] Decoder mode: {decoder_mode}")
    if decoder_mode == 'single':
        print(f"[HD-MoE] Single decoder type: {single_decoder_type}")
    print(f"[HD-MoE] Separate projects: {separate_projects}")
    print(f"[HD-MoE] Experts: {model.expert_types}")
    
    return model
