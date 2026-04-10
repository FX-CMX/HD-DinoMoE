#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化系统 - 配置模块

定义所有配置常量、参数映射、命名规则等
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# ============================================================================
# 路径配置
# ============================================================================
TOOLS_DIR = Path(__file__).parent
QUEUE_FILE = TOOLS_DIR / "experiments_queue.json"
LOGS_DIR = TOOLS_DIR / "experiment_logs"

# 权重文件路径
CHECKPOINT_DIR = str(TOOLS_DIR.parent / "pretrained")
SAT_CHECKPOINT = f"{CHECKPOINT_DIR}/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
LVD_CHECKPOINT = f"{CHECKPOINT_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# 数据集路径
DATASETS_BASE = str(TOOLS_DIR.parent / "datasets" / "datasets_new")
DATASETS = {
    "515": {
        "positive": os.path.join(DATASETS_BASE, "515", "positive"),
        "negative": os.path.join(DATASETS_BASE, "515", "negative")
    },
    "265": {
        "positive": os.path.join(DATASETS_BASE, "265", "positive"),
        "negative": os.path.join(DATASETS_BASE, "265", "negative")
    },
    "250": {
        "positive": os.path.join(DATASETS_BASE, "250", "positive"),
        "negative": os.path.join(DATASETS_BASE, "250", "negative")
    }
}

# ============================================================================
# 改进点枚举
# ============================================================================

class BackboneMode(Enum):
    """Backbone 模式"""
    SAT = "sat"
    LVD = "lvd"
    DUAL = "sat_lvd"
    
    @property
    def display_name(self):
        return {
            "sat": "SAT 单分支",
            "lvd": "LVD 单分支",
            "sat_lvd": "双分支 (SAT + LVD)"
        }[self.value]


class DecoderConfig(Enum):
    """解码器配置（4个选项）"""
    SINGLE = "single"  # 单解码器
    SHARED_PROJ_MULTI_MOE = "sproj_mmoe"  # 共享投影层 + 独立MoE
    MULTI_PROJ_MULTI_MOE = "mproj_mmoe"   # 独立投影层 + 独立MoE
    SHARED_PROJ_SHARED_MOE = "sproj_smoe"  # 共享投影层 + 共享MoE
    
    @property
    def display_name(self):
        return {
            "single": "1. 单解码器 (无 MoE)",
            "sproj_mmoe": "2. 共享投影层 + 独立MoE (multi_moe)",
            "mproj_mmoe": "3. 独立投影层 + 独立MoE (multi_moe)",
            "sproj_smoe": "4. 共享投影层 + 共享MoE (shared_moe)"
        }[self.value]
    
    @property
    def decoder_mode(self) -> str:
        """返回 --decoder_mode 参数值"""
        if self.value == "single":
            return "single"
        elif self.value in ["sproj_mmoe", "mproj_mmoe"]:
            return "multi_moe"
        else:
            return "shared_moe"
    
    @property
    def separate_projects(self) -> bool:
        """返回是否使用独立投影层"""
        return self.value == "mproj_mmoe"


class SingleDecoderType(Enum):
    """单解码器类型（仅 single 模式使用）"""
    DPT = "dpt"
    SAM = "sam"
    D2S = "d2s"
    LINEAR_ATTN = "linear_attn"


class GlareMode(Enum):
    """反光抑制模式"""
    OFF = "noglare"
    ON = "glare"
    
    @property
    def display_name(self):
        return {
            "noglare": "关闭",
            "glare": "开启"
        }[self.value]


class SampleWeightMode(Enum):
    """样本加权模式"""
    NONE = "none"
    LOSS_BASED = "loss"
    FOCAL = "focal"
    CURRICULUM = "curr"
    CLASS_AWARE = "casw"
    
    @property
    def display_name(self):
        return {
            "none": "不使用",
            "loss": "损失加权 (Loss-based): focus_mode + sample_temp",
            "focal": "Focal 加权: focal_gamma",
            "curr": "课程学习 (Curriculum): warmup_epochs + sample_temp",
            "casw": "类别感知 (Class-Aware): focus_mode + sample_temp + gate_entropy_lambda"
        }[self.value]
    
    @property
    def full_name(self):
        """用于命令行参数"""
        return {
            "none": "none",
            "loss": "loss_based",
            "focal": "focal",
            "curr": "curriculum",
            "casw": "class_aware"
        }[self.value]


class FocusMode(Enum):
    """样本关注模式（仅 loss_based 使用）"""
    HARD = "hard"
    EASY = "easy"
    BALANCED = "balanced"


# ============================================================================
# 实验配置数据类
# ============================================================================

@dataclass
class ExperimentConfig:
    """完整的实验配置"""
    
    # 改进点1: Backbone
    backbone_mode: BackboneMode = BackboneMode.SAT
    
    # 改进点2: 解码器（4个选项）
    decoder_config: DecoderConfig = DecoderConfig.SHARED_PROJ_SHARED_MOE
    single_decoder_type: SingleDecoderType = SingleDecoderType.DPT  # 仅 single 模式
    
    # 改进点3: 反光抑制
    glare_mode: GlareMode = GlareMode.OFF
    glare_penalty: float = 3.0
    glare_gamma: float = 1.0
    
    # 改进点4: 样本加权
    sample_weight_mode: SampleWeightMode = SampleWeightMode.NONE
    # loss_based / class_aware 专用
    focus_mode: FocusMode = FocusMode.HARD
    sample_temp: float = 1.0
    # curriculum 专用
    sample_warmup_epochs: int = 10
    # focal 专用
    focal_gamma: float = 2.0
    # class_aware 专用
    gate_entropy_lambda: float = 0.0
    
    # 通用参数
    dataset: str = "515"
    epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-4
    input_h: int = 1024
    input_w: int = 1024
    
    # 阶段配置（仅对多阶段实验有效）
    glare_loss_stages: str = "auto"      # "auto" / "1,2,3" / "1,2" / "3" 等
    sample_weight_stages: str = "auto"   # "auto" / "1,2,3" / "1,2" / "3" 等
    
    # 恢复标志
    is_resume: bool = False
    
    def generate_name(self) -> str:
        """生成实验名称"""
        parts = []
        
        # Backbone 部分
        parts.append(self.backbone_mode.value)
        
        # 解码器部分
        if self.decoder_config == DecoderConfig.SINGLE:
            parts.append(f"single_{self.single_decoder_type.value}")
        else:
            parts.append(self.decoder_config.value)
        
        # 反光抑制部分
        if self.glare_mode == GlareMode.ON:
            parts.append(f"glare_p{self.glare_penalty}_g{self.glare_gamma}")
            # [NEW] 显示生效阶段 (仅双分支)
            if self.backbone_mode == BackboneMode.DUAL:
                gs_str = self.glare_loss_stages.replace(",", "")
                parts.append(f"gs{gs_str}")
        else:
            parts.append("noglare")
        
        # 样本加权部分
        if self.sample_weight_mode == SampleWeightMode.NONE:
            parts.append("nosw")
        else:
            # 具体策略
            if self.sample_weight_mode == SampleWeightMode.LOSS_BASED:
                parts.append(f"loss_{self.focus_mode.value}_t{self.sample_temp}")
            elif self.sample_weight_mode == SampleWeightMode.FOCAL:
                parts.append(f"focal_g{self.focal_gamma}")
            elif self.sample_weight_mode == SampleWeightMode.CURRICULUM:
                parts.append(f"curr_w{self.sample_warmup_epochs}_t{self.sample_temp}")
            elif self.sample_weight_mode == SampleWeightMode.CLASS_AWARE:
                parts.append(f"casw_{self.focus_mode.value}_t{self.sample_temp}_gl{self.gate_entropy_lambda}")
            
            # [NEW] 显示生效阶段 (仅双分支)
            if self.backbone_mode == BackboneMode.DUAL:
                sw_str = self.sample_weight_stages.replace(",", "")
                parts.append(f"sw{sw_str}")
        
        # 数据集
        parts.append(f"d{self.dataset}")
        
        return "_".join(parts)
    
    def to_command_args(self) -> Dict[str, Any]:
        """转换为命令行参数"""
        args = {
            "--data_dir": DATASETS[self.dataset]["positive"],
            "--num_classes": "3",
            "--dino_size": "l",
            "--repo_dir": str(TOOLS_DIR.parent / "dinov3"),
            "--epochs": str(self.epochs),
            "--batch_size": str(self.batch_size),
            "--lr": str(self.lr),
            "--input_h": str(self.input_h),
            "--input_w": str(self.input_w),
            "--save_dir": str(TOOLS_DIR.parent / "runs"),
            "--mixed_precision": "no",
            "--exp_name": self.generate_name()
        }
        
        # Backbone 权重
        if self.backbone_mode in [BackboneMode.SAT, BackboneMode.DUAL]:
            args["--dino_ckpt_sat"] = SAT_CHECKPOINT
        if self.backbone_mode in [BackboneMode.LVD, BackboneMode.DUAL]:
            args["--dino_ckpt_lvd"] = LVD_CHECKPOINT
        
        # 解码器参数
        args["--decoder_mode"] = self.decoder_config.decoder_mode
        if self.decoder_config == DecoderConfig.SINGLE:
            args["--single_decoder_type"] = self.single_decoder_type.value
        if self.decoder_config.separate_projects:
            args["--separate_projects"] = True
        
        # 反光抑制参数
        if self.glare_mode == GlareMode.ON:
            args["--loss_type"] = "glare_progressive"
            args["--glare_data_dir"] = DATASETS[self.dataset]["negative"]
            args["--glare_penalty"] = str(self.glare_penalty)
            args["--glare_gamma"] = str(self.glare_gamma)
        else:
            args["--loss_type"] = "bce_dice"
        
        # 样本加权参数（按策略区分）
        args["--sample_weighting"] = self.sample_weight_mode.full_name
        if self.sample_weight_mode == SampleWeightMode.LOSS_BASED:
            args["--focus_mode"] = self.focus_mode.value
            args["--sample_temp"] = str(self.sample_temp)
        elif self.sample_weight_mode == SampleWeightMode.FOCAL:
            args["--focal_gamma"] = str(self.focal_gamma)
        elif self.sample_weight_mode == SampleWeightMode.CURRICULUM:
            args["--sample_warmup_epochs"] = str(self.sample_warmup_epochs)
            args["--sample_temp"] = str(self.sample_temp)
        elif self.sample_weight_mode == SampleWeightMode.CLASS_AWARE:
            args["--focus_mode"] = self.focus_mode.value
            args["--sample_temp"] = str(self.sample_temp)
            args["--gate_entropy_lambda"] = str(self.gate_entropy_lambda)
        
        # 阶段配置（仅对多阶段实验有效）
        if self.backbone_mode == BackboneMode.DUAL:
            if self.glare_mode == GlareMode.ON and self.glare_loss_stages != "auto":
                args["--glare_loss_stages"] = self.glare_loss_stages
            if self.sample_weight_mode != SampleWeightMode.NONE and self.sample_weight_stages != "auto":
                args["--sample_weight_stages"] = self.sample_weight_stages
        
        # 恢复标志
        if self.is_resume:
            args["--resume"] = True
        
        return args
    
    def build_command(self) -> str:
        """构建完整的命令行"""
        args = self.to_command_args()
        cmd_parts = ["python", "train_hd_moe_staged_v2.py"]
        
        for key, value in args.items():
            if value is True:
                cmd_parts.append(key)
            elif value is not None and value != "":
                cmd_parts.append(f"{key} {value}")
        
        return " ".join(cmd_parts)


# ============================================================================
# 队列状态
# ============================================================================

class ExperimentStatus(Enum):
    """实验状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueuedExperiment:
    """队列中的实验"""
    name: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    progress: int = 0
    gpu_id: Optional[int] = None


# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_GPU_RANGE = "0,1,2,3"
DEFAULT_COOLDOWN = 60


def get_improvement_summary(config: ExperimentConfig) -> str:
    """获取改进点摘要"""
    lines = []
    
    # Backbone
    lines.append(f"• Backbone: {config.backbone_mode.display_name}")
    
    # 解码器
    if config.decoder_config == DecoderConfig.SINGLE:
        lines.append(f"• 解码器: 单解码器 ({config.single_decoder_type.value})")
    else:
        lines.append(f"• 解码器: {config.decoder_config.display_name}")
    
    # 反光抑制
    if config.glare_mode == GlareMode.ON:
        lines.append(f"• 反光抑制: 开启 (penalty={config.glare_penalty}, gamma={config.glare_gamma})")
    else:
        lines.append("• 反光抑制: 关闭")
    
    # 样本加权
    if config.sample_weight_mode == SampleWeightMode.NONE:
        lines.append("• 样本加权: 不使用")
    elif config.sample_weight_mode == SampleWeightMode.LOSS_BASED:
        lines.append(f"• 样本加权: Loss-based (focus={config.focus_mode.value}, temp={config.sample_temp})")
    elif config.sample_weight_mode == SampleWeightMode.FOCAL:
        lines.append(f"• 样本加权: Focal (gamma={config.focal_gamma})")
    elif config.sample_weight_mode == SampleWeightMode.CURRICULUM:
        lines.append(f"• 样本加权: Curriculum (warmup={config.sample_warmup_epochs}, temp={config.sample_temp})")
    elif config.sample_weight_mode == SampleWeightMode.CLASS_AWARE:
        lines.append(f"• 样本加权: Class-Aware (focus={config.focus_mode.value}, temp={config.sample_temp}, gate_lambda={config.gate_entropy_lambda})")
    return "\n".join(lines)
