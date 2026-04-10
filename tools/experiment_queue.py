#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化系统 - 队列管理模块

管理 experiments_queue.json 的读写、锁定和状态更新
"""

import json
import os
import time
import fcntl
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from experiment_config import (
    ExperimentConfig, ExperimentStatus, QueuedExperiment,
    QUEUE_FILE, LOGS_DIR,
    BackboneMode, DecoderConfig, SingleDecoderType,
    GlareMode, SampleWeightMode, FocusMode
)


class ExperimentQueue:
    """实验队列管理器"""
    
    def __init__(self, queue_file: Path = QUEUE_FILE):
        self.queue_file = queue_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """确保队列文件存在"""
        if not self.queue_file.exists():
            self._save_queue({
                "version": "2.0",
                "runner_pid": None,
                "current_experiment": None,
                "gpu_range": "0,1,2,3",
                "cooldown": 60,
                "experiments": []
            })
        # 确保日志目录存在
        LOGS_DIR.mkdir(exist_ok=True)
    
    def _load_queue(self) -> Dict:
        """加载队列（带文件锁）"""
        with open(self.queue_file, 'r', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data
    
    def _save_queue(self, data: Dict):
        """保存队列（带文件锁）"""
        with open(self.queue_file, 'w', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict:
        """将配置转为可序列化的字典"""
        return {
            "backbone_mode": config.backbone_mode.value,
            "decoder_config": config.decoder_config.value,
            "single_decoder_type": config.single_decoder_type.value,
            "glare_mode": config.glare_mode.value,
            "glare_penalty": config.glare_penalty,
            "glare_gamma": config.glare_gamma,
            "sample_weight_mode": config.sample_weight_mode.value,
            "focus_mode": config.focus_mode.value,
            "sample_temp": config.sample_temp,
            "sample_warmup_epochs": config.sample_warmup_epochs,
            "focal_gamma": config.focal_gamma,
            "gate_entropy_lambda": config.gate_entropy_lambda,
            "dataset": config.dataset,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "input_h": config.input_h,
            "input_w": config.input_w,
            "glare_loss_stages": config.glare_loss_stages,
            "sample_weight_stages": config.sample_weight_stages,
            "is_resume": config.is_resume
        }
    
    def _dict_to_config(self, d: Dict) -> ExperimentConfig:
        """从字典恢复配置"""
        return ExperimentConfig(
            backbone_mode=BackboneMode(d["backbone_mode"]),
            decoder_config=DecoderConfig(d["decoder_config"]),
            single_decoder_type=SingleDecoderType(d["single_decoder_type"]),
            glare_mode=GlareMode(d["glare_mode"]),
            glare_penalty=d["glare_penalty"],
            glare_gamma=d["glare_gamma"],
            sample_weight_mode=SampleWeightMode(d["sample_weight_mode"]),
            focus_mode=FocusMode(d["focus_mode"]),
            sample_temp=d["sample_temp"],
            sample_warmup_epochs=d["sample_warmup_epochs"],
            focal_gamma=d["focal_gamma"],
            gate_entropy_lambda=d.get("gate_entropy_lambda", 0.0),
            dataset=d["dataset"],
            epochs=d["epochs"],
            batch_size=d["batch_size"],
            lr=d["lr"],
            input_h=d["input_h"],
            input_w=d["input_w"],
            glare_loss_stages=d.get("glare_loss_stages", "auto"),
            sample_weight_stages=d.get("sample_weight_stages", "auto"),
            is_resume=d.get("is_resume", False)
        )
    
    def add_experiment(self, config: ExperimentConfig) -> str:
        """添加实验到队列，返回实验名称"""
        data = self._load_queue()
        
        name = config.generate_name()
        
        # 检查是否已存在同名实验
        existing_names = [exp["name"] for exp in data["experiments"]]
        if name in existing_names:
            # 添加时间戳后缀
            timestamp = datetime.now().strftime("%H%M%S")
            name = f"{name}_{timestamp}"
        
        experiment = {
            "name": name,
            "config": self._config_to_dict(config),
            "status": ExperimentStatus.PENDING.value,
            "start_time": None,
            "end_time": None,
            "exit_code": None,
            "progress": 0,
            "gpu_id": None
        }
        
        data["experiments"].append(experiment)
        self._save_queue(data)
        
        return name
    
    def get_all_experiments(self) -> List[Dict]:
        """获取所有实验"""
        data = self._load_queue()
        return data["experiments"]
    
    def get_pending_experiments(self) -> List[Dict]:
        """获取所有待执行的实验"""
        experiments = self.get_all_experiments()
        return [exp for exp in experiments if exp["status"] == ExperimentStatus.PENDING.value]
    
    def get_next_pending(self) -> Optional[Dict]:
        """获取下一个待执行的实验"""
        pending = self.get_pending_experiments()
        return pending[0] if pending else None
    
    def update_experiment_status(self, name: str, status: ExperimentStatus, 
                                  gpu_id: int = None, progress: int = None,
                                  exit_code: int = None):
        """更新实验状态"""
        data = self._load_queue()
        
        for exp in data["experiments"]:
            if exp["name"] == name:
                exp["status"] = status.value
                
                if status == ExperimentStatus.RUNNING:
                    exp["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data["current_experiment"] = name
                    if gpu_id is not None:
                        exp["gpu_id"] = gpu_id
                
                if status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                    exp["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if exit_code is not None:
                        exp["exit_code"] = exit_code
                    if data["current_experiment"] == name:
                        data["current_experiment"] = None
                
                if progress is not None:
                    exp["progress"] = progress
                
                break
        
        self._save_queue(data)
    
    def delete_pending_experiment(self, name: str) -> bool:
        """删除待执行的实验"""
        data = self._load_queue()
        
        for i, exp in enumerate(data["experiments"]):
            if exp["name"] == name and exp["status"] == ExperimentStatus.PENDING.value:
                data["experiments"].pop(i)
                self._save_queue(data)
                return True
        
        return False
    
    def get_runner_pid(self) -> Optional[int]:
        """获取执行器 PID"""
        data = self._load_queue()
        return data.get("runner_pid")
    
    def set_runner_pid(self, pid: Optional[int]):
        """设置执行器 PID"""
        data = self._load_queue()
        data["runner_pid"] = pid
        self._save_queue(data)
    
    def is_runner_alive(self) -> bool:
        """检查执行器是否在运行"""
        pid = self.get_runner_pid()
        if pid is None:
            return False
        
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def get_execution_config(self) -> Dict:
        """获取执行配置（GPU范围、休息时间等）"""
        data = self._load_queue()
        return {
            "gpu_range": data.get("gpu_range", "0,1,2,3"),
            "cooldown": data.get("cooldown", 60)
        }
    
    def set_execution_config(self, gpu_range: str = None, cooldown: int = None):
        """设置执行配置"""
        data = self._load_queue()
        
        if gpu_range is not None:
            data["gpu_range"] = gpu_range
        if cooldown is not None:
            data["cooldown"] = cooldown
        
        self._save_queue(data)
    
    def get_statistics(self) -> Dict:
        """获取队列统计信息"""
        experiments = self.get_all_experiments()
        
        stats = {
            "total": len(experiments),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }
        
        for exp in experiments:
            status = exp["status"]
            if status == "pending":
                stats["pending"] += 1
            elif status == "running":
                stats["running"] += 1
            elif status == "completed":
                stats["completed"] += 1
            elif status == "failed":
                stats["failed"] += 1
        
        return stats
    
    def clear_completed(self):
        """清除已完成的实验记录"""
        data = self._load_queue()
        data["experiments"] = [
            exp for exp in data["experiments"]
            if exp["status"] not in ["completed", "failed"]
        ]
        self._save_queue(data)
    
    def delete_experiment(self, name: str) -> bool:
        """删除实验（待执行或失败的）"""
        data = self._load_queue()
        
        for i, exp in enumerate(data["experiments"]):
            if exp["name"] == name:
                # 只能删除待执行或失败的实验
                if exp["status"] in [ExperimentStatus.PENDING.value, ExperimentStatus.FAILED.value]:
                    data["experiments"].pop(i)
                    self._save_queue(data)
                    return True
                else:
                    return False  # 运行中或已完成的不能删除
        
        return False
    
    def clear_all(self):
        """清空所有实验记录（不影响运行中的实验进程）"""
        data = self._load_queue()
        data["experiments"] = []
        data["current_experiment"] = None
        self._save_queue(data)
    
    def get_experiment_config(self, name: str) -> Optional[ExperimentConfig]:
        """根据名称获取实验配置"""
        experiments = self.get_all_experiments()
        for exp in experiments:
            if exp["name"] == name:
                return self._dict_to_config(exp["config"])
        return None
