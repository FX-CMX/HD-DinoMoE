#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化系统 - 后台执行器

在后台持续执行实验队列中的任务
支持：
- GPU 显存检测，自动跳过被占用的 GPU
- 每个实验在独立 tmux 窗口中运行 (exp_gpu0, exp_gpu1, ...)
- 实验间休息
- 状态实时更新
"""

import os
import sys
import time
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set

from experiment_config import (
    ExperimentConfig, ExperimentStatus, LOGS_DIR,
    BackboneMode, DecoderConfig, SingleDecoderType,
    GlareMode, SampleWeightMode, FocusMode
)
from experiment_queue import ExperimentQueue


class GPUMonitor:
    """GPU 显存监控"""
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[int, float]:
        """获取所有 GPU 的显存使用百分比"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                return {}
            
            gpu_usage = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_id = int(parts[0])
                        mem_used = float(parts[1])
                        mem_total = float(parts[2])
                        usage_percent = (mem_used / mem_total * 100) if mem_total > 0 else 100
                        gpu_usage[gpu_id] = usage_percent
            
            return gpu_usage
        
        except Exception as e:
            print(f"[GPUMonitor] 警告: 无法获取 GPU 信息: {e}")
            return {}
    
    @staticmethod
    def is_gpu_free(gpu_id: int, threshold: float = 10.0) -> bool:
        """检查 GPU 是否空闲（显存使用低于阈值）"""
        usage = GPUMonitor.get_gpu_memory_usage()
        if gpu_id not in usage:
            return True  # 无法获取信息时假设空闲
        return usage[gpu_id] < threshold
    
    @staticmethod
    def get_free_gpus(gpu_list: List[int], threshold: float = 10.0) -> List[int]:
        """获取空闲的 GPU 列表"""
        usage = GPUMonitor.get_gpu_memory_usage()
        free_gpus = []
        for gpu_id in gpu_list:
            if gpu_id not in usage or usage[gpu_id] < threshold:
                free_gpus.append(gpu_id)
        return free_gpus


class TmuxManager:
    """Tmux 会话管理"""
    
    @staticmethod
    def session_exists(session_name: str) -> bool:
        """检查 tmux 会话是否存在"""
        result = subprocess.run(
            ['tmux', 'has-session', '-t', session_name],
            capture_output=True
        )
        return result.returncode == 0
    
    @staticmethod
    def create_session(session_name: str, command: str, cwd: str = None):
        """创建新的 tmux 会话并执行命令"""
        cmd = ['tmux', 'new-session', '-d', '-s', session_name]
        if cwd:
            cmd.extend(['-c', cwd])
        cmd.append(command)
        subprocess.run(cmd)
    
    @staticmethod
    def kill_session(session_name: str):
        """杀死 tmux 会话"""
        subprocess.run(['tmux', 'kill-session', '-t', session_name], capture_output=True)
    
    @staticmethod
    def is_session_running(session_name: str) -> bool:
        """检查 tmux 会话中的命令是否还在运行"""
        if not TmuxManager.session_exists(session_name):
            return False
        
        # 检查会话中是否有活动进程
        result = subprocess.run(
            ['tmux', 'list-panes', '-t', session_name, '-F', '#{pane_pid}'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return False
        
        pane_pid = result.stdout.strip()
        if not pane_pid:
            return False
        
        # 检查该 pid 的子进程
        result = subprocess.run(
            ['pgrep', '-P', pane_pid],
            capture_output=True, text=True
        )
        
        return result.returncode == 0  # 有子进程说明还在运行


class ExperimentRunner:
    """实验执行器（每个实验在独立 tmux 窗口）"""
    
    def __init__(self):
        self.queue = ExperimentQueue()
        self.running_experiments: Dict[int, str] = {}  # gpu_id -> experiment_name
        self.should_stop = False
        
        # 确保日志目录存在
        LOGS_DIR.mkdir(exist_ok=True)
        
        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n[Runner] 收到信号 {signum}，准备退出...")
        self.should_stop = True
    
    def _parse_gpu_range(self, gpu_range: str) -> List[int]:
        """解析 GPU 范围字符串"""
        gpus = []
        for part in gpu_range.split(','):
            part = part.strip()
            if '-' in part:
                start, end = part.split('-')
                gpus.extend(range(int(start), int(end) + 1))
            else:
                if part.isdigit():
                    gpus.append(int(part))
        return gpus
    
    def _get_session_name(self, gpu_id: int) -> str:
        """获取 GPU 对应的 tmux 会话名"""
        return f"exp_gpu{gpu_id}"
    
    def _check_running_experiments(self):
        """检查正在运行的实验状态"""
        finished_gpus = []
        
        for gpu_id, exp_name in list(self.running_experiments.items()):
            session_name = self._get_session_name(gpu_id)
            
            is_running = TmuxManager.is_session_running(session_name)
            log_file = LOGS_DIR / f"{exp_name}.log"
            
            if not is_running:
                # 实验已正常退出，检查结果
                log_file = LOGS_DIR / f"{exp_name}.log"
                
                # 读取日志最后几行判断是否成功
                exit_code = self._get_exit_code_from_log(log_file)
                
                if exit_code == 0:
                    self.queue.update_experiment_status(exp_name, ExperimentStatus.COMPLETED, exit_code=0)
                    print(f"[Runner] ✓ 实验 {exp_name} 完成 (GPU {gpu_id})")
                else:
                    self.queue.update_experiment_status(exp_name, ExperimentStatus.FAILED, exit_code=exit_code)
                    print(f"[Runner] ✗ 实验 {exp_name} 失败 (GPU {gpu_id}, 退出码: {exit_code})")
                
                # 清理 tmux 会话
                TmuxManager.kill_session(session_name)
                finished_gpus.append(gpu_id)
            else:
                # 给卡死的会话一个解脱：如果还是 running 但日志已经打出了 completed，说明是子进程僵死
                exit_code = self._get_exit_code_from_log(log_file)
                if exit_code == 0:
                    self.queue.update_experiment_status(exp_name, ExperimentStatus.COMPLETED, exit_code=0)
                    print(f"[Runner] ⚠️ 发现实验 {exp_name} (GPU {gpu_id}) 日志已完成但进程未退出，强制判定成功并清理")
                    TmuxManager.kill_session(session_name)
                    finished_gpus.append(gpu_id)
        
        # 移除已完成的实验
        for gpu_id in finished_gpus:
            del self.running_experiments[gpu_id]
    
    def _get_exit_code_from_log(self, log_file: Path) -> int:
        """从日志文件获取退出码"""
        if not log_file.exists():
            return -1
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                # 读取最后 20 行
                lines = f.readlines()[-20:]
                content = ''.join(lines)
                
                # 优先检查错误标志（Traceback 上下文可能包含 "Training completed" 源码行）
                if 'Traceback' in content or 'Error:' in content or 'Exception:' in content:
                    return 1
                
                # 再检查成功标志
                if 'Training completed' in content or 'Best model saved' in content:
                    return 0
                
                return 1
        except:
            return -1
    
    def _start_experiment(self, exp: Dict, gpu_id: int):
        """在独立 tmux 窗口中启动实验"""
        name = exp["name"]
        config_dict = exp["config"]
        
        # 恢复配置
        config = ExperimentConfig(
            backbone_mode=BackboneMode(config_dict["backbone_mode"]),
            decoder_config=DecoderConfig(config_dict["decoder_config"]),
            single_decoder_type=SingleDecoderType(config_dict["single_decoder_type"]),
            glare_mode=GlareMode(config_dict["glare_mode"]),
            glare_penalty=config_dict["glare_penalty"],
            glare_gamma=config_dict["glare_gamma"],
            sample_weight_mode=SampleWeightMode(config_dict["sample_weight_mode"]),
            focus_mode=FocusMode(config_dict["focus_mode"]),
            sample_temp=config_dict["sample_temp"],
            sample_warmup_epochs=config_dict["sample_warmup_epochs"],
            focal_gamma=config_dict["focal_gamma"],
            gate_entropy_lambda=config_dict.get("gate_entropy_lambda", 0.0),
            dataset=config_dict["dataset"],
            epochs=config_dict["epochs"],
            batch_size=config_dict["batch_size"],
            lr=config_dict["lr"],
            input_h=config_dict["input_h"],
            input_w=config_dict["input_w"],
            glare_loss_stages=config_dict.get("glare_loss_stages", "auto"),
            sample_weight_stages=config_dict.get("sample_weight_stages", "auto"),
            is_resume=config_dict.get("is_resume", False)
        )
        
        # 构建命令
        base_cmd = config.build_command()
        log_file = LOGS_DIR / f"{name}.log"
        
        # 完整命令：设置 CUDA_VISIBLE_DEVICES + 执行训练 + 记录日志
        full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {base_cmd} 2>&1 | tee {log_file}"
        
        session_name = self._get_session_name(gpu_id)
        tools_dir = str(Path(__file__).parent)
        
        print(f"[Runner] 启动实验: {name}")
        print(f"[Runner] GPU: {gpu_id}")
        print(f"[Runner] Tmux 会话: {session_name}")
        print(f"[Runner] 日志: {log_file}")
        
        # 如果会话已存在，先杀掉
        if TmuxManager.session_exists(session_name):
            TmuxManager.kill_session(session_name)
        
        # 创建新会话
        TmuxManager.create_session(session_name, full_cmd, cwd=tools_dir)
        
        # 更新状态
        self.queue.update_experiment_status(name, ExperimentStatus.RUNNING, gpu_id=gpu_id)
        self.running_experiments[gpu_id] = name
    
    def run(self):
        """主运行循环"""
        print("=" * 60)
        print(f"[Runner] 实验执行器启动 - PID: {os.getpid()}")
        print(f"[Runner] 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 注册 PID
        self.queue.set_runner_pid(os.getpid())
        
        # 获取执行配置
        exec_config = self.queue.get_execution_config()
        gpu_list = self._parse_gpu_range(exec_config["gpu_range"])
        cooldown = exec_config["cooldown"]
        
        print(f"[Runner] GPU 范围: {gpu_list}")
        print(f"[Runner] 可并行实验数: {len(gpu_list)}")
        print(f"[Runner] 实验间休息: {cooldown} 秒")
        print(f"[Runner] 显存占用阈值: 10%")
        print("=" * 60)
        
        last_start_time = 0
        
        while not self.should_stop:
            # 重新获取执行配置（支持动态修改）
            exec_config = self.queue.get_execution_config()
            gpu_list = self._parse_gpu_range(exec_config["gpu_range"])
            cooldown = exec_config["cooldown"]
            
            # 检查正在运行的实验
            self._check_running_experiments()
            
            # 获取空闲且未被占用的 GPU
            free_gpus = GPUMonitor.get_free_gpus(gpu_list, threshold=10.0)
            available_gpus = [g for g in free_gpus if g not in self.running_experiments]
            
            if not available_gpus:
                # 没有可用 GPU
                if free_gpus:
                    # 有空闲 GPU 但都在跑我们的实验
                    pass
                else:
                    # 所有 GPU 都被其他程序占用
                    print(f"[Runner] 所有 GPU 都被占用，等待...")
                time.sleep(10)
                continue
            
            # 获取下一个待执行的实验
            exp = self.queue.get_next_pending()
            
            if exp is None:
                # 队列为空
                if not self.running_experiments:
                    print(f"[Runner] 队列为空，等待新实验...")
                time.sleep(10)
                continue
            
            # 检查休息时间
            now = time.time()
            if now - last_start_time < cooldown and last_start_time > 0:
                remaining = int(cooldown - (now - last_start_time))
                if remaining > 0:
                    time.sleep(min(remaining, 5))
                    continue
            
            # 选择第一个可用 GPU
            gpu_id = available_gpus[0]
            
            # 启动实验
            print(f"\n[Runner] 发现待执行实验: {exp['name']}")
            print(f"[Runner] 分配 GPU {gpu_id} (显存空闲)")
            
            self._start_experiment(exp, gpu_id)
            last_start_time = time.time()
            
            # 短暂等待
            time.sleep(3)
        
        # 等待所有运行中的实验
        print("\n[Runner] 等待运行中的实验完成...")
        while self.running_experiments:
            self._check_running_experiments()
            if self.running_experiments:
                print(f"[Runner] 还有 {len(self.running_experiments)} 个实验在运行...")
                time.sleep(10)
        
        # 清理
        self.queue.set_runner_pid(None)
        
        print("\n" + "=" * 60)
        print(f"[Runner] 执行器退出")
        print("=" * 60)


def main():
    runner = ExperimentRunner()
    runner.run()


if __name__ == "__main__":
    main()
