#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化系统 - TUI 启动器

终端图形化界面，用于配置和启动消融实验
三阶段流程：选择改进点组合 → 配置参数 → 添加并启动实验
"""

import os
import sys
import subprocess
from pathlib import Path

# 检查依赖
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("需要安装 rich 库: pip install rich")
    sys.exit(1)

from experiment_config import (
    ExperimentConfig, ExperimentStatus,
    BackboneMode, DecoderConfig, SingleDecoderType,
    GlareMode, SampleWeightMode, FocusMode,
    get_improvement_summary, DATASETS
)
from experiment_queue import ExperimentQueue

console = Console()


def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header(stage: int, title: str):
    """打印阶段头部"""
    console.print(Panel(
        f"[bold cyan]HD-MoE 消融实验配置器[/bold cyan]\n[yellow]阶段 {stage}/3: {title}[/yellow]",
        border_style="blue"
    ))
    console.print()


def select_option(prompt: str, options: list, default: int = 0) -> int:
    """选择选项，返回索引"""
    console.print(f"[bold]{prompt}[/bold]")
    for i, opt in enumerate(options):
        marker = "●" if i == default else "○"
        console.print(f"  [{i+1}] {marker} {opt}")
    
    while True:
        choice = Prompt.ask("输入选项编号", default=str(default + 1))
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        console.print("[red]无效选项，请重新输入[/red]")


def stage1_select_combination(config: ExperimentConfig) -> ExperimentConfig:
    """阶段1: 选择改进点组合"""
    clear_screen()
    print_header(1, "选择改进点组合")
    
    # 1. Backbone 模式
    backbone_options = [
        "SAT 单分支",
        "LVD 单分支",
        "双分支 (SAT + LVD)"
    ]
    backbone_idx = select_option("【改进点1】Backbone 模式:", backbone_options, 
                                  [BackboneMode.SAT, BackboneMode.LVD, BackboneMode.DUAL].index(config.backbone_mode))
    config.backbone_mode = [BackboneMode.SAT, BackboneMode.LVD, BackboneMode.DUAL][backbone_idx]
    console.print()
    
    # 2. 解码器模式（4个选项）
    decoder_options = [
        "1. 单解码器 (无 MoE)",
        "2. 共享投影层 + 独立MoE (sproj_mmoe)",
        "3. 独立投影层 + 独立MoE (mproj_mmoe)",
        "4. 共享投影层 + 共享MoE (sproj_smoe)"
    ]
    decoder_list = [DecoderConfig.SINGLE, DecoderConfig.SHARED_PROJ_MULTI_MOE, 
                    DecoderConfig.MULTI_PROJ_MULTI_MOE, DecoderConfig.SHARED_PROJ_SHARED_MOE]
    decoder_idx = select_option("【改进点2】解码器模式:", decoder_options,
                                 decoder_list.index(config.decoder_config))
    config.decoder_config = decoder_list[decoder_idx]
    console.print()
    
    # 2.1 如果是单解码器，选择类型
    if config.decoder_config == DecoderConfig.SINGLE:
        single_options = ["DPT", "SAM", "D2S", "Linear Attention"]
        single_idx = select_option("单解码器类型:", single_options,
                                    [SingleDecoderType.DPT, SingleDecoderType.SAM, 
                                     SingleDecoderType.D2S, SingleDecoderType.LINEAR_ATTN].index(config.single_decoder_type))
        config.single_decoder_type = [SingleDecoderType.DPT, SingleDecoderType.SAM, 
                                       SingleDecoderType.D2S, SingleDecoderType.LINEAR_ATTN][single_idx]
        console.print()
    
    # 3. 反光抑制
    glare_options = ["关闭", "开启"]
    glare_idx = select_option("【改进点3】反光抑制:", glare_options,
                               [GlareMode.OFF, GlareMode.ON].index(config.glare_mode))
    config.glare_mode = [GlareMode.OFF, GlareMode.ON][glare_idx]
    console.print()
    
    # 4. 样本加权
    sample_options = [
        "不使用",
        "Loss-based (难例挖掘): focus_mode + sample_temp",
        "Focal 加权: focal_gamma",
        "Curriculum (课程学习): warmup_epochs + sample_temp",
        "Class-Aware (类别感知): focus_mode + sample_temp + gate_entropy_lambda"
    ]
    sample_list = [SampleWeightMode.NONE, SampleWeightMode.LOSS_BASED,
                   SampleWeightMode.FOCAL, SampleWeightMode.CURRICULUM,
                   SampleWeightMode.CLASS_AWARE]
    sample_idx = select_option("【改进点4】样本加权策略:", sample_options,
                                sample_list.index(config.sample_weight_mode))
    config.sample_weight_mode = sample_list[sample_idx]
    
    return config


def stage2_configure_params(config: ExperimentConfig) -> ExperimentConfig:
    """阶段2: 配置详细参数"""
    clear_screen()
    print_header(2, "配置实验参数")
    
    # 当前选择摘要
    console.print(Panel(get_improvement_summary(config), title="当前改进点组合", border_style="green"))
    console.print()
    
    # 通用参数
    console.print("[bold cyan]── 通用参数 ──[/bold cyan]")
    
    # 数据集选择
    dataset_options = list(DATASETS.keys())
    dataset_idx = select_option("数据集:", dataset_options, dataset_options.index(config.dataset))
    config.dataset = dataset_options[dataset_idx]
    
    config.epochs = IntPrompt.ask("训练轮数 (Epochs)", default=config.epochs)
    config.lr = FloatPrompt.ask("学习率", default=config.lr)
    console.print()
    
    # 反光抑制参数
    if config.glare_mode == GlareMode.ON:
        console.print("[bold cyan]── 反光抑制参数 ──[/bold cyan]")
        config.glare_penalty = FloatPrompt.ask("惩罚因子 (penalty)", default=config.glare_penalty)
        config.glare_gamma = FloatPrompt.ask("Gamma 参数", default=config.glare_gamma)
        console.print()
    
    # 样本加权参数（按策略显示对应参数）
    if config.sample_weight_mode == SampleWeightMode.LOSS_BASED:
        console.print("[bold cyan]── 样本加权参数 (Loss-based) ──[/bold cyan]")
        focus_options = ["hard (难例优先)", "easy (简单优先)", "balanced (平衡)"]
        focus_idx = select_option("关注模式:", focus_options,
                                   [FocusMode.HARD, FocusMode.EASY, FocusMode.BALANCED].index(config.focus_mode))
        config.focus_mode = [FocusMode.HARD, FocusMode.EASY, FocusMode.BALANCED][focus_idx]
        config.sample_temp = FloatPrompt.ask("温度参数 (sample_temp)", default=config.sample_temp)
        console.print()
    
    elif config.sample_weight_mode == SampleWeightMode.FOCAL:
        console.print("[bold cyan]── 样本加权参数 (Focal) ──[/bold cyan]")
        config.focal_gamma = FloatPrompt.ask("Focal Gamma", default=config.focal_gamma)
        console.print()
    
    elif config.sample_weight_mode == SampleWeightMode.CURRICULUM:
        console.print("[bold cyan]── 样本加权参数 (Curriculum) ──[/bold cyan]")
        config.sample_warmup_epochs = IntPrompt.ask("预热轮数 (warmup_epochs)", default=config.sample_warmup_epochs)
        config.sample_temp = FloatPrompt.ask("温度参数 (sample_temp)", default=config.sample_temp)
        console.print()
    
    elif config.sample_weight_mode == SampleWeightMode.CLASS_AWARE:
        console.print("[bold cyan]── 样本加权参数 (Class-Aware 类别感知) ──[/bold cyan]")
        focus_options = ["hard (难例优先)", "easy (简单优先)", "balanced (平衡)"]
        focus_idx = select_option("关注模式:", focus_options,
                                   [FocusMode.HARD, FocusMode.EASY, FocusMode.BALANCED].index(config.focus_mode))
        config.focus_mode = [FocusMode.HARD, FocusMode.EASY, FocusMode.BALANCED][focus_idx]
        config.sample_temp = FloatPrompt.ask("温度参数 (sample_temp)", default=config.sample_temp)
        config.gate_entropy_lambda = FloatPrompt.ask("门控熵调制系数 (gate_entropy_lambda, 0=不使用)", default=config.gate_entropy_lambda)
        console.print()
    
    # 阶段配置（仅三阶段实验有效）
    if config.backbone_mode == BackboneMode.DUAL:
        console.print("[bold cyan]── 阶段配置 (三阶段训练) ──[/bold cyan]")
        console.print("[dim]可选: auto(自动) / 1,2,3(全部) / 1,2 / 3(仅第三阶段) 等[/dim]")
        
        if config.glare_mode == GlareMode.ON:
            config.glare_loss_stages = Prompt.ask("反光损失启用阶段", default=config.glare_loss_stages)
        
        if config.sample_weight_mode != SampleWeightMode.NONE:
            config.sample_weight_stages = Prompt.ask("样本加权启用阶段", default=config.sample_weight_stages)
        
        console.print()
    
    return config


def stage3_confirm_and_start(config: ExperimentConfig, queue: ExperimentQueue):
    """阶段3: 确认并启动实验"""
    clear_screen()
    print_header(3, "确认并启动实验")
    
    exp_name = config.generate_name()
    
    # 显示完整配置
    console.print(Panel(
        f"[bold green]实验名称:[/bold green] {exp_name}",
        border_style="green"
    ))
    console.print()
    
    # 改进点摘要
    console.print(Panel(get_improvement_summary(config), title="改进点配置", border_style="cyan"))
    console.print()
    
    # 其他参数
    table = Table(title="其他参数", show_header=True, header_style="bold magenta")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("数据集", config.dataset)
    table.add_row("Epochs", str(config.epochs))
    table.add_row("学习率", str(config.lr))
    
    console.print(table)
    console.print()
    
    # 命令预览
    cmd = config.build_command()
    console.print(Panel(cmd, title="命令预览", border_style="dim"))
    console.print()
    
    # 操作选择（根据执行器状态显示不同选项）
    runner_alive = queue.is_runner_alive()
    
    console.print("[bold]请选择操作:[/bold]")
    console.print("  [A] 添加到队列")
    if not runner_alive:
        console.print("  [S] 添加并立即启动执行器")
    else:
        exec_config = queue.get_execution_config()
        console.print(f"  [dim](执行器运行中, GPU: {exec_config['gpu_range']})[/dim]")
    console.print("  [B] 返回修改")
    console.print("  [Q] 退出")
    
    valid_choices = ["a", "b", "q"]
    if not runner_alive:
        valid_choices.append("s")
    
    while True:
        choice = Prompt.ask("选择", choices=valid_choices, default="a").lower()
        
        if choice == "a":
            name = queue.add_experiment(config)
            console.print(f"[green]✓ 实验 '{name}' 已添加到队列[/green]")
            
            if runner_alive:
                console.print("[cyan]执行器运行中，实验将自动开始[/cyan]")
            
            if Confirm.ask("是否继续添加新实验?", default=True):
                return "continue"
            return "exit"
        
        elif choice == "s":
            name = queue.add_experiment(config)
            console.print(f"[green]✓ 实验 '{name}' 已添加到队列[/green]")
            
            # 询问 GPU 范围
            exec_config = queue.get_execution_config()
            console.print(f"\n[dim]当前 GPU 范围: {exec_config['gpu_range']}, 休息时间: {exec_config['cooldown']}s[/dim]")
            
            if Confirm.ask("是否修改执行配置?", default=False):
                new_gpu_range = Prompt.ask("GPU 范围 (逗号分隔)", default=exec_config['gpu_range'])
                new_cooldown = IntPrompt.ask("实验间休息时间 (秒)", default=exec_config['cooldown'])
                queue.set_execution_config(gpu_range=new_gpu_range, cooldown=new_cooldown)
            
            console.print("[cyan]正在启动执行器...[/cyan]")
            # 在 tmux 中启动执行器
            tools_dir = Path(__file__).parent
            runner_path = tools_dir / "experiment_runner.py"
            subprocess.Popen([
                "tmux", "new-session", "-d", "-s", "exp_runner",
                "python", str(runner_path)
            ])
            console.print("[green]✓ 执行器已在 tmux 会话 'exp_runner' 中启动[/green]")
            console.print("[dim]使用 'tmux attach -t exp_runner' 查看输出[/dim]")
            
            if Confirm.ask("是否继续添加新实验?", default=True):
                return "continue"
            return "exit"
        
        elif choice == "b":
            return "back"
        
        elif choice == "q":
            return "exit"


def main():
    """主函数"""
    console.print(Panel(
        "[bold cyan]HD-MoE 消融实验配置器[/bold cyan]\n"
        "[dim]终端图形化界面 - 快速配置和启动消融实验[/dim]",
        border_style="blue"
    ))
    
    queue = ExperimentQueue()
    config = ExperimentConfig()
    
    # 显示当前队列状态
    stats = queue.get_statistics()
    if stats["total"] > 0:
        console.print(f"\n[dim]当前队列: {stats['pending']} 待执行, {stats['running']} 运行中, {stats['completed']} 已完成[/dim]")
    
    
    # 辅助函数：检查日志是否真的成功
    def is_log_successful(name: str) -> bool:
        from experiment_config import LOGS_DIR
        import json as _json
        
        log_file = LOGS_DIR / f"{name}.log"
        if not log_file.exists():
            return False
        
        # 方法1：检查日志文本
        log_ok = False
        try:
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 5000))
                content = f.read()
                
                # 优先检查错误标志（Traceback 上下文可能包含成功标志的源码行）
                error_markers = ["Traceback (most recent call last)", "Error:", "Exception:"]
                has_error = any(m in content for m in error_markers)
                if has_error:
                    return False  # 有错误直接判定失败
                
                success_markers = [
                    "Training completed successfully",
                    "Best Val Dice (Stage 3)",
                    "Best model saved",
                ]
                log_ok = any(m in content for m in success_markers)
        except:
            return False
        
        if not log_ok:
            return False
        
        # 方法2：验证训练数据完整性
        # 查找本实验的 runs 目录
        runs_base = Path("/home/ubuntu/beifen/yyx_code/DINO_model/segdino/segdino_mult/runs")
        exp_dir = runs_base / name
        if not exp_dir.exists():
            return log_ok  # 如果 runs 目录不存在，退回到日志检查结果
        
        # 查找该实验在队列中的配置
        try:
            all_exps = queue.get_all_experiments()
            exp_config = None
            for e in all_exps:
                if e['name'] == name:
                    exp_config = e.get('config', {})
                    break
            
            if exp_config:
                backbone_mode = exp_config.get('backbone_mode', 'sat')
                expected_epochs = exp_config.get('epochs', 50)
                
                if backbone_mode == 'sat_lvd':
                    # 双分支：必须有 stage3 的训练日志
                    stage3_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and 'stage3' in d.name]
                    if not stage3_dirs:
                        return False  # Stage 3 目录不存在
                    
                    # 检查 stage3 训练日志中的 epoch 数
                    for s3_dir in stage3_dirs:
                        log_json = s3_dir / 'training_log.json'
                        if log_json.exists():
                            with open(log_json) as f:
                                data = _json.load(f)
                            history = data.get('history', data)
                            epochs = history.get('epochs', [])
                            if len(epochs) >= expected_epochs:
                                return True
                    return False  # Stage 3 训练日志不完整
                else:
                    # 单分支：检查 stage1 的训练日志
                    stage1_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and 'stage1' in d.name]
                    if stage1_dirs:
                        log_json = stage1_dirs[0] / 'training_log.json'
                        if log_json.exists():
                            with open(log_json) as f:
                                data = _json.load(f)
                            history = data.get('history', data)
                            epochs = history.get('epochs', [])
                            return len(epochs) >= expected_epochs
        except:
            pass  # 如果验证出错，退回日志结果
        
        return log_ok

    # 检查异常状态：
    # 1. 状态为 RUNNING 但执行器未运行 -> 中断
    # 2. 状态为 COMPLETED 但日志显示未成功 -> 假成功
    
    runner_alive = queue.is_runner_alive()
    experiments = queue.get_all_experiments()
    suspicious_experiments = []
    
    for exp in experiments:
        status = exp["status"]
        name = exp["name"]
        
        # 情况1: Runner 死掉的 Running 实验
        if not runner_alive and status == "running":
            suspicious_experiments.append(exp)
        
        # 情况2: 假成功 (无论 Runner 是否运行，Completed 都应该是最终状态)
        elif status == "completed":
            if not is_log_successful(name):
                # 这是一个假成功实验
                suspicious_experiments.append(exp)
        
        # 情况3: 失败实验 (Failed) - 用户可能也想恢复
        elif status == "failed":
            # 标记为 suspicious 以便提示恢复，但加上不同的 reason
            suspicious_experiments.append(exp)

    if suspicious_experiments:
        console.print(f"\n[bold red]检测到 {len(suspicious_experiments)} 个可能异常的实验：[/bold red]")
        for i, exp in enumerate(suspicious_experiments):
            if exp['status'] == 'running':
                reason = "执行器中断"
            elif exp['status'] == 'failed':
                reason = "上次运行失败"
            else:
                reason = "日志缺失成功标志(假成功)"
            
            console.print(f"  [{i+1}] {exp['name']} [{exp['status'].upper()}] ({reason})")
        
        console.print("\n[dim]请输入要恢复的实验编号 (例如: 1,3)，输入 'all' 恢复全部，Enter 跳过[/dim]")
        selection = Prompt.ask("选择恢复").lower().strip()
        
        to_resume = []
        if selection == 'all':
            to_resume = suspicious_experiments
        elif selection:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                for idx in indices:
                    if 0 <= idx < len(suspicious_experiments):
                        to_resume.append(suspicious_experiments[idx])
            except ValueError:
                console.print("[red]无效输入[/red]")
        
        # 执行恢复
        if to_resume:
            count = 0
            for exp in to_resume:
                config = queue._dict_to_config(exp["config"])
                config.is_resume = True
                
                queue.update_experiment_status(exp["name"], ExperimentStatus.FAILED)
                queue.delete_experiment(exp["name"])
                
                new_name = queue.add_experiment(config)
                console.print(f"  [green]✓ 已恢复实验 '{new_name}'[/green]")
                count += 1
            console.print(f"[green]成功恢复 {count} 个实验[/green]")
        
        # 处理未恢复的实验（询问是否删除）
        remaining = [exp for exp in suspicious_experiments if exp not in to_resume]
        if remaining:
            console.print(f"\n[yellow]还有 {len(remaining)} 个未恢复的异常实验记录。[/yellow]")
            if Confirm.ask("是否删除这些记录 (清理队列)?", default=False):
                del_count = 0
                for exp in remaining:
                    # 先确保状态允许删除 (Running 不能直接删，需先改状态)
                    if exp['status'] not in ['pending', 'failed']:
                        queue.update_experiment_status(exp['name'], ExperimentStatus.FAILED)
                    
                    if queue.delete_experiment(exp['name']):
                        console.print(f"  [dim]已删除 '{exp['name']}'[/dim]")
                        del_count += 1
                console.print(f"[green]已清理 {del_count} 条记录[/green]")
        
        console.print()
    
    # 如果执行器已运行，显示当前配置
    if queue.is_runner_alive():
        exec_config = queue.get_execution_config()
        console.print(f"[green]执行器运行中[/green] - GPU: {exec_config['gpu_range']}, 休息: {exec_config['cooldown']}s")
        if Confirm.ask("是否修改执行器 GPU 范围?", default=False):
            new_gpu_range = Prompt.ask("GPU 范围", default=exec_config['gpu_range'])
            new_cooldown = IntPrompt.ask("休息时间 (秒)", default=exec_config['cooldown'])
            queue.set_execution_config(gpu_range=new_gpu_range, cooldown=new_cooldown)
            console.print("[green]✓ 配置已更新，将在下一个实验生效[/green]")
        console.print()
    
    if not Confirm.ask("开始配置新实验?", default=True):
        return
    
    current_stage = 1
    
    while True:
        if current_stage == 1:
            config = stage1_select_combination(config)
            current_stage = 2
        
        elif current_stage == 2:
            console.print("\n[dim]按 Enter 继续，或输入 'b' 返回上一步[/dim]")
            go_next = Prompt.ask("", default="").lower()
            if go_next == "b":
                current_stage = 1
                continue
            
            config = stage2_configure_params(config)
            current_stage = 3
        
        elif current_stage == 3:
            console.print("\n[dim]按 Enter 继续，或输入 'b' 返回上一步[/dim]")
            go_next = Prompt.ask("", default="").lower()
            if go_next == "b":
                current_stage = 2
                continue
            
            result = stage3_confirm_and_start(config, queue)
            
            if result == "continue":
                config = ExperimentConfig()  # 重置配置
                current_stage = 1
            elif result == "back":
                current_stage = 2
            else:
                break
    
    console.print("\n[green]感谢使用！祝实验顺利！[/green]")


if __name__ == "__main__":
    main()
