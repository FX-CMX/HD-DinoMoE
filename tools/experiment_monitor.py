#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验自动化系统 - TUI 监控器

终端图形化界面，用于监控实验状态和动态管理队列
功能：
- 实时显示实验状态
- 动态添加/删除实验
- 查看实验日志
- 修改执行配置
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.text import Text
    from rich import print as rprint
except ImportError:
    print("需要安装 rich 库: pip install rich")
    sys.exit(1)

from experiment_config import ExperimentConfig, ExperimentStatus, LOGS_DIR, get_improvement_summary
from experiment_queue import ExperimentQueue

console = Console()


def clear_screen():
    """清屏"""
    os.system('clear' if os.name != 'nt' else 'cls')


def get_status_icon(status: str) -> str:
    """获取状态图标"""
    icons = {
        "pending": "[ ]",
        "running": "[▶]",
        "completed": "[✓]",
        "failed": "[✗]"
    }
    return icons.get(status, "[?]")


def get_status_color(status: str) -> str:
    """获取状态颜色"""
    colors = {
        "pending": "dim",
        "running": "cyan",
        "completed": "green",
        "failed": "red"
    }
    return colors.get(status, "white")


def display_queue_status(queue: ExperimentQueue):
    """显示队列状态"""
    clear_screen()
    
    # 标题
    console.print(Panel(
        "[bold cyan]HD-MoE 实验监控器[/bold cyan]",
        border_style="blue"
    ))
    
    # 执行器状态
    runner_alive = queue.is_runner_alive()
    runner_pid = queue.get_runner_pid()
    
    if runner_alive:
        runner_status = f"[green]运行中[/green] (PID: {runner_pid})"
    else:
        runner_status = "[yellow]未运行[/yellow]"
    
    # 统计信息
    stats = queue.get_statistics()
    
    # 执行配置
    exec_config = queue.get_execution_config()
    
    # 信息面板
    info_text = (
        f"执行器状态: {runner_status}\n"
        f"队列统计: {stats['pending']} 待执行 | {stats['running']} 运行中 | "
        f"{stats['completed']} 已完成 | {stats['failed']} 失败\n"
        f"GPU 范围: {exec_config['gpu_range']} | 休息间隔: {exec_config['cooldown']}s | 显存阈值: 10%"
    )
    console.print(Panel(info_text, border_style="dim"))
    console.print()
    
    # 实验列表
    experiments = queue.get_all_experiments()
    
    if not experiments:
        console.print("[dim]队列为空[/dim]")
    else:
        table = Table(title="实验队列", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("状态", width=6)
        table.add_column("实验名称", min_width=40)
        table.add_column("GPU", width=5)
        table.add_column("开始时间", width=20)
        table.add_column("结果", width=10)
        
        for i, exp in enumerate(experiments):
            status = exp["status"]
            icon = get_status_icon(status)
            color = get_status_color(status)
            
            gpu = str(exp.get("gpu_id", "-")) if exp.get("gpu_id") is not None else "-"
            start_time = exp.get("start_time", "-") or "-"
            
            if status == "running":
                result = "运行中"
            elif status == "completed":
                result = "成功"
            elif status == "failed":
                result = f"失败({exp.get('exit_code', '?')})"
            else:
                result = "-"
            
            table.add_row(
                str(i + 1),
                f"[{color}]{icon}[/{color}]",
                f"[{color}]{exp['name']}[/{color}]",
                gpu,
                start_time,
                result
            )
        
        console.print(table)
    
    console.print()


def show_menu():
    """显示菜单"""
    console.print("[bold]操作菜单:[/bold]")
    console.print("  [R] 刷新状态")
    console.print("  [A] 添加新实验")
    console.print("  [D] 删除实验 (待执行/失败)")
    console.print("  [L] 查看实验日志")
    console.print("  [S] 启动/停止执行器")
    console.print("  [C] 修改执行配置")
    console.print("  [X] 清空队列 (删除所有记录)")
    console.print("  [Q] 退出监控器")
    console.print()


def add_experiment(queue: ExperimentQueue):
    """添加新实验（调用启动器流程）"""
    console.print("\n[cyan]正在启动实验配置器...[/cyan]")
    
    # 导入启动器模块
    from experiment_launcher import stage1_select_combination, stage2_configure_params
    from experiment_config import get_improvement_summary
    
    config = ExperimentConfig()
    
    # 阶段1
    config = stage1_select_combination(config)
    
    # 阶段2
    if Prompt.ask("\n继续配置参数? (Enter 继续, b 返回)", default="").lower() == "b":
        return
    config = stage2_configure_params(config)
    
    exp_name = config.generate_name()
    
    # 显示摘要
    console.print(Panel(
        f"[bold green]实验名称:[/bold green] {exp_name}\n\n"
        f"{get_improvement_summary(config)}\n\n"
        f"数据集: {config.dataset} | Epochs: {config.epochs}",
        title="实验配置摘要",
        border_style="green"
    ))
    
    if Confirm.ask("确认添加到队列?", default=True):
        name = queue.add_experiment(config)
        console.print(f"[green]✓ 实验 '{name}' 已添加到队列[/green]")
    
    Prompt.ask("\n按 Enter 返回")


def delete_experiment(queue: ExperimentQueue):
    """删除实验（待执行或失败的）"""
    experiments = queue.get_all_experiments()
    
    # 筛选可删除的实验（待执行或失败）
    deletable = [exp for exp in experiments if exp["status"] in ["pending", "failed"]]
    
    if not deletable:
        console.print("[yellow]没有可删除的实验（只能删除待执行和失败的实验）[/yellow]")
        Prompt.ask("\n按 Enter 返回")
        return
    
    console.print("\n[bold]可删除的实验列表:[/bold]")
    for i, exp in enumerate(deletable):
        status_icon = get_status_icon(exp["status"])
        console.print(f"  [{i+1}] {status_icon} {exp['name']}")
    
    try:
        choice = IntPrompt.ask("输入要删除的实验编号 (0 取消)", default=0)
        if choice == 0:
            return
        
        if 1 <= choice <= len(deletable):
            name = deletable[choice - 1]["name"]
            if Confirm.ask(f"确认删除实验 '{name}'?", default=False):
                if queue.delete_experiment(name):
                    console.print(f"[green]✓ 实验 '{name}' 已从队列中删除[/green]")
                else:
                    console.print(f"[red]删除失败（可能实验正在运行）[/red]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
    
    Prompt.ask("\n按 Enter 返回")


def view_log(queue: ExperimentQueue):
    """查看实验日志"""
    experiments = queue.get_all_experiments()
    
    if not experiments:
        console.print("[yellow]没有实验记录[/yellow]")
        Prompt.ask("\n按 Enter 返回")
        return
    
    console.print("\n[bold]实验列表:[/bold]")
    for i, exp in enumerate(experiments):
        status_icon = get_status_icon(exp["status"])
        gpu_info = f" (GPU {exp.get('gpu_id', '?')})" if exp.get("gpu_id") is not None else ""
        console.print(f"  [{i+1}] {status_icon} {exp['name']}{gpu_info}")
    
    try:
        choice = IntPrompt.ask("输入要查看的实验编号 (0 取消)", default=0)
        if choice == 0:
            return
        
        if 1 <= choice <= len(experiments):
            exp = experiments[choice - 1]
            name = exp["name"]
            gpu_id = exp.get("gpu_id")
            log_file = LOGS_DIR / f"{name}.log"
            
            if log_file.exists():
                console.print(f"\n[cyan]日志文件: {log_file}[/cyan]")
                if gpu_id is not None:
                    console.print(f"[dim]实时查看: tmux attach -t exp_gpu{gpu_id}[/dim]")
                console.print("[dim]显示最后 50 行...[/dim]\n")
                
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-50:]:
                        console.print(line.rstrip())
            else:
                console.print(f"[yellow]日志文件不存在: {log_file}[/yellow]")
                if gpu_id is not None and exp["status"] == "running":
                    console.print(f"[dim]实验可能正在启动，尝试: tmux attach -t exp_gpu{gpu_id}[/dim]")
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
    
    Prompt.ask("\n按 Enter 返回")


def toggle_runner(queue: ExperimentQueue):
    """启动/停止执行器"""
    if queue.is_runner_alive():
        pid = queue.get_runner_pid()
        console.print(f"[yellow]执行器正在运行 (PID: {pid})[/yellow]")
        console.print("[dim]每个实验在独立 tmux 窗口中运行 (exp_gpu0, exp_gpu1, ...)[/dim]")
        
        if Confirm.ask("是否停止执行器? (当前实验将继续完成)", default=False):
            try:
                os.kill(pid, 15)  # SIGTERM
                console.print("[green]✓ 已发送停止信号[/green]")
            except Exception as e:
                console.print(f"[red]停止失败: {e}[/red]")
    else:
        console.print("[dim]执行器未运行[/dim]")
        
        if Confirm.ask("是否启动执行器?", default=True):
            tools_dir = Path(__file__).parent
            runner_path = tools_dir / "experiment_runner.py"
            
            # 在 tmux 中启动
            try:
                # 先检查是否已有 exp_runner 会话
                result = subprocess.run(
                    ["tmux", "has-session", "-t", "exp_runner"],
                    capture_output=True
                )
                
                if result.returncode == 0:
                    # 会话已存在，杀掉重建
                    subprocess.run(["tmux", "kill-session", "-t", "exp_runner"])
                
                subprocess.Popen([
                    "tmux", "new-session", "-d", "-s", "exp_runner",
                    "python", str(runner_path)
                ])
                console.print("[green]✓ 执行器已在 tmux 会话 'exp_runner' 中启动[/green]")
                console.print("[dim]查看执行器: tmux attach -t exp_runner[/dim]")
                console.print("[dim]查看实验: tmux attach -t exp_gpu<N>[/dim]")
            except Exception as e:
                console.print(f"[red]启动失败: {e}[/red]")
    
    Prompt.ask("\n按 Enter 返回")


def modify_config(queue: ExperimentQueue):
    """修改执行配置"""
    exec_config = queue.get_execution_config()
    
    console.print("\n[bold]当前执行配置:[/bold]")
    console.print(f"  GPU 范围: {exec_config['gpu_range']}")
    console.print(f"  实验间休息: {exec_config['cooldown']} 秒")
    console.print(f"  显存占用阈值: 10% (固定)")
    console.print()
    
    if Confirm.ask("是否修改配置?", default=True):
        gpu_range = Prompt.ask("GPU 范围 (逗号分隔)", default=exec_config['gpu_range'])
        cooldown = IntPrompt.ask("实验间休息时间 (秒)", default=exec_config['cooldown'])
        
        queue.set_execution_config(gpu_range=gpu_range, cooldown=cooldown)
        
        console.print("[green]✓ 配置已更新[/green]")
        console.print("[dim]注意: 修改将在下一个实验开始时生效[/dim]")
    
    Prompt.ask("\n按 Enter 返回")


def clear_all_experiments(queue: ExperimentQueue):
    """清空队列（删除所有记录）"""
    stats = queue.get_statistics()
    total = stats['total']
    
    if total == 0:
        console.print("[yellow]队列已为空[/yellow]")
        Prompt.ask("\n按 Enter 返回")
        return
    
    console.print(f"\n[bold red]警告：将删除队列中的所有 {total} 个实验记录[/bold red]")
    console.print(f"  - {stats['pending']} 个待执行")
    console.print(f"  - {stats['running']} 个运行中")
    console.print(f"  - {stats['completed']} 个已完成")
    console.print(f"  - {stats['failed']} 个失败")
    console.print()
    console.print("[dim]注意：这只会删除队列记录，不会删除实验文件和日志[/dim]")
    
    if stats['running'] > 0:
        console.print("[yellow]警告：有实验正在运行，清空队列不会停止运行中的实验[/yellow]")
    
    if Confirm.ask("确认清空所有实验记录?", default=False):
        queue.clear_all()
        console.print("[green]✓ 队列已清空[/green]")
    
    Prompt.ask("\n按 Enter 返回")


def main():
    """主函数"""
    queue = ExperimentQueue()
    
    while True:
        display_queue_status(queue)
        show_menu()
        
        choice = Prompt.ask("选择操作", choices=["r", "a", "d", "l", "s", "c", "x", "q"], default="r").lower()
        
        if choice == "r":
            continue
        elif choice == "a":
            add_experiment(queue)
        elif choice == "d":
            delete_experiment(queue)
        elif choice == "l":
            view_log(queue)
        elif choice == "s":
            toggle_runner(queue)
        elif choice == "c":
            modify_config(queue)
        elif choice == "x":
            clear_all_experiments(queue)
        elif choice == "q":
            console.print("\n[green]再见！[/green]")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[监控器已退出]")
        sys.exit(0)
