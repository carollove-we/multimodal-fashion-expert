import matplotlib.pyplot as plt
import json
import os
from pathlib import Path


def plot_loss_curve(loss_history, output_dir, save_name="loss_curve.png"):
    """
    绘制loss变化曲线
    
    Args:
        loss_history: dict，包含 'epochs' 和 'losses' 两个列表
                     {'epochs': [1, 2, 3, ...], 'losses': [loss1, loss2, loss3, ...]}
        output_dir: 输出目录
        save_name: 保存图片的文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epochs = loss_history.get('epochs', [])
    losses = loss_history.get('losses', [])
    
    if not epochs or not losses:
        print("Error: loss_history is empty!")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, marker='o', linewidth=2, markersize=6, label='Loss')
    
    # 美化图表
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加最小值标记
    min_loss_idx = losses.index(min(losses))
    min_loss = losses[min_loss_idx]
    min_epoch = epochs[min_loss_idx]
    plt.annotate(f'Min: {min_loss:.4f}', 
                xy=(min_epoch, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 保存图表
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    plt.close()


def save_loss_history(loss_history, output_dir, save_name="loss_history.json"):
    """
    保存loss历史数据到JSON文件
    
    Args:
        loss_history: dict，包含 'epochs' 和 'losses' 两个列表
        output_dir: 输出目录
        save_name: 保存文件的名称
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_path = os.path.join(output_dir, save_name)
    with open(save_path, 'w') as f:
        json.dump(loss_history, f, indent=4)
    print(f"Loss history saved to {save_path}")


def load_loss_history(output_dir, save_name="loss_history.json"):
    """
    从JSON文件加载loss历史数据
    
    Args:
        output_dir: 输出目录
        save_name: 文件名称
    
    Returns:
        dict: loss历史数据
    """
    save_path = os.path.join(output_dir, save_name)
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Loss history file not found at {save_path}")
        return {'epochs': [], 'losses': []}


def plot_loss_with_moving_average(loss_history, output_dir, window_size=1, save_name="loss_curve_smooth.png"):
    """
    绘制loss变化曲线，并添加移动平均线
    
    Args:
        loss_history: dict，包含 'epochs' 和 'losses' 两个列表
        output_dir: 输出目录
        window_size: 移动平均窗口大小
        save_name: 保存图片的文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    epochs = loss_history.get('epochs', [])
    losses = loss_history.get('losses', [])
    
    if not epochs or not losses:
        print("Error: loss_history is empty!")
        return
    
    # 计算移动平均
    moving_avg = []
    for i in range(len(losses)):
        start = max(0, i - window_size + 1)
        avg = sum(losses[start:i+1]) / (i - start + 1)
        moving_avg.append(avg)
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, marker='o', linewidth=1, markersize=4, alpha=0.5, label='Raw Loss')
    plt.plot(epochs, moving_avg, linewidth=2.5, label=f'Moving Average (window={window_size})')
    
    # 美化图表
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Curve with Moving Average', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # 添加最小值标记
    min_loss_idx = losses.index(min(losses))
    min_loss = losses[min_loss_idx]
    min_epoch = epochs[min_loss_idx]
    plt.annotate(f'Min: {min_loss:.4f}', 
                xy=(min_epoch, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 保存图表
    save_path = os.path.join(output_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve with moving average saved to {save_path}")
    plt.close()



