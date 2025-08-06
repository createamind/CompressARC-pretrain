import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap
import logging


def visualize_reconstructions(model, data_loader, device, num_samples=8, save_path=None):
    """可视化模型重建结果"""
    # 创建ARC网格的颜色映射(10种颜色)
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    cmap = ListedColormap(arc_colors)
    
    model.eval()
    
    # 获取样本
    samples = []
    for batch in data_loader:
        grids = batch['grid'].to(device)
        for i in range(min(grids.size(0), num_samples - len(samples))):
            samples.append(grids[i])
        if len(samples) >= num_samples:
            break
    
    # 确保有足够的样本
    num_samples = min(num_samples, len(samples))
    
    # 创建网格图
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2.5))
    
    with torch.no_grad():
        for i, grid in enumerate(samples):
            # 获取原始网格
            original = grid.cpu().numpy()
            
            # 重建
            reconstructed = model.reconstruct(grid.unsqueeze(0))
            reconstructed = reconstructed.cpu().numpy().squeeze()
            
            # 计算准确率
            accuracy = (original == reconstructed).mean() * 100
            
            # 绘制原始网格
            axes[i, 0].imshow(original, cmap=cmap, vmin=0, vmax=9)
            axes[i, 0].set_title(f"原始 ({original.shape[0]}x{original.shape[1]})")
            axes[i, 0].axis('off')
            
            # 绘制重建网格
            axes[i, 1].imshow(reconstructed, cmap=cmap, vmin=0, vmax=9)
            axes[i, 1].set_title(f"重建 (准确率: {accuracy:.1f}%)")
            axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"重建可视化已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_training_metrics(metrics, save_path=None):
    """绘制训练指标"""
    epochs = len(metrics['train_loss'])
    epoch_range = range(1, epochs + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失
    axs[0, 0].plot(epoch_range, metrics['train_loss'], label='训练')
    axs[0, 0].plot(epoch_range, metrics['val_loss'], label='验证')
    axs[0, 0].set_title('总损失')
    axs[0, 0].set_xlabel('轮次')
    axs[0, 0].set_ylabel('损失')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 重建损失
    axs[0, 1].plot(epoch_range, metrics['train_recon_loss'], label='训练')
    axs[0, 1].plot(epoch_range, metrics['val_recon_loss'], label='验证')
    axs[0, 1].set_title('重建损失')
    axs[0, 1].set_xlabel('轮次')
    axs[0, 1].set_ylabel('损失')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # VQ损失
    axs[1, 0].plot(epoch_range, metrics['train_vq_loss'], label='训练')
    axs[1, 0].plot(epoch_range, metrics['val_vq_loss'], label='验证')
    axs[1, 0].set_title('VQ损失')
    axs[1, 0].set_xlabel('轮次')
    axs[1, 0].set_ylabel('损失')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # 准确率
    axs[1, 1].plot(epoch_range, metrics['train_accuracy'], label='训练')
    axs[1, 1].plot(epoch_range, metrics['val_accuracy'], label='验证')
    axs[1, 1].set_title('像素准确率')
    axs[1, 1].set_xlabel('轮次')
    axs[1, 1].set_ylabel('准确率 (%)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"训练指标图表已保存至: {save_path}")
        plt.close()
    else:
        plt.show()