import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from matplotlib.colors import ListedColormap


def setup_logging(log_dir, name="arc_vqvae", log_level=logging.INFO):
    """设置详细的日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"日志配置完成，保存至: {log_file}")
    return logger


def log_model_summary(model):
    """记录模型结构摘要"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型结构摘要:")
    logging.info(f"总参数数量: {param_count:,}")
    logging.info(f"可训练参数数量: {trainable_param_count:,}")
    
    # 记录每层参数数量
    layer_info = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_info.append((name, param.numel()))
    
    # 按参数数量排序
    layer_info.sort(key=lambda x: x[1], reverse=True)
    
    # 记录前10个最大层
    logging.info(f"参数最多的10层:")
    for name, count in layer_info[:10]:
        logging.info(f"  {name}: {count:,}")


def save_codebook_usage_plot(vqvae, save_path, epoch=None):
    """可视化编码本使用情况"""
    stats = vqvae.get_codebook_usage()
    used_percent = stats['used_percent']
    usage_percents = stats['code_usage_percent']
    
    plt.figure(figsize=(12, 6))
    
    # 绘制编码本使用率柱状图
    plt.bar(np.arange(len(usage_percents)), usage_percents, color='skyblue')
    plt.axhline(y=np.mean(usage_percents), color='r', linestyle='-', label=f'平均: {np.mean(usage_percents):.2f}%')
    
    # 添加标题和标签
    epoch_str = f" (轮次 {epoch})" if epoch is not None else ""
    plt.title(f"编码本使用率{epoch_str} - {stats['used_codes']}/{stats['total_codes']} 编码被使用 ({used_percent:.1f}%)")
    plt.xlabel("编码索引")
    plt.ylabel("使用率 (%)")
    plt.legend()
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"编码本使用情况图表已保存至: {save_path}")
    return stats


def save_grid_visualization(original, reconstructed, save_path):
    """可视化原始网格和重建网格的对比"""
    # 创建ARC网格的颜色映射(10种颜色)
    arc_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    cmap = ListedColormap(arc_colors)
    
    plt.figure(figsize=(10, 5))
    
    # 绘制原始网格
    plt.subplot(1, 2, 1)
    plt.imshow(original.cpu().numpy(), cmap=cmap, vmin=0, vmax=9)
    plt.title("原始网格")
    plt.axis('off')
    
    # 绘制重建网格
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.cpu().numpy(), cmap=cmap, vmin=0, vmax=9)
    plt.title("重建网格")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


def log_metrics_to_json(metrics, save_path):
    """保存训练指标到JSON文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将张量转换为Python标量
    processed_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            processed_metrics[k] = v.item()
        elif isinstance(v, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in v):
            processed_metrics[k] = [x.item() for x in v]
        else:
            processed_metrics[k] = v
    
    # 保存JSON文件
    with open(save_path, 'w') as f:
        json.dump(processed_metrics, f, indent=2)
    
    logging.info(f"训练指标已保存至: {save_path}")
    return processed_metrics