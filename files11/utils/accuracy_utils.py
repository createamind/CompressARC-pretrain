import torch
import numpy as np
import logging


def determine_background_color(grid, pixel_threshold_pct=40):
    """确定单个网格的背景颜色"""
    # 转换为numpy数组以便处理
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    # 统计颜色
    unique, counts = np.unique(grid, return_counts=True)
    total_pixels = grid.size
    
    # 计算每种颜色占比
    color_percentages = {color: (count / total_pixels) * 100 
                        for color, count in zip(unique, counts)}
    
    # 找出占比最高的颜色
    bg_color = max(color_percentages.items(), key=lambda x: x[1])
    
    # 如果占比高于阈值，认定为背景色
    if bg_color[1] >= pixel_threshold_pct:
        return int(bg_color[0])
    
    # 默认返回0
    return 0


def batch_determine_background(batch_grids, pixel_threshold_pct=40):
    """确定批次中每个网格的背景颜色"""
    # 批次处理 - 对每个样本确定背景色
    bg_colors = []
    
    for i in range(batch_grids.shape[0]):
        grid = batch_grids[i]
        bg_color = determine_background_color(grid, pixel_threshold_pct)
        bg_colors.append(bg_color)
        
    return torch.tensor(bg_colors, device=batch_grids.device)


def calculate_accuracies(predictions, targets, masks=None, background_threshold=40):
    """
    计算总体、背景和前景准确率
    
    参数:
    - predictions: 模型预测结果 [B, H, W]
    - targets: 目标网格 [B, H, W]
    - masks: 可选掩码，指示有效区域 [B, H, W]
    - background_threshold: 背景色检测阈值
    
    返回:
    - 总准确率、背景准确率、前景准确率、背景占比
    """
    # 确定每个样本的背景色
    batch_size = targets.shape[0]
    bg_colors = batch_determine_background(targets, background_threshold)
    
    # 初始化计数器
    total_correct = 0
    total_pixels = 0
    bg_correct = 0
    bg_pixels = 0
    fg_correct = 0
    fg_pixels = 0
    
    # 对每个样本分别计算
    for i in range(batch_size):
        pred = predictions[i]
        target = targets[i]
        bg_color = bg_colors[i]
        
        # 创建背景和前景掩码
        bg_mask = (target == bg_color)
        fg_mask = (target != bg_color)
        
        # 如果提供了掩码，应用它
        if masks is not None:
            mask = masks[i]
            bg_mask = bg_mask & mask
            fg_mask = fg_mask & mask
        
        # 计算准确率
        correct = (pred == target)
        total_correct += correct.sum().item()
        total_pixels += target.numel() if masks is None else mask.sum().item()
        
        bg_correct += (correct & bg_mask).sum().item()
        bg_pixels += bg_mask.sum().item()
        
        fg_correct += (correct & fg_mask).sum().item()
        fg_pixels += fg_mask.sum().item()
    
    # 计算准确率百分比
    total_acc = (total_correct / total_pixels * 100) if total_pixels > 0 else 0
    bg_acc = (bg_correct / bg_pixels * 100) if bg_pixels > 0 else 0
    fg_acc = (fg_correct / fg_pixels * 100) if fg_pixels > 0 else 0
    bg_ratio = (bg_pixels / total_pixels * 100) if total_pixels > 0 else 0
    
    return total_acc, bg_acc, fg_acc, bg_ratio