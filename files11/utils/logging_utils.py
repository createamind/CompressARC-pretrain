import os
import logging
from datetime import datetime
import json
import torch
import torch.nn.functional as F


def setup_logging(log_dir, name="arc_autoencoder", log_level=logging.INFO):
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