import os
import torch
import argparse

def add_checkpoint_reload_functionality(parser):
    """为命令行参数解析器添加检查点重载相关参数"""
    parser.add_argument("--resume", type=str, default="",
                       help="从指定检查点恢复训练 (例如: 'checkpoints/rule_guided/20250801_123456/rule_guided_vae_epoch_10.pt')")
    parser.add_argument("--reset_optimizer", action="store_true",
                       help="加载检查点时重置优化器状态")
    parser.add_argument("--reset_lr", action="store_true",
                       help="加载检查点时重置学习率调度器")
    return parser

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, args=None):
    """加载检查点，恢复训练状态 - 增强版，支持多种格式"""
    print(f"从检查点恢复训练: {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 检测检查点格式
    if isinstance(checkpoint, dict):
        # 检查是否是我们期望的格式
        print(f"检查点包含以下键: {list(checkpoint.keys())}")

        # 尝试加载模型状态
        if 'model_state_dict' in checkpoint:
            # 标准格式
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型状态 (标准格式)")
        elif 'state_dict' in checkpoint:
            # 替代格式
            model.load_state_dict(checkpoint['state_dict'])
            print(f"已加载模型状态 (替代格式)")
        else:
            # 检查字典是否直接就是模型状态
            try:
                model.load_state_dict(checkpoint)
                print("已加载模型状态 (直接状态字典格式)")
            except:
                raise ValueError("无法识别的检查点格式，找不到模型状态")

        # 加载优化器状态(如果提供了优化器并且不重置)
        if optimizer is not None and not (args and args.reset_optimizer):
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("已加载优化器状态")
            else:
                print("检查点中没有优化器状态，使用新初始化的优化器")

        # 加载调度器状态(如果提供了调度器并且不重置)
        if scheduler is not None and not (args and args.reset_lr):
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"已加载学习率调度器状态，当前学习率: {scheduler.get_last_lr()[0]:.6f}")
            else:
                print("检查点中没有调度器状态，使用新初始化的调度器")

        # 获取起始轮次
        start_epoch = checkpoint.get('epoch', 0)

    else:
        # 假设这是一个直接的模型状态字典
        try:
            model.load_state_dict(checkpoint)
            print("已加载模型状态 (旧格式)")
            start_epoch = 0  # 不知道轮次，从0开始
        except:
            raise ValueError("无法加载检查点，既不是字典也不是状态字典")

    # 返回起始轮次
    return start_epoch, {}