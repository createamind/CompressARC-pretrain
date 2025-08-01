import os
import torch
import argparse

def add_checkpoint_reload_functionality(parser):
    """为命令行参数解析器添加检查点重载相关参数"""
    parser.add_argument("--resume", type=str, default="",
                       help="从指定检查点恢复训练 (例如: 'checkpoints/rule_guided/20250801_123456/checkpoint_epoch_10.pt')")
    parser.add_argument("--reset_optimizer", action="store_true",
                       help="加载检查点时重置优化器状态")
    parser.add_argument("--reset_lr", action="store_true", 
                       help="加载检查点时重置学习率调度器")
    return parser

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """加载检查点，恢复训练状态"""
    print(f"从检查点恢复训练: {checkpoint_path}")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")
        
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型状态 (轮次 {checkpoint['epoch']})")
    
    # 加载优化器状态（除非选择重置）
    if 'optimizer_state_dict' in checkpoint and not args.reset_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("已加载优化器状态")
    else:
        print("使用新初始化的优化器")
        
    # 加载学习率调度器（除非选择重置）
    if 'scheduler_state_dict' in checkpoint and not args.reset_lr:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"已加载学习率调度器状态, 当前学习率: {scheduler.get_last_lr()[0]:.6f}")
    else:
        print("使用新初始化的学习率调度器")
        
    # 返回起始轮次和已训练步数
    start_epoch = checkpoint.get('epoch', 0)
    steps_trained = checkpoint.get('steps', 0)
    
    return start_epoch, steps_trained