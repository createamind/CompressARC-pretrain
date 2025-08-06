import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
import logging
from datetime import datetime

from models.autoencoder import ARCAutoencoder
from utils.data_utils import create_arc_dataloader
from utils.logging_utils import setup_logging, log_model_summary, log_metrics_to_json
from utils.accuracy_utils import calculate_accuracies


def train_autoencoder(args):
    """训练ARC网格自编码器模型"""
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"arc_autoencoder_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置日志
    log_dir = os.path.join(run_dir, 'logs')
    setup_logging(log_dir, name="arc_autoencoder", log_level=logging.DEBUG if args.debug else logging.INFO)
    
    # 记录配置
    logging.info(f"运行ARC自编码器训练 - {run_name}")
    logging.info(f"配置: {args}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        logging.info(f"已设置随机种子: {args.seed}")
    
    # 加载数据
    logging.info(f"从 {args.data_dir} 加载ARC数据...")
    train_loader, val_loader, dataset = create_arc_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_grid_size=args.max_grid_size,
        pad_to_size=args.pad_size,
        val_split=args.val_split
    )
    logging.info(f"数据加载完成。训练集: {len(train_loader) * args.batch_size} 样本, "
                f"验证集: {len(val_loader) * args.batch_size} 样本")
    
    # 创建模型
    model = ARCAutoencoder(
        num_categories=args.num_categories,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    )
    model.to(device)
    log_model_summary(model)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8  # 增加数值稳定性
    )
    
    # 学习率调度器
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - args.warmup_epochs,
            eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif args.lr_scheduler == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    
    # 用于记录最佳模型
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # 训练指标跟踪
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_bg_acc': [], 'val_bg_acc': [],
        'train_fg_acc': [], 'val_fg_acc': [],
        'train_bg_ratio': [], 'val_bg_ratio': [],
        'learning_rates': []
    }
    
    # 训练循环
    logging.info("开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 学习率预热
        if epoch <= args.warmup_epochs and args.lr_scheduler != 'cosine_warmup':
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * (epoch / args.warmup_epochs)
            logging.info(f"学习率预热: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练一个轮次
        train_loss, train_acc, train_bg_acc, train_fg_acc, train_bg_ratio = \
            train_epoch(model, train_loader, optimizer, device, args)
        
        # 评估
        val_loss, val_acc, val_bg_acc, val_fg_acc, val_bg_ratio = \
            evaluate(model, val_loader, device, args)
        
        # 更新学习率
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        metrics['learning_rates'].append(current_lr)
        
        # 添加指标
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_bg_acc'].append(train_bg_acc)
        metrics['val_bg_acc'].append(val_bg_acc)
        metrics['train_fg_acc'].append(train_fg_acc)
        metrics['val_fg_acc'].append(val_fg_acc)
        metrics['train_bg_ratio'].append(train_bg_ratio)
        metrics['val_bg_ratio'].append(val_bg_ratio)
        
        # 保存最佳模型(按验证损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(run_dir, f'best_model_loss.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(args)
            }, model_path)
            logging.info(f"保存最佳损失模型: {model_path}")
        
        # 保存最佳模型(按验证准确率)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(run_dir, f'best_model_acc.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(args)
            }, model_path)
            logging.info(f"保存最佳准确率模型: {model_path}")
        
        # 定期保存检查点
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'metrics': metrics,
                'config': vars(args)
            }, checkpoint_path)
            logging.info(f"保存检查点: {checkpoint_path}")
        
        # 保存指标数据
        metrics_json = os.path.join(run_dir, 'metrics.json')
        log_metrics_to_json(metrics, metrics_json)
        
        # 打印进度
        epoch_time = time.time() - epoch_start_time
        logging.info(f"轮次 {epoch}/{args.epochs} 完成 - 耗时: {epoch_time:.2f}s")
        logging.info(f"  训练损失: {train_loss:.4f}")
        logging.info(f"  训练准确率: 总体 {train_acc:.2f}%, 背景 {train_bg_acc:.2f}%, "
                   f"前景 {train_fg_acc:.2f}%, 背景占比 {train_bg_ratio:.1f}%")
        logging.info(f"  验证损失: {val_loss:.4f}")
        logging.info(f"  验证准确率: 总体 {val_acc:.2f}%, 背景 {val_bg_acc:.2f}%, "
                   f"前景 {val_fg_acc:.2f}%, 背景占比 {val_bg_ratio:.1f}%")
        logging.info(f"  学习率: {current_lr:.6f}")
    
    # 保存最终模型
    final_model_path = os.path.join(run_dir, f'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'metrics': metrics,
        'config': vars(args)
    }, final_model_path)
    logging.info(f"保存最终模型: {final_model_path}")
    
    # 保存最终指标
    final_metrics_json = os.path.join(run_dir, 'final_metrics.json')
    log_metrics_to_json(metrics, final_metrics_json)
    
    logging.info(f"训练完成! 最佳验证损失: {best_val_loss:.4f}, 最佳验证准确率: {best_val_acc:.2f}%")
    logging.info(f"结果保存至: {run_dir}")
    
    return model, metrics, run_dir


def train_epoch(model, dataloader, optimizer, device, args):
    """训练一个轮次"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 准确率统计
    all_acc = 0
    all_bg_acc = 0
    all_fg_acc = 0
    all_bg_ratio = 0
    
    # 如果是调试模式，只使用少量批次
    max_batches = len(dataloader)
    if args.debug:
        max_batches = min(max_batches, args.debug_batches)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        # 获取网格数据和掩码
        grids = batch['grid'].to(device)
        masks = batch['mask'].to(device) if 'mask' in batch else None
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(grids)
        
        # 计算重建损失 - 使用加权交叉熵，增加前景像素权重
        if args.weighted_loss:
            weights = torch.ones(args.num_categories, device=device)
            weights[0] = 0.2  # 背景色(通常是0)权重降低
            recon_loss = F.cross_entropy(
                outputs.view(-1, args.num_categories), 
                grids.view(-1),
                weight=weights,
                reduction='mean'
            )
        else:
            # 如果有掩码，只计算有效区域的损失
            if masks is not None:
                flat_outputs = outputs.permute(0, 2, 3, 1).reshape(-1, args.num_categories)
                flat_grids = grids.reshape(-1)
                flat_masks = masks.reshape(-1)
                valid_outputs = flat_outputs[flat_masks]
                valid_targets = flat_grids[flat_masks]
                recon_loss = F.cross_entropy(valid_outputs, valid_targets)
            else:
                recon_loss = F.cross_entropy(outputs.view(-1, args.num_categories), grids.view(-1))
        
        # 总损失
        loss = recon_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪(防止梯度爆炸)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 计算准确率 - 区分背景和前景
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            batch_acc, batch_bg_acc, batch_fg_acc, batch_bg_ratio = calculate_accuracies(
                pred, grids, masks, background_threshold=args.bg_threshold
            )
            
            all_acc += batch_acc
            all_bg_acc += batch_bg_acc
            all_fg_acc += batch_fg_acc
            all_bg_ratio += batch_bg_ratio
        
        num_batches += 1
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_acc = all_acc / num_batches
    avg_bg_acc = all_bg_acc / num_batches
    avg_fg_acc = all_fg_acc / num_batches
    avg_bg_ratio = all_bg_ratio / num_batches
    
    return avg_loss, avg_acc, avg_bg_acc, avg_fg_acc, avg_bg_ratio


def evaluate(model, dataloader, device, args):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 准确率统计
    all_acc = 0
    all_bg_acc = 0
    all_fg_acc = 0
    all_bg_ratio = 0
    
    # 如果是调试模式，只使用少量批次
    max_batches = len(dataloader)
    if args.debug:
        max_batches = min(max_batches, args.debug_batches)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            # 获取网格数据和掩码
            grids = batch['grid'].to(device)
            masks = batch['mask'].to(device) if 'mask' in batch else None
            
            # 前向传播
            outputs = model(grids)
            
            # 计算重建损失
            if args.weighted_loss:
                weights = torch.ones(args.num_categories, device=device)
                weights[0] = 0.2  # 背景色(通常是0)权重降低
                recon_loss = F.cross_entropy(
                    outputs.view(-1, args.num_categories), 
                    grids.view(-1),
                    weight=weights,
                    reduction='mean'
                )
            else:
                # 如果有掩码，只计算有效区域的损失
                if masks is not None:
                    flat_outputs = outputs.permute(0, 2, 3, 1).reshape(-1, args.num_categories)
                    flat_grids = grids.reshape(-1)
                    flat_masks = masks.reshape(-1)
                    valid_outputs = flat_outputs[flat_masks]
                    valid_targets = flat_grids[flat_masks]
                    recon_loss = F.cross_entropy(valid_outputs, valid_targets)
                else:
                    recon_loss = F.cross_entropy(outputs.view(-1, args.num_categories), grids.view(-1))
            
            # 总损失
            loss = recon_loss
            
            # 累计损失
            total_loss += loss.item()
            
            # 计算准确率
            pred = outputs.argmax(dim=1)
            batch_acc, batch_bg_acc, batch_fg_acc, batch_bg_ratio = calculate_accuracies(
                pred, grids, masks, background_threshold=args.bg_threshold
            )
            
            all_acc += batch_acc
            all_bg_acc += batch_bg_acc
            all_fg_acc += batch_fg_acc
            all_bg_ratio += batch_bg_ratio
            
            num_batches += 1
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    avg_acc = all_acc / num_batches
    avg_bg_acc = all_bg_acc / num_batches
    avg_fg_acc = all_fg_acc / num_batches
    avg_bg_ratio = all_bg_ratio / num_batches
    
    return avg_loss, avg_acc, avg_bg_acc, avg_fg_acc, avg_bg_ratio


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ARC网格自编码器训练')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default='data/training',
                        help='ARC数据目录路径')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='输出目录路径')
    parser.add_argument('--max-grid-size', type=int, default=30,
                        help='最大处理的网格大小')
    parser.add_argument('--pad-size', type=int, default=None,
                        help='填充网格到固定大小，None表示不填充')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='验证集比例')
    
    # 模型相关参数
    parser.add_argument('--num-categories', type=int, default=10,
                        help='颜色类别数量')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='潜在表示维度')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 128],
                        help='编码器/解码器隐藏层维度')
    
    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批量大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载的工作线程数')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'cosine_warmup'],
                        help='学习率调度器类型')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='学习率预热轮次')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--weighted-loss', action='store_true',
                        help='使用加权损失函数，前景像素权重更高')
    parser.add_argument('--bg-threshold', type=int, default=40,
                        help='背景像素判定阈值(百分比)')
    
    # 保存相关
    parser.add_argument('--save-interval', type=int, default=10,
                        help='保存检查点的轮次间隔')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备，例如：cpu, cuda, cuda:0')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--debug-batches', type=int, default=5,
                        help='调试模式下处理的批次数')
    
    args = parser.parse_args()
    return args


def main():
    """主函数入口"""
    # 解析命令行参数
    args = parse_args()
    
    # 训练模型
    model, metrics, run_dir = train_autoencoder(args)
    
    return model, metrics, run_dir


if __name__ == "__main__":
    main()