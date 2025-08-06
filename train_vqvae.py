import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import logging
from datetime import datetime

from models.vqvae import ArcGridVQVAE
from utils.data_utils import create_arc_dataloader
from utils.logging_utils import setup_logging, log_model_summary, save_codebook_usage_plot, log_metrics_to_json
from utils.visualization import visualize_reconstructions, plot_training_metrics
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG, DEBUG_CONFIG


def train_vqvae(args):
    """训练ARC网格VQVAE模型"""
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"arc_vqvae_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置日志
    log_dir = os.path.join(run_dir, 'logs')
    setup_logging(log_dir, name="arc_vqvae", log_level=logging.DEBUG if args.debug else logging.INFO)
    
    # 记录配置
    logging.info(f"运行VQVAE训练 - {run_name}")
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
    model = ArcGridVQVAE(
        num_categories=args.num_categories,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        hidden_dims=args.hidden_dims,
        use_ema=args.use_ema
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
    
    # 用于记录最佳模型
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    # 训练指标跟踪
    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_recon_loss': [], 'val_recon_loss': [],
        'train_vq_loss': [], 'val_vq_loss': [],
        'train_perplexity': [], 'val_perplexity': [],
        'train_accuracy': [], 'val_accuracy': [],
        'learning_rates': []
    }
    
    # 训练循环
    logging.info("开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 学习率预热
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate * (epoch / args.warmup_epochs)
            logging.info(f"学习率预热: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练一个轮次
        train_loss, train_recon_loss, train_vq_loss, train_perplexity, train_accuracy = \
            train_epoch(model, train_loader, optimizer, device, args)
        
        # 评估
        val_loss, val_recon_loss, val_vq_loss, val_perplexity, val_accuracy = \
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
        metrics['train_recon_loss'].append(train_recon_loss)
        metrics['val_recon_loss'].append(val_recon_loss)
        metrics['train_vq_loss'].append(train_vq_loss)
        metrics['val_vq_loss'].append(val_vq_loss)
        metrics['train_perplexity'].append(train_perplexity)
        metrics['val_perplexity'].append(val_perplexity)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)
        
        # 保存最佳模型(按验证损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(run_dir, f'best_model_loss.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'config': vars(args)
            }, model_path)
            logging.info(f"保存最佳损失模型: {model_path}")
        
        # 保存最佳模型(按验证准确率)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model_path = os.path.join(run_dir, f'best_model_acc.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
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
                'val_accuracy': val_accuracy,
                'metrics': metrics,
                'config': vars(args)
            }, checkpoint_path)
            logging.info(f"保存检查点: {checkpoint_path}")
        
        # 可视化重建结果
        if args.vis_interval > 0 and epoch % args.vis_interval == 0:
            vis_dir = os.path.join(run_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 可视化重建
            vis_path = os.path.join(vis_dir, f'reconstructions_epoch_{epoch}.png')
            visualize_reconstructions(model, val_loader, device, num_samples=8, save_path=vis_path)
            
            # 可视化编码本使用情况
            codebook_path = os.path.join(vis_dir, f'codebook_usage_epoch_{epoch}.png')
            save_codebook_usage_plot(model, codebook_path, epoch)
            
            # 绘制训练指标
            metrics_path = os.path.join(vis_dir, f'metrics_epoch_{epoch}.png')
            plot_training_metrics(metrics, metrics_path)
            
            # 保存指标数据
            metrics_json = os.path.join(run_dir, 'metrics.json')
            log_metrics_to_json(metrics, metrics_json)
        
        # 打印进度
        epoch_time = time.time() - epoch_start_time
        logging.info(f"轮次 {epoch}/{args.epochs} 完成 - 耗时: {epoch_time:.2f}s")
        logging.info(f"  训练损失: {train_loss:.4f}, 重建: {train_recon_loss:.4f}, VQ: {train_vq_loss:.4f}, "
                    f"复杂度: {train_perplexity:.2f}, 准确率: {train_accuracy:.2f}%")
        logging.info(f"  验证损失: {val_loss:.4f}, 重建: {val_recon_loss:.4f}, VQ: {val_vq_loss:.4f}, "
                    f"复杂度: {val_perplexity:.2f}, 准确率: {val_accuracy:.2f}%")
        logging.info(f"  学习率: {current_lr:.6f}")
    
    # 保存最终模型
    final_model_path = os.path.join(run_dir, f'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
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
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    correct = 0
    total = 0
    num_batches = 0
    
    # 如果是调试模式，只使用少量批次
    max_batches = len(dataloader)
    if args.debug:
        max_batches = min(max_batches, args.debug_batches)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        # 获取网格数据
        grids = batch['grid'].to(device)
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs, vq_loss, perplexity = model(grids)
        
        # 计算重建损失(交叉熵)
        recon_loss = F.cross_entropy(outputs.view(-1, args.num_categories), grids.view(-1))
        
        # 总损失
        loss = recon_loss + args.vq_weight * vq_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪(防止梯度爆炸)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()
        
        # 计算准确率
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            correct += (pred == grids).float().sum().item()
            total += grids.numel()
        
        num_batches += 1
    
    # 计算平均损失和准确率
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_vq_loss = total_vq_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    accuracy = 100 * correct / total
    
    return avg_loss, avg_recon_loss, avg_vq_loss, avg_perplexity, accuracy


def evaluate(model, dataloader, device, args):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0
    correct = 0
    total = 0
    num_batches = 0
    
    # 如果是调试模式，只使用少量批次
    max_batches = len(dataloader)
    if args.debug:
        max_batches = min(max_batches, args.debug_batches)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            # 获取网格数据
            grids = batch['grid'].to(device)
            
            # 前向传播
            outputs, vq_loss, perplexity = model(grids)
            
            # 计算重建损失(交叉熵)
            recon_loss = F.cross_entropy(outputs.view(-1, args.num_categories), grids.view(-1))
            
            # 总损失
            loss = recon_loss + args.vq_weight * vq_loss
            
            # 累计损失
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            
            # 计算准确率
            pred = outputs.argmax(dim=1)
            correct += (pred == grids).float().sum().item()
            total += grids.numel()
            
            num_batches += 1
    
    # 计算平均损失和准确率
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_vq_loss = total_vq_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    accuracy = 100 * correct / total
    
    return avg_loss, avg_recon_loss, avg_vq_loss, avg_perplexity, accuracy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ARC网格VQVAE训练')
    
    # 数据相关参数
    parser.add_argument('--data-dir', type=str, default=DATA_CONFIG['data_path'],
                        help='ARC数据目录路径')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_CONFIG['checkpoint_dir'],
                        help='输出目录路径')
    parser.add_argument('--max-grid-size', type=int, default=DATA_CONFIG['max_grid_size'],
                        help='最大处理的网格大小')
    parser.add_argument('--pad-size', type=int, default=DATA_CONFIG['pad_to_size'],
                        help='填充网格到固定大小，None表示不填充')
    parser.add_argument('--val-split', type=float, default=DATA_CONFIG['val_split'],
                        help='验证集比例')
    
    # 模型相关参数
    parser.add_argument('--num-categories', type=int, default=MODEL_CONFIG['num_categories'],
                        help='颜色类别数量')
    parser.add_argument('--latent-dim', type=int, default=MODEL_CONFIG['latent_dim'],
                        help='潜在表示维度')
    parser.add_argument('--num-embeddings', type=int, default=MODEL_CONFIG['num_embeddings'],
                        help='VQ编码本大小')
    parser.add_argument('--commitment-cost', type=float, default=MODEL_CONFIG['commitment_cost'],
                        help='VQ承诺损失权重')
    parser.add_argument('--decay', type=float, default=MODEL_CONFIG['decay'],
                        help='EMA衰减率')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=MODEL_CONFIG['hidden_dims'],
                        help='编码器/解码器隐藏层维度')
    parser.add_argument('--use-ema', action='store_true', default=MODEL_CONFIG['use_ema'],
                        help='使用EMA更新编码本')
    
    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='批量大小')
    parser.add_argument('--num-workers', type=int, default=TRAIN_CONFIG['num_workers'],
                        help='数据加载的工作线程数')
    parser.add_argument('--learning-rate', type=float, default=TRAIN_CONFIG['learning_rate'],
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG['weight_decay'],
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['epochs'],
                        help='训练轮次')
    parser.add_argument('--lr-scheduler', type=str, default=TRAIN_CONFIG['lr_scheduler'],
                        choices=['cosine', 'step', 'plateau'],
                        help='学习率调度器类型')
    parser.add_argument('--warmup-epochs', type=int, default=TRAIN_CONFIG['warmup_epochs'],
                        help='学习率预热轮次')
    parser.add_argument('--grad-clip', type=float, default=TRAIN_CONFIG['grad_clip'],
                        help='梯度裁剪阈值')
    parser.add_argument('--vq-weight', type=float, default=1.0,
                        help='VQ损失权重')
    
    # 保存和可视化相关
    parser.add_argument('--save-interval', type=int, default=TRAIN_CONFIG['save_interval'],
                        help='保存检查点的轮次间隔')
    parser.add_argument('--eval-interval', type=int, default=TRAIN_CONFIG['eval_interval'],
                        help='评估的轮次间隔')
    parser.add_argument('--vis-interval', type=int, default=TRAIN_CONFIG['vis_interval'],
                        help='可视化的轮次间隔')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备，例如：cpu, cuda, cuda:0')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--debug', action='store_true', default=DEBUG_CONFIG['debug_mode'],
                        help='调试模式')
    parser.add_argument('--debug-batches', type=int, default=DEBUG_CONFIG['debug_batches'],
                        help='调试模式下处理的批次数')
    
    args = parser.parse_args()
    return args


def main():
    """主函数入口"""
    # 解析命令行参数
    args = parse_args()
    
    # 训练模型
    model, metrics, run_dir = train_vqvae(args)
    
    return model, metrics, run_dir


if __name__ == "__main__":
    main()