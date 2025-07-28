import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
import json
import torch.nn.functional as F
from model import DiscreteVAE
from discrete_vae import DiscreteVAE as FullDiscreteVAE
from utils import compute_task_complexity, grid_augmentation
import time
from collections import defaultdict

# Dataset class with curriculum learning capability
class ARCDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []  # List of (file_path, complexity) tuples
        self.task_data = {}  # Cache for task data
        self.complexity_percentile = 1.0  # Default: use all samples
        self.current_indices = []  # Indices of samples to use in current curriculum stage
        self.load_data()
    
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        start_time = time.time()
        
        # Load all tasks and compute complexities
        all_tasks = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    self.task_data[file_path] = task_data
                    complexity = compute_task_complexity(task_data)
                    all_tasks.append((file_path, complexity, task_data))
        
        # Sort by complexity
        all_tasks.sort(key=lambda x: x[1])
        
        # Convert tasks to grid samples
        for file_path, complexity, task_data in all_tasks:
            for train in task_data['train']:
                input_grid = np.array(train['input'])
                self.samples.append((input_grid, complexity, file_path))
        
        # Initialize current indices to all samples
        self.current_indices = list(range(len(self.samples)))
        
        print(f"Loaded {len(self.samples)} examples from {len(all_tasks)} tasks")
        print(f"Complexity range: {self.samples[0][1]:.2f} - {self.samples[-1][1]:.2f}")
        print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    def set_complexity_threshold(self, percentile):
        """Set curriculum threshold to use only samples below the given percentile"""
        self.complexity_percentile = percentile
        if percentile >= 1.0:
            self.current_indices = list(range(len(self.samples)))
        else:
            # Calculate complexity threshold based on percentile
            sorted_complexities = sorted([sample[1] for sample in self.samples])
            threshold_idx = int(percentile * len(sorted_complexities))
            threshold = sorted_complexities[threshold_idx]
            
            # Update indices
            self.current_indices = [
                i for i, (_, complexity, _) in enumerate(self.samples) 
                if complexity <= threshold
            ]
        
        print(f"Curriculum updated: Using {len(self.current_indices)}/{len(self.samples)} samples ({percentile*100:.0f}%)")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # Map index to current curriculum subset
        real_idx = self.current_indices[idx]
        grid, _, _ = self.samples[real_idx]
        
        # Apply data augmentation
        if np.random.random() < 0.5:  # 50% chance of augmentation
            grid = grid_augmentation(grid)
        
        # Convert grid to one-hot encoding
        h, w = grid.shape
        one_hot = np.zeros((10, 30, 30))  # Fixed size for all grids
        
        # Copy original grid data to fixed-size array
        for i in range(min(h, 30)):
            for j in range(min(w, 30)):
                one_hot[grid[i, j], i, j] = 1
        
        # Create target grid (padded to 30x30)
        target_grid = np.zeros((30, 30), dtype=np.long)
        target_grid[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        
        # Convert to tensors
        return torch.tensor(one_hot, dtype=torch.float), torch.tensor(target_grid, dtype=torch.long)

# 标准VAE损失函数（带周期性KL退火）
def vae_loss(reconstruction, x, mu, logvar, beta_weight, current_step, total_steps):
    """
    带周期性KL退火的VAE损失函数
    """
    batch_size = x.size(0)
    
    # 扁平化目标张量
    x_flat = x.reshape(batch_size, -1)
    
    # 计算重构损失
    recon_loss = 0
    for i in range(x_flat.size(1)):  # 对每个位置
        logits = reconstruction[:, i, :]  # [batch_size, num_categories]
        target = x_flat[:, i]            # [batch_size]
        recon_loss += F.cross_entropy(logits, target)
    
    # 平均所有位置的损失
    recon_loss = recon_loss / x_flat.size(1)
    
    # 周期性KL权重退火
    min_beta = 0.05  # 提高最小值
    max_beta = 1.2   # 略微提高最大值
    
    # 使用周期性退火而不是单调增加
    cycle_length = total_steps // 3  # 每个周期为训练总步数的1/3
    cycle_position = (current_step % cycle_length) / cycle_length
    beta = min_beta + (max_beta - min_beta) * 0.5 * (1 + np.cos(cycle_position * np.pi))
    
    # KL散度计算
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size
    
    # 总损失
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss, beta * kld

# 离散VAE损失函数（重构损失 + VQ损失）
def discrete_vae_loss(reconstruction, x, mu, logvar, vq_loss, beta_kl=0.1, beta_vq=1.0, current_step=0, total_steps=1):
    """
    离散VAE损失函数，结合VQ损失和可选的KL损失
    """
    batch_size = x.size(0)
    
    # 扁平化目标张量
    x_flat = x.reshape(batch_size, -1)
    
    # 计算重构损失
    recon_loss = 0
    for i in range(x_flat.size(1)):  # 对每个位置
        logits = reconstruction[:, i, :]  # [batch_size, num_categories]
        target = x_flat[:, i]            # [batch_size]
        recon_loss += F.cross_entropy(logits, target)
    
    # 平均所有位置的损失
    recon_loss = recon_loss / x_flat.size(1)
    
    # 周期性KL权重退火（如果需要）
    min_beta = 0.01
    max_beta = beta_kl
    cycle_length = total_steps // 3
    cycle_position = (current_step % cycle_length) / cycle_length
    kl_weight = min_beta + (max_beta - min_beta) * 0.5 * (1 + np.cos(cycle_position * np.pi))
    
    # KL散度计算
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size
    
    # 总损失 = 重构损失 + KL损失 + VQ损失
    total_loss = recon_loss + kl_weight * kld + beta_vq * vq_loss
    
    return total_loss, recon_loss, kl_weight * kld, beta_vq * vq_loss

def train_model(data_path, save_dir, epochs=100, batch_size=32, learning_rate=1e-3, 
               beta=1.0, accumulation_steps=4, use_amp=True, use_residual=True,
               use_discrete_vae=True, codebook_size=512, embedding_dim=64):
    """
    训练VAE模型，可选择使用标准VAE或完全离散VAE
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化数据集和数据加载器
    dataset = ARCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True, num_workers=4, pin_memory=True)
    
    # 根据选项初始化模型
    if use_discrete_vae:
        print("使用完全离散化VAE模型")
        model = FullDiscreteVAE(
            grid_size=30,
            num_categories=10,
            hidden_dim=512,
            latent_dim=256,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
    else:
        print("使用标准VAE模型")
        model = DiscreteVAE(use_residual=use_residual)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 增加权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    
    # 调整学习率策略
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=5e-5)
    
    # 初始化梯度缩放器
    scaler = GradScaler() if use_amp else None
    
    print(f"训练设备: {device}")
    print(f"数据集大小: {len(dataset)} 样本")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"使用自动混合精度: {use_amp}")
    if not use_discrete_vae:
        print(f"使用残差连接: {use_residual}")
    else:
        print(f"编码本大小: {codebook_size}")
        print(f"嵌入维度: {embedding_dim}")
    print(f"使用梯度累积: {accumulation_steps} 步")
    print(f"权重衰减: 5e-4")
    
    # 计算总步数
    total_steps = epochs * (len(dataloader) // accumulation_steps)
    current_step = 0
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        vq_loss_sum = 0
        
        # 加快课程学习进度
        if epoch < epochs * 0.15:  # 减半时间
            dataset.set_complexity_threshold(0.3)
        elif epoch < epochs * 0.35:  # 减半时间
            dataset.set_complexity_threshold(0.7)
        elif epoch < epochs * 0.6:  # 添加中间阶段
            dataset.set_complexity_threshold(0.85)
        else:
            dataset.set_complexity_threshold(1.0)
        
        # 重置优化器
        optimizer.zero_grad()
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # 使用可选的混合精度前向传播
            if use_amp:
                with autocast():
                    if use_discrete_vae:
                        recon_batch, mu, logvar, quantized, vq_loss, indices = model(data)
                        loss, recon_term, kl_term, vq_term = discrete_vae_loss(
                            recon_batch, target, mu, logvar, vq_loss, beta_kl=0.1, beta_vq=1.0,
                            current_step=current_step, total_steps=total_steps
                        )
                    else:
                        recon_batch, mu, logvar = model(data)
                        loss, recon_term, kl_term = vae_loss(
                            recon_batch, target, mu, logvar, beta, current_step, total_steps)
                        vq_term = torch.tensor(0.0).to(device)
                
                # 使用缩放器的反向传播
                scaler.scale(loss / accumulation_steps).backward()
            else:
                # 标准前向传播
                if use_discrete_vae:
                    recon_batch, mu, logvar, quantized, vq_loss, indices = model(data)
                    loss, recon_term, kl_term, vq_term = discrete_vae_loss(
                        recon_batch, target, mu, logvar, vq_loss, beta_kl=0.1, beta_vq=1.0,
                        current_step=current_step, total_steps=total_steps
                    )
                else:
                    recon_batch, mu, logvar = model(data)
                    loss, recon_term, kl_term = vae_loss(
                        recon_batch, target, mu, logvar, beta, current_step, total_steps)
                    vq_term = torch.tensor(0.0).to(device)
                
                # 标准反向传播
                (loss / accumulation_steps).backward()
            
            # 记录损失
            total_loss += loss.item()
            recon_loss_sum += recon_term.item()
            kl_loss_sum += kl_term.item()
            if use_discrete_vae:
                vq_loss_sum += vq_term.item()
            
            # 累积梯度后更新权重
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    # 缩放梯度进行裁剪
                    scaler.unscale_(optimizer)
                    
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if use_amp:
                    # 使用缩放器更新
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准更新
                    optimizer.step()
                
                # 重置梯度
                optimizer.zero_grad()
                
                # 增加步计数器
                current_step += 1
            
            # 打印进度
            if batch_idx % 10 == 0:
                if use_discrete_vae:
                    print(f'轮次: {epoch+1}/{epochs}, 批次: {batch_idx}/{len(dataloader)}, '
                         f'损失: {loss.item():.4f}, 重构: {recon_term.item():.4f}, '
                         f'KL: {kl_term.item():.4f}, VQ: {vq_term.item():.4f}')
                else:
                    # 获取当前KL权重
                    cycle_length = total_steps // 3
                    cycle_position = (current_step % cycle_length) / cycle_length
                    current_beta = 0.05 + (1.2 - 0.05) * 0.5 * (1 + np.cos(cycle_position * np.pi))
                    
                    print(f'轮次: {epoch+1}/{epochs}, 批次: {batch_idx}/{len(dataloader)}, '
                         f'损失: {loss.item():.4f}, 重构: {recon_term.item():.4f}, '
                         f'KL: {kl_term.item():.4f}, KL权重: {current_beta:.4f}')
        
        # 更新学习率调度器
        scheduler.step()
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_loss_sum / len(dataloader)
        avg_kl = kl_loss_sum / len(dataloader)
        
        # 打印epoch摘要
        epoch_time = time.time() - start_time
        if use_discrete_vae:
            avg_vq = vq_loss_sum / len(dataloader)
            print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, 平均损失: {avg_loss:.4f}, '
                  f'平均重构: {avg_recon:.4f}, 平均KL: {avg_kl:.4f}, 平均VQ: {avg_vq:.4f}, '
                  f'学习率: {scheduler.get_last_lr()[0]:.6f}')
        else:
            print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, 平均损失: {avg_loss:.4f}, '
                  f'平均重构: {avg_recon:.4f}, 平均KL: {avg_kl:.4f}, 学习率: {scheduler.get_last_lr()[0]:.6f}')
        
        # 保存模型检查点
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model_type = "discrete" if use_discrete_vae else "standard"
            checkpoint_path = os.path.join(save_dir, f'{model_type}_model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f'检查点已保存到 {checkpoint_path}')
    
    # 保存最终模型
    model_type = "discrete" if use_discrete_vae else "standard"
    final_model_path = os.path.join(save_dir, f'final_{model_type}_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存到 {final_model_path}')
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VAE models on ARC tasks")
    parser.add_argument("--data", type=str, default="/home/zdx/github/VSAHDC/ARC-AGI-2/data/training", 
                        help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence term")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_residual", action="store_true", help="Disable residual connections")
    parser.add_argument("--standard_vae", action="store_true", help="Use standard VAE instead of discrete VAE")
    parser.add_argument("--codebook_size", type=int, default=512, help="Codebook size for discrete VAE")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for discrete VAE")
    parser.add_argument("--bg_threshold", type=int, default=40, help="Background color threshold percentage")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data, 
        save_dir=args.save_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr, 
        beta=args.beta,
        accumulation_steps=args.accumulation_steps,
        use_amp=not args.no_amp,
        use_residual=not args.no_residual,
        use_discrete_vae=not args.standard_vae,
        codebook_size=args.codebook_size,
        embedding_dim=args.embedding_dim
    )