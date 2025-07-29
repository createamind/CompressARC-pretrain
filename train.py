import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
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
import datetime
from hierarchical_vae import ObjectOrientedHierarchicalVAE


# Dataset class with curriculum learning capability
class ARCDataset(Dataset):
    # [保持不变]...
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

# 标准VAE损失函数和离散VAE损失函数保持不变...
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

# 离散VAE损失函数
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

def check_gpu():
    """检查并详细报告GPU状态"""
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU训练 (速度会非常慢)")
        return False, "cpu", None

    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("警告: 虽然CUDA可用，但未找到可用的GPU设备，将使用CPU训练")
        return False, "cpu", None

    # 打印所有可用GPU的详细信息
    print(f"检测到 {device_count} 个GPU:")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} ({total_memory:.2f} GB)")

    # 默认使用第一个GPU
    device_id = 0
    device = f"cuda:{device_id}"

    # 测试内存分配
    try:
        test_tensor = torch.zeros((100, 100), device=device)
        del test_tensor
        print(f"成功在 {device} 上分配测试张量")
    except Exception as e:
        print(f"警告: GPU内存分配测试失败: {e}")
        print("将尝试继续使用GPU，但可能会遇到问题")

    return True, device, device_id


def train_model(data_path, save_dir, epochs=100, batch_size=32, learning_rate=1e-3,
               beta=1.0, accumulation_steps=4, use_amp=True, use_residual=True,
               use_discrete_vae=True, codebook_size=512, embedding_dim=64, gpu_id=None,
               use_hierarchical_vae=False, recon_weight=1.0, vq_weight=1.0):
    """
    训练VAE模型，可选择使用标准VAE或完全离散VAE
    """
    # 创建以时间戳命名的子目录
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 保存运行参数到JSON文件
    params = {
        "timestamp": timestamp,
        "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "beta": beta,
        "accumulation_steps": accumulation_steps,
        "use_amp": use_amp,
        "use_residual": use_residual,
        "use_discrete_vae": use_discrete_vae,
        "codebook_size": codebook_size,
        "embedding_dim": embedding_dim,
    }

    with open(os.path.join(run_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # 检查GPU状态
    has_gpu, device_str, detected_gpu_id = check_gpu()

    # 如果指定了GPU ID，优先使用指定的GPU
    if gpu_id is not None and has_gpu:
        if gpu_id >= torch.cuda.device_count():
            print(f"警告: 指定的GPU ID {gpu_id} 超出可用GPU数量范围，将使用默认GPU")
        else:
            device_str = f"cuda:{gpu_id}"
            print(f"使用指定的GPU {gpu_id}")

    device = torch.device(device_str)

    # 初始化数据集和数据加载器
    dataset = ARCDataset(data_path)

    # 设置数据加载器的工作进程数 (根据CPU核心数)
    num_workers = min(4, os.cpu_count() or 1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=num_workers,
                            pin_memory=has_gpu)  # 使用pin_memory加速GPU传输

    # 根据选项初始化模型

    if use_hierarchical_vae:
        print("使用层次化面向对象VAE模型")
        model = ObjectOrientedHierarchicalVAE(
            grid_size=30,
            num_categories=10,
            pixel_codebook_size=512,
            object_codebook_size=256,
            relation_codebook_size=128,
            pixel_dim=64,
            object_dim=128,
            relation_dim=64
        )
    elif use_discrete_vae:
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



    # 将模型移至设备
    model.to(device)
    print(f"模型已移至设备: {device}")

    # 增加权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # 调整学习率策略
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=5e-5)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 从3增加到10
        T_mult=1,  # 从2减少到1，保持周期稳定
        eta_min=1e-4  # 从5e-5增加到1e-4，提高最小学习率
    )

    # 只有在GPU上才使用混合精度训练
    use_amp = use_amp and has_gpu
    scaler = GradScaler() if use_amp else None

    print(f"训练设备: {device}")
    if has_gpu:
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB / "
              f"{torch.cuda.get_device_properties(device).total_memory/1024**3:.2f}GB")
    print(f"数据集大小: {len(dataset)} 样本")
    print(f"数据加载器工作进程数: {num_workers}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"使用自动混合精度: {use_amp}")
    if not use_discrete_vae:
        print(f"使用残差连接: {use_residual}")
    else:
        print(f"编码本大小: {codebook_size}")
        print(f"嵌入维度: {embedding_dim}")
    print(f"使用梯度累积: {accumulation_steps} 步")
    print(f"权重衰减: 5e-4")
    print(f"结果将保存到: {run_dir}")

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
        if epoch < epochs * 0.1:  # 减半时间
            dataset.set_complexity_threshold(0.8)
        elif epoch < epochs * 0.3:  # 减半时间
            dataset.set_complexity_threshold(0.9)
        elif epoch < epochs * 0.6:  # 添加中间阶段
            dataset.set_complexity_threshold(1.0)
        else:
            dataset.set_complexity_threshold(1.0)

        # 重置优化器
        optimizer.zero_grad()

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(dataloader):
            # 确保数据在正确的设备上
            data, target = data.to(device), target.to(device)

            # 使用可选的混合精度前向传播
            if use_amp:
                with autocast(device_type=device.type):
                    if use_hierarchical_vae:
                        recon_batch, _, _, _, vq_loss, _ = model(data)
                        # 层次化VAE只有重构损失和VQ损失
                        recon_term = 0
                        target_flat = target.reshape(batch_size, -1)
                        for i in range(target_flat.size(1)):
                            logits = recon_batch[:, i, :]
                            recon_term += F.cross_entropy(logits, target_flat[:, i])
                        recon_term = recon_term / target_flat.size(1)

                        # 检查并处理异常损失值
                        if torch.isnan(vq_loss) or torch.isinf(vq_loss) or vq_loss > 1e5:
                            print(f"警告: VQ损失异常 ({vq_loss.item():.4f})，使用小常数替代")
                            vq_loss = torch.tensor(0.1, device=device, requires_grad=True)

                        loss = recon_weight * recon_term + vq_weight * vq_loss
                        kl_term = torch.tensor(0.0).to(device)
                        vq_term = vq_loss

                        # 检查总损失
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告: 总损失异常，重置为仅重构损失")
                            loss = recon_weight * recon_term

                    elif use_discrete_vae:
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

                # 更严格的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                # 检查梯度是否含有NaN
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print("警告: 检测到NaN梯度，跳过此更新步骤")
                    optimizer.zero_grad()
                else:
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
                if has_gpu:
                    gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                    mem_str = f", GPU内存: {gpu_mem:.2f}GB"
                else:
                    mem_str = ""

                if use_discrete_vae:
                    print(f'轮次: {epoch+1}/{epochs}, 批次: {batch_idx}/{len(dataloader)}, '
                         f'损失: {loss.item():.4f}, 重构: {recon_term.item():.4f}, '
                         f'KL: {kl_term.item():.4f}, VQ: {vq_term.item():.4f}{mem_str}')
                else:
                    # 获取当前KL权重
                    cycle_length = total_steps // 3
                    cycle_position = (current_step % cycle_length) / cycle_length
                    current_beta = 0.05 + (1.2 - 0.05) * 0.5 * (1 + np.cos(cycle_position * np.pi))

                    print(f'轮次: {epoch+1}/{epochs}, 批次: {batch_idx}/{len(dataloader)}, '
                         f'损失: {loss.item():.4f}, 重构: {recon_term.item():.4f}, '
                         f'KL: {kl_term.item():.4f}, KL权重: {current_beta:.4f}{mem_str}')

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

        # 记录训练日志
        log_data = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "vq_loss": avg_vq if use_discrete_vae else 0.0,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time
        }

        # 保存训练日志
        log_path = os.path.join(run_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # 保存模型检查点
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model_type = "discrete" if use_discrete_vae else "standard"
            checkpoint_path = os.path.join(run_dir, f'{model_type}_model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f'检查点已保存到 {checkpoint_path}')

        # 在评估阶段添加
        # if epoch % 5 == 0:
            # 收集一批数据的编码索引
            indices_list = []
            with torch.no_grad():
                for data, _ in itertools.islice(dataloader, 10):  # 取10批
                    _, _, _, _, _, indices = model(data.to(device))
                    indices_list.extend(indices.cpu().numpy().flatten())

            # 计算编码本使用率
            unique_indices = np.unique(indices_list)
            usage_ratio = len(unique_indices) / model.pixel_vq.num_embeddings
            print(f"编码本使用率: {usage_ratio:.2%}, 使用{len(unique_indices)}/{model.pixel_vq.num_embeddings}个编码")

    # 保存最终模型
    model_type = "discrete" if use_discrete_vae else "standard"
    final_model_path = os.path.join(run_dir, f'final_{model_type}_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存到 {final_model_path}')

    return model, run_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VAE models on ARC tasks")
    parser.add_argument("--data", type=str, default="/home/zdx/github/VSAHDC/ARC-AGI-2/data/training",
                        help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence term")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_residual", action="store_true", help="Disable residual connections")
    parser.add_argument("--standard_vae", action="store_true", help="Use standard VAE instead of discrete VAE")
    parser.add_argument("--codebook_size", type=int, default=512, help="Codebook size for discrete VAE")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for discrete VAE")
    parser.add_argument("--bg_threshold", type=int, default=40, help="Background color threshold percentage")
    parser.add_argument("--gpu", type=int, default=0, help="Specific GPU to use (e.g. 0, 1, etc)")
    parser.add_argument("--hierarchical_vae", action="store_true", help="Use hierarchical object-oriented VAE")
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--vq_weight", type=float, default=1.0, help="Weight for VQ loss")


    args = parser.parse_args()

    model, run_dir = train_model(
        use_hierarchical_vae=args.hierarchical_vae,
        recon_weight=args.recon_weight,
        vq_weight=args.vq_weight,
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
        embedding_dim=args.embedding_dim,
        gpu_id=args.gpu
    )

    print(f"训练完成，结果保存在: {run_dir}")