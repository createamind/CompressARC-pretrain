import math
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import time
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from rule_guided_vae import RuleGuidedVAE
from arc_dataset import *

from utils import compute_task_complexity

from hierarchical_vae import VectorQuantizer
from checkpoint_reload import add_checkpoint_reload_functionality, load_checkpoint





import logging





# 在现有的train_rule_guided.py文件中添加以下函数

def train_rule_guided_from_pretrained(
    pretrained_path,
    data_path,
    save_dir,
    epochs=200,
    batch_size=4,
    learning_rate=5e-4,
    recon_weight=5.0,
    vq_weight=0.1,
    rule_weight=1.0,
    gpu_id=None,
    resume_path="",
    val_split=0.1,
    finetune_ae=True,
    seed=42
):
    """
    从预训练的自编码器继续训练规则提取和应用能力

    参数:
        pretrained_path: 预训练的自编码器权重路径
        finetune_ae: 是否微调自编码器
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置设备
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"rule_train_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(run_dir, "training.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    # 加载数据并划分训练/验证集
    dataset = get_arc_dataset(data_path)
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * val_split))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_arc_tasks
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_arc_tasks
    )

    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 初始化模型
    model = RuleGuidedVAE(
        grid_size=30,
        num_categories=10,
        pixel_codebook_size=512,
        object_codebook_size=256,
        rule_codebook_size=128,
        pixel_dim=64,
        object_dim=128,
        relation_dim=64,
        rule_dim=128
    )
    model.to(device)

    # 加载预训练的自编码器权重
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info(f"已加载预训练的自编码器权重: {pretrained_path}")

    # 确定哪些参数需要训练
    if finetune_ae:
        # 微调自编码器 + 训练规则相关参数
        trainable_params = model.parameters()
        logging.info("训练所有参数（微调自编码器 + 规则组件）")
    else:
        # 冻结自编码器参数
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.object_encoder.parameters():
            param.requires_grad = False

        # 只训练规则相关参数
        trainable_params = []
        trainable_params.extend(model.rule_encoder.parameters())
        trainable_params.extend(model.rule_applier.parameters())
        trainable_params.extend(model.rule_quantizer.parameters())
        trainable_params.extend(model.object_quantizer.parameters())  # 包括对象量化器
        logging.info("冻结自编码器参数，仅训练规则相关组件")

    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.0007,
    max_lr=0.002,
    step_size_up=10,
    cycle_momentum=False
)

    # 检查是否有需要恢复的训练断点
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        logging.info(f"从断点恢复: {resume_path}")
        logging.info(f"继续训练从轮次 {start_epoch}, 最佳验证损失: {best_val_loss:.6f}")

    # 定义用于评估的指标函数
    def compute_metrics(pred, target):
        """计算评估指标"""
        # 转换为类别索引
        pred_idx = torch.argmax(pred, dim=2)
        target_idx = torch.argmax(target, dim=1).reshape(*pred_idx.shape)

        # 计算像素准确率
        pixel_acc = (pred_idx == target_idx).float().mean().item()

        # 计算非背景像素准确率
        non_bg_mask = (target_idx != 0)
        if torch.any(non_bg_mask):
            non_bg_acc = ((pred_idx == target_idx) & non_bg_mask).float().sum() / non_bg_mask.float().sum()
            non_bg_acc = non_bg_acc.item()
        else:
            non_bg_acc = 1.0

        # 计算是否完全解决任务
        task_solved = torch.all(pred_idx == target_idx, dim=1).float().mean().item()

        return {
            'pixel_acc': pixel_acc,
            'non_bg_acc': non_bg_acc,
            'task_solved': task_solved
        }

    # 验证函数
    def validate_model(model, val_loader, device):
        """验证模型性能"""
        model.eval()
        val_losses = defaultdict(float)
        val_metrics = defaultdict(float)
        total_tasks = 0

        with torch.no_grad():
            for batch_tasks in val_loader:
                for task in batch_tasks:
                    train_examples = task['train']
                    test_input = task['test']['input'].to(device).unsqueeze(0)
                    test_output = task['test']['output'].to(device).unsqueeze(0)

                    # 随机选择一个训练样例
                    train_idx = np.random.randint(0, len(train_examples))
                    train_input, train_output = train_examples[train_idx]
                    train_input = train_input.to(device).unsqueeze(0)
                    train_output = train_output.to(device).unsqueeze(0)

                    # 提取规则并应用
                    rule = model.extract_rule(train_input, train_output)
                    pred_output = model.apply_rule(test_input, rule)

                    # 计算损失
                    recon_loss = F.cross_entropy(
                        pred_output.reshape(-1, model.num_categories),
                        torch.argmax(test_output, dim=1).reshape(-1)
                    )

                    vq_loss = rule.get('vq_loss', 0)
                    if not isinstance(vq_loss, float):
                        vq_loss = vq_loss.item()

                    total_loss = recon_weight * recon_loss + vq_weight * vq_loss

                    val_losses['total'] += total_loss.item()
                    val_losses['recon'] += recon_loss.item()
                    val_losses['vq'] += vq_loss

                    # 计算指标
                    metrics = compute_metrics(pred_output, test_output)
                    for k, v in metrics.items():
                        val_metrics[k] += v

                    total_tasks += 1

        # 计算平均值
        avg_losses = {k: v / total_tasks for k, v in val_losses.items()}
        avg_metrics = {k: v / total_tasks for k, v in val_metrics.items()}

        model.train()
        return avg_losses, avg_metrics

    # 训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = defaultdict(float)
        train_metrics = defaultdict(float)
        total_tasks = 0
        nan_count = 0
        epoch_start_time = time.time()

        for batch_tasks in train_dataloader:
            for task in batch_tasks:
                # 提取训练样例和测试样例
                train_examples = task['train']
                test_input = task['test']['input'].to(device).unsqueeze(0)
                test_output = task['test']['output'].to(device).unsqueeze(0)

                # 随机选择一个训练样例
                train_idx = np.random.randint(0, len(train_examples))
                train_input, train_output = train_examples[train_idx]
                train_input = train_input.to(device).unsqueeze(0)
                train_output = train_output.to(device).unsqueeze(0)

                # 前向传播
                optimizer.zero_grad()

                # 提取规则
                rule = model.extract_rule(train_input, train_output)

                # 应用规则预测测试输出
                pred_output = model.apply_rule(test_input, rule)

                # 计算损失
                recon_loss = F.cross_entropy(
                    pred_output.reshape(-1, model.num_categories),
                    torch.argmax(test_output, dim=1).reshape(-1)
                )

                vq_loss = rule.get('vq_loss', 0)
                total_loss = recon_weight * recon_loss + vq_weight * vq_loss

                # 检查NaN
                if torch.isnan(total_loss):
                    logging.warning(f"警告: 任务 {task.get('id', '未知')} 检测到NaN损失，跳过更新")
                    nan_count += 1
                    continue

                # 反向传播
                total_loss.backward()

                # 检查NaN梯度
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        logging.warning(f"警告: 任务 {task.get('id', '未知')} 检测到NaN梯度，跳过更新")
                        break

                if has_nan_grad:
                    nan_count += 1
                    continue

                # 梯度裁剪和更新
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                # 记录损失
                train_losses['total'] += total_loss.item()
                train_losses['recon'] += recon_loss.item()
                train_losses['vq'] += vq_loss if isinstance(vq_loss, float) else vq_loss.item()

                # 计算指标
                with torch.no_grad():
                    metrics = compute_metrics(pred_output, test_output)
                    for k, v in metrics.items():
                        train_metrics[k] += v

                total_tasks += 1

        # 更新学习率
        scheduler.step()

        # 计算平均指标
        avg_losses = {k: v / total_tasks for k, v in train_losses.items()} if total_tasks > 0 else {}
        avg_metrics = {k: v / total_tasks for k, v in train_metrics.items()} if total_tasks > 0 else {}

        # 验证
        logging.info("执行验证...")
        val_avg_losses, val_metrics = validate_model(model, val_dataloader, device)

        # 计算轮次耗时
        epoch_time = time.time() - epoch_start_time

        # 打印进度
        log_message = (f"轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s\n"
                       f"训练 - 损失: {avg_losses.get('total', 0):.4f}, 准确率: {avg_metrics.get('pixel_acc', 0):.4f}, "
                       f"任务解决率: {avg_metrics.get('task_solved', 0):.4f}\n"
                       f"验证 - 损失: {val_avg_losses.get('total', 0):.4f}, 准确率: {val_metrics.get('pixel_acc', 0):.4f}, "
                       f"任务解决率: {val_metrics.get('task_solved', 0):.4f}")

        if nan_count > 0:
            log_message += f"\nNaN梯度/损失出现次数: {nan_count}"

        logging.info(log_message)

        # 记录最佳验证损失
        if val_avg_losses.get('total', float('inf')) < best_val_loss:
            best_val_loss = val_avg_losses['total']
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': val_metrics
            }, best_model_path)
            logging.info(f"新的最佳模型已保存! 验证损失: {best_val_loss:.6f}")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(run_dir, f"rule_guided_vae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            logging.info(f"检查点已保存到 {checkpoint_path}")

    logging.info(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    return model, run_dir




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



def plot_results(task_id, input_grid, output_grid, predicted_grid, save_path):
    """修复后的结果可视化函数，处理不同格式的预测输出"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 转换input和output的one-hot为颜色网格
    input_colors = torch.argmax(input_grid, dim=0).cpu().numpy()
    output_colors = torch.argmax(output_grid, dim=0).cpu().numpy()

    # 处理predicted_grid的不同格式
    if predicted_grid.ndim == 1:
        # 如果是1D数组 (10,)，它可能是单个网格单元的类别概率
        print(f"预测是1D数组: {predicted_grid.shape}")
        # 创建一个空的1x1网格并设置唯一单元格的颜色
        pred_colors = np.zeros((1, 1), dtype=np.int32)
        pred_colors[0, 0] = torch.argmax(predicted_grid).item()
    elif predicted_grid.ndim == 2:
        # 如果是2D数组，可能是 (grid_cells=900, num_categories=10)
        # 或者已经是 (30, 30) 的索引数组
        if predicted_grid.shape[1] == 10:  # 格式: (grid_cells, num_categories)
            print(f"预测是扁平化的类别分布: {predicted_grid.shape}")
            pred_colors = torch.argmax(predicted_grid, dim=1).cpu().numpy()
            # 重塑为30x30网格
            grid_size = int(np.sqrt(pred_colors.shape[0]))
            pred_colors = pred_colors.reshape(grid_size, grid_size)
        else:
            # 已经是颜色索引数组
            print(f"预测已经是索引数组: {predicted_grid.shape}")
            pred_colors = predicted_grid.cpu().numpy()
    elif predicted_grid.ndim == 3:
        # 如果是3D数组，可能是 (batch=1, grid_cells=900, num_categories=10)
        # 或者是 (num_categories=10, height=30, width=30)
        if predicted_grid.shape[0] == 10:  # one-hot格式 (C, H, W)
            # print(f"预测是one-hot格式: {predicted_grid.shape}")
            pred_colors = torch.argmax(predicted_grid, dim=0).cpu().numpy()
        else:
            # 批次格式 (B, grid_cells, C)
            print(f"预测是批次格式: {predicted_grid.shape}")
            pred_probs = predicted_grid[0]  # 取第一个批次样本
            pred_colors = torch.argmax(pred_probs, dim=1).cpu().numpy()
            # 重塑为网格
            grid_size = int(np.sqrt(pred_colors.shape[0]))
            pred_colors = pred_colors.reshape(grid_size, grid_size)
    else:
        print(f"无法识别的预测格式: {predicted_grid.shape}")
        pred_colors = np.zeros((5, 5), dtype=np.int32)

    # print(f"处理后的形状: input={input_colors.shape}, output={output_colors.shape}, pred={pred_colors.shape}")

    # 裁剪网格以显示实际内容
    def crop_grid(grid):
        nonzero = np.where(grid > 0)
        if len(nonzero) == 2 and len(nonzero[0]) > 0:
            min_h, min_w = np.min(nonzero[0]), np.min(nonzero[1])
            max_h, max_w = np.max(nonzero[0]), np.max(nonzero[1])
            # 添加小边距
            min_h = max(0, min_h - 1)
            min_w = max(0, min_w - 1)
            max_h = min(grid.shape[0] - 1, max_h + 1)
            max_w = min(grid.shape[1] - 1, max_w + 1)
            return grid[min_h:max_h+1, min_w:max_w+1]
        return grid[:5, :5]  # 默认显示左上角5x5区域

    input_display = crop_grid(input_colors)
    output_display = crop_grid(output_colors)
    pred_display = crop_grid(pred_colors)

    # 绘制网格
    axes[0].imshow(input_display, vmin=0, vmax=9)
    axes[0].set_title('Input')
    axes[0].grid(True, color='black', linewidth=0.5)

    axes[1].imshow(output_display, vmin=0, vmax=9)
    axes[1].set_title('Expected Output')
    axes[1].grid(True, color='black', linewidth=0.5)

    axes[2].imshow(pred_display, vmin=0, vmax=9)
    axes[2].set_title('Predicted Output')
    axes[2].grid(True, color='black', linewidth=0.5)

    plt.suptitle(f'Task: {task_id}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def enhance_model_stability(model, clip_value=0.1):
    """增强模型训练稳定性的各种技巧"""
    # 1. 使用较小的梯度裁剪阈值
    original_clip_value = 0.5
    # print(f"增强稳定性: 梯度裁剪阈值从 {original_clip_value} 降低到 {clip_value}")

    # 2. 为所有BatchNorm和LayerNorm层添加eps
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.LayerNorm, nn.GroupNorm)):
            module.eps = 1e-5

    # 3. 为VQ模块设置更保守的commitment_cost
    for name, module in model.named_modules():
        if isinstance(module, VectorQuantizer):
            old_cost = module.commitment_cost
            module.commitment_cost = 0.1  # 降低commitment_cost值
            # print(f"增强稳定性: VQ模块 {name} 的commitment_cost从 {old_cost} 调整为 {module.commitment_cost}")

    return clip_value  # 返回新的裁剪阈值









# 新增评估函数
def evaluate_model(model, dataloader, device, recon_weight=5.0, vq_weight=0.1):
    """评估模型在给定数据加载器上的性能"""
    model.eval()
    eval_losses = defaultdict(float)
    correct_predictions = 0
    total_pixels = 0
    total_tasks = 0

    with torch.no_grad():
        for batch_tasks in dataloader:
            for task in batch_tasks:
                task_id = task['id']
                train_examples = task['train']
                test_input = task['test']['input'].to(device)
                test_output = task['test']['output'].to(device)

                # 选择第一个训练样例
                train_idx = 0
                train_input, train_output = train_examples[train_idx]
                train_input = train_input.to(device).unsqueeze(0)
                train_output = train_output.to(device).unsqueeze(0)
                test_input = test_input.unsqueeze(0)
                test_output = test_output.unsqueeze(0)

                # 1. 提取规则
                rule = model.extract_rule(train_input, train_output)

                # 2. 应用规则到测试输入
                predicted_output = model.apply_rule(test_input, rule)

                # 3. 计算重构损失
                recon_loss = F.cross_entropy(
                    predicted_output.reshape(-1, model.num_categories),
                    torch.argmax(test_output, dim=1).reshape(-1)
                )

                # 4. 计算VQ损失
                vq_loss = rule['vq_loss']

                # 5. 总损失
                total_loss = recon_weight * recon_loss + vq_weight * vq_loss

                # 记录损失
                eval_losses['total'] += total_loss.item()
                eval_losses['recon'] += recon_loss.item()
                eval_losses['vq'] += vq_loss.item()

                # 计算像素级准确率
                pred_classes = torch.argmax(predicted_output, dim=2).reshape(-1)
                true_classes = torch.argmax(test_output, dim=1).reshape(-1)

                correct_predictions += (pred_classes == true_classes).sum().item()
                total_pixels += true_classes.size(0)
                total_tasks += 1

    # 计算平均损失和准确率
    avg_losses = {k: v / total_tasks for k, v in eval_losses.items()}
    pixel_accuracy = correct_predictions / total_pixels if total_pixels > 0 else 0

    return avg_losses, pixel_accuracy


def train_rule_guided_vae(data_path, save_dir, epochs=50, batch_size=4,
                         learning_rate=1e-3, rule_weight=1.0, recon_weight=5.0,
                         vq_weight=0.1, gpu_id=None, resume_path="",
                         val_split=0.1, seed=42):  # 添加验证集参数
    """训练规则引导VAE模型，带有验证集支持"""
    # 创建以时间戳命名的子目录
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # 保存运行参数
    params = {
        "timestamp": timestamp,
        "start_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "rule_weight": rule_weight,
        "recon_weight": recon_weight,
        "vq_weight": vq_weight,
        "val_split": val_split,  # 添加验证集比例
        "seed": seed  # 添加随机种子
    }

    with open(os.path.join(run_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # 创建结果目录
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    val_results_dir = os.path.join(run_dir, "validation_results")
    os.makedirs(val_results_dir, exist_ok=True)

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

    # 设置随机种子以确保可重现的数据集划分
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载并划分数据集
    full_dataset = get_arc_dataset(data_path)

    # 计算验证集大小
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # 随机划分数据集
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"数据集划分: 训练集 {train_size} 任务, 验证集 {val_size} 任务")

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=collate_arc_tasks
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=collate_arc_tasks
    )

    # 初始化模型
    model = RuleGuidedVAE(
        grid_size=30,
        num_categories=10,
        pixel_codebook_size=64,
        object_codebook_size=256,
        rule_codebook_size=128,
        pixel_dim=8,
        object_dim=128,
        relation_dim=64,
        rule_dim=128
    )
    model.to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # 混合精度训练
    use_amp = has_gpu
    scaler = GradScaler() if use_amp else None

    print(f"训练设备: {device}")
    if has_gpu:
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB / "
              f"{torch.cuda.get_device_properties(device).total_memory/1024**3:.2f}GB")
    print(f"训练集大小: {len(train_dataset)} 任务")
    print(f"验证集大小: {len(val_dataset)} 任务")
    print(f"批量大小: {batch_size} 任务/批次")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"结果将保存到: {run_dir}")

    # 尝试恢复训练
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        args = argparse.Namespace()
        args.reset_optimizer = False
        args.reset_lr = False
        try:
            start_epoch, _ = load_checkpoint(resume_path, model, optimizer, scheduler, args)
            print(f"成功加载检查点，从轮次 {start_epoch} 继续训练")
        except Exception as e:
            print(f"加载检查点失败: {e}")
            print("继续使用初始化模型训练")

    # 记录最佳验证损失以用于模型选择
    best_val_loss = float('inf')
    best_epoch = -1

    # 用于绘制训练和验证曲线
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()

        epoch_losses = defaultdict(float)
        task_count = 0
        correct_train_predictions = 0
        total_train_pixels = 0

        for batch_tasks in train_dataloader:

            for task_idx, task in enumerate(batch_tasks):
                # 安全获取任务ID
                try:
                    if isinstance(task, dict) and 'id' in task:
                        task_id = task['id']
                    else:
                        task_id = f"task_{task_count}"
                        if isinstance(task, dict):
                            print(f"警告: 任务缺少ID字段，键: {list(task.keys())}")
                        else:
                            print(f"警告: 任务不是字典而是 {type(task)}")
                except Exception as e:
                    task_id = f"task_{task_count}"
                    print(f"获取任务ID时发生错误: {e}")

            # for task in batch_tasks:
            #     task_id = task['id']
                train_examples = task['train']
                test_input = task['test']['input'].to(device)
                test_output = task['test']['output'].to(device)

                # 如果任务只有一个训练样例，直接使用
                # 否则，随机选择一个作为示例
                train_idx = 0 if len(train_examples) == 1 else np.random.randint(0, len(train_examples))
                train_input, train_output = train_examples[train_idx]
                train_input = train_input.to(device).unsqueeze(0)  # 添加批次维度
                train_output = train_output.to(device).unsqueeze(0)
                test_input = test_input.unsqueeze(0)
                test_output = test_output.unsqueeze(0)

                # 使用混合精度训练
                if use_amp:
                    with autocast(device_type=device.type):
                        # 1. 提取规则
                        rule = model.extract_rule(train_input, train_output)

                        # 2. 应用规则到测试输入
                        predicted_output = model.apply_rule(test_input, rule)

                        # 3. 计算重构损失
                        recon_loss = F.cross_entropy(
                            predicted_output.reshape(-1, model.num_categories),
                            torch.argmax(test_output, dim=1).reshape(-1)
                        )

                        # 4. 计算VQ损失
                        vq_loss = rule['vq_loss']

                        # 5. 总损失
                        total_loss = recon_weight * recon_loss + vq_weight * vq_loss

                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()

                    # 检查梯度
                    has_nan_grad = False
                    try:
                        scaler.unscale_(optimizer)

                        for param in model.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    has_nan_grad = True
                                    break

                        if not has_nan_grad:
                            clip_value = enhance_model_stability(model)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                    except RuntimeError as e:
                        print(f"警告: 缩放器错误: {e}")
                        has_nan_grad = True

                    if has_nan_grad:
                        print(f"警告: 任务 {task_id} 检测到NaN梯度，跳过更新")
                        if use_amp:
                            scaler.update()
                    else:
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                # else:
                else:
                    # 非混合精度训练
                    # 1. 提取规则
                    rule = model.extract_rule(train_input, train_output)

                    # 2. 应用规则到测试输入
                    predicted_output = model.apply_rule(test_input, rule)

                    # 3. 计算重构损失
                    recon_loss = F.cross_entropy(
                        predicted_output.reshape(-1, model.num_categories),
                        torch.argmax(test_output, dim=1).reshape(-1)
                    )

                    # 4. 计算VQ损失
                    vq_loss = rule['vq_loss']

                    # 5. 总损失
                    total_loss = recon_weight * recon_loss + vq_weight * vq_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()

                    # 检查梯度
                    has_nan_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break

                    if has_nan_grad:
                        print(f"警告: 任务 {task_id} 检测到NaN梯度，跳过更新")
                    else:
                        clip_value = enhance_model_stability(model)

                        # 在反向传播处使用新的裁剪阈值
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                    # 非混合精度训练部分保持原样
                    # ...

                # 计算训练准确率
                with torch.no_grad():
                    pred_classes = torch.argmax(predicted_output, dim=2).reshape(-1)
                    true_classes = torch.argmax(test_output, dim=1).reshape(-1)

                    correct_train_predictions += (pred_classes == true_classes).sum().item()
                    total_train_pixels += true_classes.size(0)

                # 记录损失
                epoch_losses['total'] += total_loss.item()
                epoch_losses['recon'] += recon_loss.item()
                epoch_losses['vq'] += vq_loss.item()
                task_count += 1

                # 定期可视化训练结果
                if task_count % 10 == 0 or task_count == len(train_dataset):
                    # 保存预测结果图像
                    save_path = os.path.join(
                        results_dir,
                        f'epoch_{epoch+1}_task_{task_id.replace(".json", "")}.png'
                    )
                    with torch.no_grad():
                        # 修改：正确处理解码器输出格式
                        pred_output = F.softmax(predicted_output[0], dim=1)
                        grid_size = int(math.sqrt(pred_output.size(0)))
                        reshaped_pred = pred_output.view(grid_size, grid_size, -1).permute(2, 0, 1)
                        plot_results(task_id, test_input[0], test_output[0], reshaped_pred, save_path)

        # 更新学习率
        scheduler.step()

        # 计算训练平均损失和准确率
        avg_train_losses = {k: v / task_count for k, v in epoch_losses.items()}
        train_accuracy = correct_train_predictions / total_train_pixels if total_train_pixels > 0 else 0

        train_losses.append(avg_train_losses['total'])
        train_accuracies.append(train_accuracy)

        # 验证阶段
        # print("执行验证...")
        val_avg_losses, val_accuracy = evaluate_model(
            model, val_dataloader, device, recon_weight, vq_weight
        )

        val_losses.append(val_avg_losses['total'])
        val_accuracies.append(val_accuracy)

        # 可视化部分验证样本
        model.eval()
        with torch.no_grad():
            for i, batch_tasks in enumerate(val_dataloader):
                if i > 2:  # 只可视化前几个批次
                    break

                for task in batch_tasks:
                    task_id = task['id']
                    train_examples = task['train']
                    test_input = task['test']['input'].to(device)
                    test_output = task['test']['output'].to(device)

                    # 使用第一个训练样例
                    train_input, train_output = train_examples[0]
                    train_input = train_input.to(device).unsqueeze(0)
                    train_output = train_output.to(device).unsqueeze(0)
                    test_input = test_input.unsqueeze(0)
                    test_output = test_output.unsqueeze(0)

                    # 提取规则并应用
                    rule = model.extract_rule(train_input, train_output)
                    predicted_output = model.apply_rule(test_input, rule)

                    # 保存验证预测结果
                    save_path = os.path.join(
                        val_results_dir,
                        f'epoch_{epoch+1}_val_{task_id.replace(".json", "")}.png'
                    )

                    # 处理输出格式
                    pred_output = F.softmax(predicted_output[0], dim=1)
                    grid_size = int(math.sqrt(pred_output.size(0)))
                    reshaped_pred = pred_output.view(grid_size, grid_size, -1).permute(2, 0, 1)

                    plot_results(task_id, test_input[0], test_output[0], reshaped_pred, save_path)

        # 检查是否为最佳模型
        if val_avg_losses['total'] < best_val_loss:
            best_val_loss = val_avg_losses['total']
            best_epoch = epoch + 1

            # 保存最佳模型
            best_model_path = os.path.join(run_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': val_accuracy
            }, best_model_path)
            print(f"新的最佳模型已保存! 验证损失: {best_val_loss:.4f}")

        # 打印epoch摘要
        epoch_time = time.time() - start_time
        # print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s')
        print(f'训练 - 损失: {avg_train_losses["total"]:.4f}, 准确率: {train_accuracy:.4f}')
        print(f'验证 - 损失: {val_avg_losses["total"]:.4f}, 准确率: {val_accuracy:.4f}')
        # print(f'最佳轮次: {best_epoch}, 最佳验证损失: {best_val_loss:.4f}')
        # print(f'学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, '
              f'平均损失: {avg_train_losses["total"]:.4f}, '
              f'平均重构: {avg_train_losses["recon"]:.4f}, '
              f'平均VQ: {avg_train_losses["vq"]:.4f}, '
            #   f'NaN梯度次数: {epoch_nan_count}, '
              f'学习率: {scheduler.get_last_lr()[0]:.6f}')
        print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, '
              f'平均损失: {val_avg_losses["total"]:.4f}, '
              f'平均重构: {val_avg_losses["recon"]:.4f}, '
              f'平均VQ: {val_avg_losses["vq"]:.4f}, '
            #   f'NaN梯度次数: {epoch_nan_count}, '
              f'学习率: {scheduler.get_last_lr()[0]:.6f}')

        # 保存训练日志
        log_data = {
            "epoch": epoch + 1,
            "train_loss": avg_train_losses["total"],
            "train_recon_loss": avg_train_losses["recon"],
            "train_vq_loss": avg_train_losses["vq"],
            "train_accuracy": train_accuracy,
            "val_loss": val_avg_losses["total"],
            "val_recon_loss": val_avg_losses["recon"],
            "val_vq_loss": val_avg_losses["vq"],
            "val_accuracy": val_accuracy,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss
        }

        log_path = os.path.join(run_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # 绘制损失和准确率曲线
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # 绘制损失曲线
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
            plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证损失')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.title('训练与验证损失')
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='训练准确率')
            plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='验证准确率')
            plt.xlabel('轮次')
            plt.ylabel('准确率')
            plt.legend()
            plt.title('训练与验证准确率')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f'learning_curves_epoch_{epoch+1}.png'))
            plt.close()

        # 保存模型检查点
        if (epoch + 1) % 13 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(run_dir, f'rule_guided_vae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_losses["total"],
                'val_loss': val_avg_losses["total"],
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            print(f'检查点已保存到 {checkpoint_path}')

    # 保存最终模型
    final_model_path = os.path.join(run_dir, f'final_rule_guided_vae.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }, final_model_path)
    print(f'最终模型已保存到 {final_model_path}')

    # 打印训练总结
    print("\n训练总结:")
    print(f"最佳验证损失: {best_val_loss:.4f} (轮次 {best_epoch})")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print(f"最终训练准确率: {train_accuracies[-1]:.4f}")
    print(f"最终验证准确率: {val_accuracies[-1]:.4f}")

    return model, run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Rule-Guided VAE on ARC tasks")
    parser.add_argument("--data", type=str, default="data/training",
                        help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/rule_guided/",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of tasks per batch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--rule_weight", type=float, default=1.0, help="Weight for rule loss")
    parser.add_argument("--recon_weight", type=float, default=5.0, help="Weight for reconstruction loss")
    parser.add_argument("--vq_weight", type=float, default=0.1, help="Weight for VQ loss")
    parser.add_argument("--gpu", type=int, default=0, help="Specific GPU to use")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Percentage of data to use for validation (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser = add_checkpoint_reload_functionality(parser)

    args = parser.parse_args()

    model, run_dir = train_rule_guided_vae(
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        rule_weight=args.rule_weight,
        recon_weight=args.recon_weight,
        vq_weight=args.vq_weight,
        gpu_id=args.gpu,
        resume_path=args.resume,
        val_split=args.val_split,
        seed=args.seed
    )

    print(f"训练完成，结果保存在: {run_dir}")
