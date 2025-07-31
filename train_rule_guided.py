import math
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

from rule_guided_vae import RuleGuidedVAE
from arc_dataset import get_arc_dataloader
from utils import compute_task_complexity

from hierarchical_vae import                              VectorQuantizer


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



import torch
import numpy as np
import matplotlib.pyplot as plt

def robust_plot_results(task_id, input_grid, output_grid, predicted_grid, save_path):
    """健壮的可视化函数，可处理各种形状的预测输出"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 转换输入和输出的one-hot为颜色网格
    input_colors = torch.argmax(input_grid, dim=0).cpu().numpy()
    output_colors = torch.argmax(output_grid, dim=0).cpu().numpy()

    # 处理预测输出的形状问题
    print(f"处理预测输出，原始形状: {predicted_grid.shape}")

    if isinstance(predicted_grid, torch.Tensor):
        # 张量情况
        if predicted_grid.dim() == 4:  # [batch, C, H, W]
            pred_colors = torch.argmax(predicted_grid[0], dim=0).cpu().numpy()
        elif predicted_grid.dim() == 3:
            if predicted_grid.shape[0] == 10:  # [C, H, W]
                pred_colors = torch.argmax(predicted_grid, dim=0).cpu().numpy()
            else:  # [batch, H, W] 或其他
                pred_colors = predicted_grid[0].cpu().numpy()
        elif predicted_grid.dim() == 2:
            # 检查是否是 [H*W, C] 格式
            if predicted_grid.shape[1] == 10:
                # 尝试确定网格大小
                grid_size = int(np.sqrt(predicted_grid.shape[0] + 0.5))
                if abs(grid_size**2 - predicted_grid.shape[0]) < 5:  # 允许小误差
                    pred_colors = torch.argmax(predicted_grid, dim=1).cpu().numpy()
                    pred_colors = pred_colors[:grid_size*grid_size].reshape(grid_size, grid_size)
                else:
                    # 无法确定合理的网格形状，使用原始形状
                    pred_colors = torch.argmax(predicted_grid, dim=1).cpu().numpy()
                    pred_colors = pred_colors.reshape(-1, 1)  # 显示为单列
            else:
                pred_colors = predicted_grid.cpu().numpy()
        else:  # 1D张量
            # 使用一个小的网格来显示
            size = min(5, max(1, int(np.sqrt(predicted_grid.numel()))))
            pred_colors = predicted_grid.cpu().numpy()
            pred_colors = pred_colors[:size*size].reshape(size, size)
    else:
        # 非张量情况
        pred_colors = np.array([[0]])  # 默认显示

    print(f"处理后的形状: input={input_colors.shape}, output={output_colors.shape}, pred={pred_colors.shape}")

    # 绘制网格
    axes[0].imshow(input_colors, vmin=0, vmax=9)
    axes[0].set_title('Input')
    axes[0].grid(True, color='black', linewidth=0.5)

    axes[1].imshow(output_colors, vmin=0, vmax=9)
    axes[1].set_title('Expected Output')
    axes[1].grid(True, color='black', linewidth=0.5)

    axes[2].imshow(pred_colors, vmin=0, vmax=9)
    axes[2].set_title(f'Predicted Output {pred_colors.shape}')
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

def train_rule_guided_vae(data_path, save_dir, epochs=50, batch_size=4,
                         learning_rate=1e-3, rule_weight=1.0, recon_weight=5.0,
                         vq_weight=0.1, gpu_id=None):
    """训练规则引导VAE模型"""
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
        "vq_weight": vq_weight
    }

    with open(os.path.join(run_dir, "train_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # 创建结果目录
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

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

    # 初始化数据加载器
    dataloader, dataset = get_arc_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1)
    )

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
    print(f"数据集大小: {len(dataset)} 任务")
    print(f"批量大小: {batch_size} 任务/批次")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"结果将保存到: {run_dir}")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        epoch_losses = defaultdict(float)
        task_count = 0

        for batch_tasks in dataloader:
            for task in batch_tasks:
                task_id = task['id']
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
                    if use_amp:
                        try:
                            scaler.unscale_(optimizer)

                            for param in model.parameters():
                                if param.grad is not None:
                                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                        has_nan_grad = True
                                        break

                            if not has_nan_grad:
                                clip_value = enhance_model_stability(model)

                                # 在反向传播处使用新的裁剪阈值
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

                                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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

                # 记录损失
                epoch_losses['total'] += total_loss.item()
                epoch_losses['recon'] += recon_loss.item()
                epoch_losses['vq'] += vq_loss.item()
                task_count += 1

                # 定期可视化结果
                if task_count % 10 == 0 or task_count == len(dataset):
                    # 保存预测结果图像
                    save_path = os.path.join(
                        results_dir,
                        f'epoch_{epoch+1}_task_{task_id.replace(".json", "")}.png'
                    )

                    # with torch.no_grad():
                    #     # 修改：正确处理解码器输出格式
                    #     pred_output = F.softmax(predicted_output[0], dim=1)  # [grid_cells, num_categories]

                    #     # 将输出重塑为网格
                    #     grid_size = int(math.sqrt(pred_output.size(0)))
                    #     reshaped_pred = pred_output.view(grid_size, grid_size, -1).permute(2, 0, 1)  # [C, H, W]

                    #     plot_results(task_id, test_input[0], test_output[0], reshaped_pred, save_path)



                    # 在train_rule_guided.py中添加导入
                    # from fix_prediction_pipeline import direct_output_patch

                    from fix_tuple_output import handle_tuple_output

                    # 在训练循环中
                    with torch.no_grad():
                        # 获取预测输出
                        predicted_output = model(test_input)

                        # 处理元组输出
                        if isinstance(predicted_output, tuple):
                            prediction = handle_tuple_output(predicted_output)
                        else:
                            prediction = predicted_output

                        # 继续使用修复后的预测
                        print(f"处理后的预测形状: {prediction.shape}")

                        # 使用第一个批次样本
                        pred = prediction[0]

                        # 可视化
                        robust_plot_results(task_id, test_input[0], test_output[0], pred, save_path)


        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_losses = {k: v / task_count for k, v in epoch_losses.items()}

        # 打印epoch摘要
        epoch_time = time.time() - start_time
        print(f'轮次: {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, '
              f'平均损失: {avg_losses["total"]:.4f}, '
              f'平均重构: {avg_losses["recon"]:.4f}, '
              f'平均VQ: {avg_losses["vq"]:.4f}, '
              f'学习率: {scheduler.get_last_lr()[0]:.6f}')

        # 保存训练日志
        log_data = {
            "epoch": epoch + 1,
            "loss": avg_losses["total"],
            "recon_loss": avg_losses["recon"],
            "vq_loss": avg_losses["vq"],
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time
        }

        log_path = os.path.join(run_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # 保存模型检查点
        if (epoch + 1) % 13 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(run_dir, f'rule_guided_vae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_losses["total"]
            }, checkpoint_path)
            print(f'检查点已保存到 {checkpoint_path}')

    # 保存最终模型
    final_model_path = os.path.join(run_dir, f'final_rule_guided_vae.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存到 {final_model_path}')

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
        gpu_id=args.gpu
    )

    print(f"训练完成，结果保存在: {run_dir}")