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
from fix_training_stall import *

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
    # print(f"处理预测输出，原始形状: {predicted_grid.shape}")

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

    # print(f"处理后的形状: input={input_colors.shape}, output={output_colors.shape}, pred={pred_colors.shape}")

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









def enhance_model_stability(model):
    """应用稳定性增强措施"""

    # 检查模型是否已应用稳定性修复
    if not hasattr(model, '_stability_enhanced'):
        print("应用模型稳定性增强...")

        # 1. 将解码器中的 LeakyReLU 替换为 ReLU
        for name, module in model.named_modules():
            if isinstance(module, nn.LeakyReLU) and 'decoder' in name:
                # 获取父模块和位置
                path = name.split('.')
                parent = model
                for part in path[:-1]:
                    parent = getattr(parent, part)

                # 替换激活函数
                setattr(parent, path[-1], nn.ReLU())

        # 2. 增加 GroupNorm 的 eps 值
        for name, module in model.named_modules():
            if isinstance(module, nn.GroupNorm):
                module.eps = 1e-5

        # 3. 对最终层应用特殊初始化
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'final_decoder'):
            final_layer = None
            for module in model.decoder.final_decoder:
                if isinstance(module, nn.Conv2d):
                    final_layer = module

            if final_layer:
                with torch.no_grad():
                    nn.init.xavier_uniform_(final_layer.weight, gain=0.5)
                    nn.init.zeros_(final_layer.bias)

        # 标记模型已增强
        model._stability_enhanced = True
        # 初始使用小的裁剪值
        return 0.25
    else:
        # 已增强模型使用常规裁剪值
        return 0.5


from nan_gradient_debugger import NaNGradientDebugger
from fix_decoder_output import fix_decoder_output_format

def train_rule_guided_vae(data_path, save_dir, epochs=50, batch_size=4,
                         learning_rate=1e-3, rule_weight=1.0, recon_weight=5.0,
                         vq_weight=0.1, gpu_id=None):
    """训练规则引导VAE模型 - 简化并增强稳定性的版本"""
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

    # 应用模型稳定性增强
    # clip_value = enhance_model_stability(model)


    # 初始化NaN梯度调试器
    debugger = NaNGradientDebugger(model, log_dir=os.path.join(run_dir, "nan_debug"))
    debugger.register_hooks()
    print("已启用NaN梯度追踪与分析")

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0007,
        max_lr=0.002,
        step_size_up=10,
        cycle_momentum=False
    )

    fixed_plot_results = fix_decoder_output_format(model)

    clip_value = fix_training_stall(model, optimizer)

    print(f"训练设备: {device}")
    if has_gpu:
        print(f"GPU内存使用情况: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB / "
              f"{torch.cuda.get_device_properties(device).total_memory/1024**3:.2f}GB")
    print(f"数据集大小: {len(dataset)} 任务")
    print(f"批量大小: {batch_size} 任务/批次")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"结果将保存到: {run_dir}")

    # NaN梯度追踪变量
    consecutive_nan = 0
    nan_threshold = 5

    # 训练循环
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        epoch_losses = defaultdict(float)
        task_count = 0
        epoch_nan_count = 0  # 本轮次NaN计数

        for batch_tasks in dataloader:
            for task in batch_tasks:
                task_id = task['id']
                train_examples = task['train']
                test_input = task['test']['input'].to(device)
                test_output = task['test']['output'].to(device)

                # 记录当前任务输入，以便在出现NaN时分析
                debugger.save_input_on_nan(task_id, test_input)

                # 如果任务只有一个训练样例，直接使用
                # 否则，随机选择一个作为示例
                train_idx = 0 if len(train_examples) == 1 else np.random.randint(0, len(train_examples))
                train_input, train_output = train_examples[train_idx]
                train_input = train_input.to(device).unsqueeze(0)  # 添加批次维度
                train_output = train_output.to(device).unsqueeze(0)
                test_input = test_input.unsqueeze(0)
                test_output = test_output.unsqueeze(0)

                # 1. 提取规则
                rule = model.extract_rule(train_input, train_output)

                # 2. 应用规则到测试输入
                predicted_output = model.apply_rule(test_input, rule)

                # 3. 计算重构损失 - 使用稳定版本
                # recon_loss = stable_cross_entropy(
                #     predicted_output.reshape(-1, model.num_categories),
                #     torch.argmax(test_output, dim=1).reshape(-1),
                #     epsilon=1e-7
                # )
                recon_loss = unstable_cross_entropy(
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
                    debugger.log_nan_task(task_id)  # 记录NaN任务
                    epoch_nan_count += 1
                    consecutive_nan += 1

                    # 连续NaN超过阈值时降低学习率
                    if consecutive_nan >= nan_threshold:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.8
                        print(f"检测到连续{consecutive_nan}次NaN梯度，学习率降低至{optimizer.param_groups[0]['lr']:.6f}")
                        clip_value = max(0.1, clip_value * 0.8)  # 降低裁剪阈值
                        print(f"梯度裁剪阈值降低至{clip_value:.3f}")
                        consecutive_nan = 0  # 重置计数
                else:
                    consecutive_nan = 0  # 重置连续NaN计数

                    # 应用分层梯度裁剪
                    # layer_specific_gradient_clipping(model, optimizer, clip_value)
                    verify_func = verify_parameter_updates(model, optimizer)

                    # 在此处替换梯度裁剪函数
                    # layer_specific_gradient_clipping(model, optimizer, clip_value)
                    modified_layer_specific_gradient_clipping(model, optimizer, clip_value)

                    # 更新参数
                    optimizer.step()
                    verify_func()

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

                    # 在训练循环中
                    with torch.no_grad():
                        # 获取预测输出
                        predicted_outputs = model(test_input)

                        # 处理元组输出
                        if isinstance(predicted_outputs, tuple):
                            prediction = predicted_outputs[0]
                        else:
                            prediction = predicted_outputs

                        # 使用第一个批次样本
                        pred = prediction[0]

                        # 可视化
                        fixed_plot_results(task_id, test_input[0], test_output[0], pred, save_path)

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
              f'NaN梯度次数: {epoch_nan_count}, '
              f'学习率: {scheduler.get_last_lr()[0]:.6f}')

        # 每5个epoch分析NaN梯度情况
        if (epoch + 1) % 5 == 0:
            print(f"\n===== 轮次 {epoch+1} NaN梯度分析 =====")
            debugger.analyze_and_report()

        # 保存训练日志
        log_data = {
            "epoch": epoch + 1,
            "loss": avg_losses["total"],
            "recon_loss": avg_losses["recon"],
            "vq_loss": avg_losses["vq"],
            "nan_count": epoch_nan_count,
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

    # 训练结束，生成最终NaN梯度分析报告
    print("\n===== 训练结束，最终NaN梯度分析 =====")
    debugger.analyze_and_report()

    # 清理钩子
    debugger.remove_hooks()

    # 保存最终模型
    final_model_path = os.path.join(run_dir, f'final_rule_guided_vae.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'最终模型已保存到 {final_model_path}')

    return model, run_dir


def stable_cross_entropy(pred, target, epsilon=1e-7):
    """数值稳定的交叉熵损失"""
    # 确保预测输出在有效范围内
    if isinstance(pred, torch.Tensor) and pred.requires_grad:
        # 只有当需要梯度时才使用clamp
        pred = torch.clamp(pred, min=epsilon, max=1.0-epsilon)

    # 计算交叉熵
    return F.cross_entropy(pred, target)

def layer_specific_gradient_clipping(model, optimizer, global_clip_value=0.5):
    """针对不同层应用不同的梯度裁剪"""
    # 定义层特定的裁剪值
    clip_settings = {
        'decoder.final_decoder': global_clip_value * 0.2,   # 最终解码器层使用更严格的裁剪
        'decoder.low_decoder': global_clip_value * 0.5,     # 低级解码器使用中等裁剪
        'decoder': global_clip_value * 0.8,                 # 其他解码器部分使用较大裁剪
        'default': global_clip_value                        # 默认使用标准裁剪值
    }

    # 按层分组参数
    param_groups = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        # 分配到合适的组
        group_name = 'default'
        for key in clip_settings:
            if key in name:
                group_name = key
                break

        if group_name not in param_groups:
            param_groups[group_name] = []
        param_groups[group_name].append(param)

    # 对每组应用不同的裁剪
    for group_name, params in param_groups.items():
        if params:
            torch.nn.utils.clip_grad_norm_(params, clip_settings[group_name])



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