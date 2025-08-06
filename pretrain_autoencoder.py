import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import numpy as np
import time
from collections import defaultdict
import json
import logging
# 导入混合精度训练模块
from torch.cuda.amp import autocast, GradScaler

from arc_dataset import get_arc_dataset, collate_arc_tasks
from rule_guided_vae import RuleGuidedVAE

def pretrain_autoencoder(
    data_path,
    save_dir,
    epochs=100,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    gpu_id=None,
    resume_path="",
    seed=42,
    # 添加混合精度开关参数
    use_amp=True
):
    """预训练自编码器组件"""
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 设置设备
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")
        # CPU不支持混合精度训练
        if device.type == "cpu":
            use_amp = False
    print(f"使用设备: {device}")
    print(f"混合精度训练: {'启用' if use_amp else '禁用'}")

    # 创建结果目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pretrain_dir = os.path.join(save_dir, f"pretrain_{timestamp}")
    os.makedirs(pretrain_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(pretrain_dir, "pretrain.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    # 加载数据集
    dataset = get_arc_dataset(data_path)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_arc_tasks,
        pin_memory=True,  # 加快数据传输
        persistent_workers=True  # 保持工作进程活跃
    )

    # 初始化模型
    model = RuleGuidedVAE(
        grid_size=30,
        num_categories=10,
        pixel_codebook_size=512,
        object_codebook_size=1024,
        rule_codebook_size=128,
        pixel_dim=64,
        object_dim=32,
        relation_dim=64,
        rule_dim=128
    )
    model.to(device)

    # 对模型进行编译优化（如果PyTorch版本支持）
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logging.info("已应用模型编译优化")
        except Exception as e:
            logging.info(f"模型编译失败: {e}")

    # 定义优化器，只更新编码器和解码器参数
    encoder_decoder_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.object_encoder.parameters())
    optimizer = optim.Adam(encoder_decoder_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.0007,
        max_lr=0.002,
        step_size_up=10,
        cycle_momentum=False
    )

    # 初始化梯度缩放器(仅当使用混合精度时)
    scaler = GradScaler() if use_amp else None

    # 冻结规则相关组件
    for param in model.rule_encoder.parameters():
        param.requires_grad = False
    for param in model.rule_applier.parameters():
        param.requires_grad = False
    for param in model.rule_quantizer.parameters():
        param.requires_grad = False

    # 加载断点（如果有）
    start_epoch = 0
    best_loss = float('inf')
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        # 加载梯度缩放器状态(如果有)
        if use_amp and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logging.info(f"从断点恢复: {resume_path}")
        logging.info(f"继续训练从轮次 {start_epoch}")

    # 统计将要重建的网格总数
    total_grids = 0
    for batch_tasks in dataloader:
        for task in batch_tasks:
            # 每个任务的训练样例
            total_grids += len(task['train']) * 2  # 每个训练样例有输入和输出
            # 任务的测试输入
            total_grids += 1  # 测试输入

    logging.info(f"数据集中共有 {total_grids} 个网格将进行重建训练")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0
        recon_loss_sum = 0
        batch_count = 0
        grid_count = 0
        pixel_correct = 0
        pixel_total = 0
        epoch_start_time = time.time()

        for batch_tasks in dataloader:
            for task in batch_tasks:
                # 重建训练样例的输入和输出
                train_examples = task['train']
                for input_grid, output_grid in train_examples:
                    # 重建输入网格
                    input_grid = input_grid.to(device).unsqueeze(0)

                    # 使用混合精度进行前向传播
                    with autocast(enabled=use_amp):
                        recon_input = model.reconstruct_grid(input_grid)
                        # 计算重建损失
                        input_loss = F.cross_entropy(
                            recon_input.reshape(-1, model.num_categories),
                            torch.argmax(input_grid, dim=1).reshape(-1)
                        )

                    # 使用梯度缩放器进行反向传播和优化
                    optimizer.zero_grad()
                    if use_amp:
                        scaler.scale(input_loss).backward()
                        scaler.unscale_(optimizer)  # 在梯度裁剪前反缩放
                        torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        input_loss.backward()
                        torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                        optimizer.step()

                    # 统计
                    recon_loss_sum += input_loss.item()
                    grid_count += 1

                    # 计算像素准确率
                    with torch.no_grad():  # 确保不计算梯度
                        pred_classes = torch.argmax(recon_input, dim=2).reshape(-1)
                        true_classes = torch.argmax(input_grid, dim=1).reshape(-1)
                        pixel_correct += (pred_classes == true_classes).sum().item()
                        pixel_total += true_classes.numel()

                    # 重建输出网格
                    output_grid = output_grid.to(device).unsqueeze(0)

                    # 使用混合精度进行前向传播
                    with autocast(enabled=use_amp):
                        recon_output = model.reconstruct_grid(output_grid)
                        # 计算重建损失
                        output_loss = F.cross_entropy(
                            recon_output.reshape(-1, model.num_categories),
                            torch.argmax(output_grid, dim=1).reshape(-1)
                        )

                    # 使用梯度缩放器进行反向传播和优化
                    optimizer.zero_grad()
                    if use_amp:
                        scaler.scale(output_loss).backward()
                        scaler.unscale_(optimizer)  # 在梯度裁剪前反缩放
                        torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output_loss.backward()
                        torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                        optimizer.step()

                    # 统计
                    recon_loss_sum += output_loss.item()
                    grid_count += 1

                    # 计算像素准确率
                    with torch.no_grad():
                        pred_classes = torch.argmax(recon_output, dim=2).reshape(-1)
                        true_classes = torch.argmax(output_grid, dim=1).reshape(-1)
                        pixel_correct += (pred_classes == true_classes).sum().item()
                        pixel_total += true_classes.numel()

                # 重建测试输入
                test_input = task['test']['input'].to(device).unsqueeze(0)

                # 使用混合精度进行前向传播
                with autocast(enabled=use_amp):
                    recon_test_input = model.reconstruct_grid(test_input)
                    # 计算重建损失
                    test_input_loss = F.cross_entropy(
                        recon_test_input.reshape(-1, model.num_categories),
                        torch.argmax(test_input, dim=1).reshape(-1)
                    )

                # 使用梯度缩放器进行反向传播和优化
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(test_input_loss).backward()
                    scaler.unscale_(optimizer)  # 在梯度裁剪前反缩放
                    torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    test_input_loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder_decoder_params, max_norm=1.0)
                    optimizer.step()

                # 统计
                recon_loss_sum += test_input_loss.item()
                grid_count += 1

                # 计算像素准确率
                with torch.no_grad():
                    pred_classes = torch.argmax(recon_test_input, dim=2).reshape(-1)
                    true_classes = torch.argmax(test_input, dim=1).reshape(-1)
                    pixel_correct += (pred_classes == true_classes).sum().item()
                    pixel_total += true_classes.numel()

                batch_count += 1

        # 计算平均损失和准确率
        avg_recon_loss = recon_loss_sum / grid_count if grid_count > 0 else 0
        pixel_accuracy = pixel_correct / pixel_total if pixel_total > 0 else 0

        # 更新学习率
        scheduler.step(avg_recon_loss)

        # 计算轮次耗时
        epoch_time = time.time() - epoch_start_time

        # 记录日志
        log_message = (f"轮次 {epoch+1}/{epochs}, 耗时: {epoch_time:.2f}s, "
                      f"平均重建损失: {avg_recon_loss:.4f}, 像素准确率: {pixel_accuracy:.4f}, "
                      f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info(log_message)

        # 保存检查点
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': min(best_loss, avg_recon_loss)
        }

        # 如果使用混合精度，也保存scaler状态
        if use_amp:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

        checkpoint_path = os.path.join(pretrain_dir, f"autoencoder_epoch_{epoch+1}.pt")
        torch.save(checkpoint_dict, checkpoint_path)

        # 保存最佳模型
        if avg_recon_loss < best_loss:
            best_loss = avg_recon_loss
            best_model_path = os.path.join(pretrain_dir, "best_autoencoder.pt")
            torch.save(checkpoint_dict, best_model_path)
            logging.info(f"新的最佳模型已保存! 重建损失: {best_loss:.6f}")

    logging.info(f"自编码器预训练完成，最佳重建损失: {best_loss:.6f}")
    return os.path.join(pretrain_dir, "best_autoencoder.pt")