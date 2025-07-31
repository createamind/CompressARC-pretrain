import torch
import torch.nn as nn
import torch.nn.functional as F



def fix_training_stall(model, optimizer):
    """修复训练停滞问题"""
    print("\n==== 应用训练停滞修复 ====")

    # 1. 检查并修复梯度裁剪过度问题
    global_clip_value = 5.0  # 增大梯度裁剪阈值
    print(f"将梯度裁剪阈值增大到 {global_clip_value}")

    # 2. 重置优化器统计量
    optimizer.state.clear()
    print("已重置优化器状态")

    # 3. 提高学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 3.0  # 提高学习率
        print(f"学习率调整为: {param_group['lr']:.6f}")

    # 4. 恢复LeakyReLU激活函数以增加梯度信号
    activation_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and 'decoder' in name:
            parent = model
            path = name.split('.')
            for part in path[:-1]:
                parent = getattr(parent, part)
            setattr(parent, path[-1], nn.LeakyReLU(0.2))
            activation_count += 1

    print(f"已将{activation_count}个ReLU激活函数恢复为LeakyReLU")

    return global_clip_value

def modified_layer_specific_gradient_clipping(model, optimizer, global_clip_value=5.0):
    """修改后的分层梯度裁剪，使用更大的裁剪值"""
    # 定义层特定的裁剪值
    clip_settings = {
        'decoder.final_decoder': global_clip_value * 0.5,   # 降低限制，但仍相对严格
        'decoder.low_decoder': global_clip_value * 0.7,     # 降低限制
        'decoder': global_clip_value * 0.9,                 # 降低限制
        'default': global_clip_value                        # 使用较大的标准裁剪值
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

def verify_parameter_updates(model, optimizer):
    """验证参数是否正在更新"""
    # 保存一些关键参数的副本
    param_copies = {}
    for name, param in model.named_parameters():
        if ('decoder.final_decoder' in name or 'decoder.low_decoder' in name) and param.requires_grad:
            param_copies[name] = param.detach().clone()

    # 返回验证函数
    def verify_after_step():
        changes = []
        for name, old_param in param_copies.items():
            new_param = model.get_parameter(name)
            diff = torch.abs(new_param - old_param).sum().item()
            if diff > 0:
                changes.append((name, diff))

        # if changes:
        #     print(f"确认参数已更新: {len(changes)}个参数发生变化")
        #     for name, diff in changes[:3]:  # 只显示前3个
        #         print(f"  - {name}: 变化量 = {diff:.6f}")
        # else:
        #     print("警告: 没有参数发生变化！可能存在重大问题")

    return verify_after_step

def unstable_cross_entropy(pred, target):
    """恢复常规交叉熵损失，允许更大梯度"""
    return F.cross_entropy(pred, target)