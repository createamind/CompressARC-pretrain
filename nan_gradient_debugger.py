import torch
import numpy as np
import os
import json
from collections import defaultdict
import time

class NaNGradientDebugger:
    """NaN梯度调试工具"""
    
    def __init__(self, model, log_dir="./nan_debug_logs"):
        self.model = model
        self.log_dir = log_dir
        self.nan_stats = defaultdict(list)
        self.registered_hooks = []
        self.task_nan_counts = defaultdict(int)
        self.last_inputs = None
        self.debug_mode = False
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
    def enable_debug_mode(self):
        """启用详细调试模式"""
        self.debug_mode = True
        print("启用NaN梯度详细调试模式")
        
    def disable_debug_mode(self):
        """禁用详细调试模式"""
        self.debug_mode = False
        
    def register_hooks(self):
        """为所有参数注册梯度钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, param_name=name: self._gradient_hook(grad, param_name))
                self.registered_hooks.append(hook)
                
        print(f"已为{len(self.registered_hooks)}个参数注册NaN梯度监控钩子")
        
    def _gradient_hook(self, grad, param_name):
        """梯度钩子函数，检测NaN和异常值"""
        if grad is None:
            return
            
        # 检查NaN和Inf
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            self.nan_stats[param_name].append({
                'time': time.time(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item(),
                'max_abs': float(torch.max(torch.abs(grad)).item()) if not torch.isnan(grad).all() else None
            })
            
            if self.debug_mode:
                # 保存产生NaN的参数信息
                param_data = self.model.get_parameter(param_name).detach().cpu().numpy()
                grad_data = grad.detach().cpu().numpy()
                
                # 保存为NPZ文件以便后续分析
                timestamp = int(time.time())
                np.savez(
                    f"{self.log_dir}/nan_debug_{param_name.replace('.', '_')}_{timestamp}.npz",
                    param_data=param_data,
                    grad_data=grad_data
                )
                
                print(f"⚠️ 参数 {param_name} 检测到NaN/Inf梯度，已保存调试信息")
                
        # 检查梯度爆炸(但不是Inf)
        elif torch.max(torch.abs(grad)) > 1000:  # 阈值可调整
            self.nan_stats[param_name].append({
                'time': time.time(),
                'has_nan': False,
                'has_inf': False,
                'max_abs': float(torch.max(torch.abs(grad)).item())
            })
            
            if self.debug_mode:
                print(f"⚠️ 参数 {param_name} 梯度值异常大: {torch.max(torch.abs(grad)).item():.4f}")
        
        return grad  # 返回原始梯度，不修改
    
    def save_input_on_nan(self, task_id, inputs):
        """当检测到NaN时保存输入数据以便复现问题"""
        self.last_inputs = (task_id, inputs.detach().cpu() if isinstance(inputs, torch.Tensor) else inputs)
    
    def log_nan_task(self, task_id):
        """记录产生NaN的任务ID"""
        self.task_nan_counts[task_id] += 1
        
        if self.last_inputs and self.last_inputs[0] == task_id and self.debug_mode:
            # 保存最后处理的输入数据
            timestamp = int(time.time())
            torch.save(self.last_inputs[1], f"{self.log_dir}/nan_input_{task_id}_{timestamp}.pt")
            print(f"已保存导致NaN的输入数据: {task_id}")
    
    def analyze_and_report(self):
        """分析并报告NaN梯度模式"""
        if not self.nan_stats:
            print("没有检测到NaN梯度")
            return
            
        print("\n===== NaN梯度分析报告 =====")
        
        # 按NaN频率排序参数
        param_nan_counts = {param: len(stats) for param, stats in self.nan_stats.items()}
        sorted_params = sorted(param_nan_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"检测到的NaN梯度总次数: {sum(param_nan_counts.values())}")
        print(f"受影响的参数数量: {len(param_nan_counts)}")
        
        # 报告前10个最常见的NaN梯度参数
        print("\n最常出现NaN的参数:")
        for param, count in sorted_params[:10]:
            print(f"  {param}: {count}次")
            
        # 分析产生NaN的任务
        if self.task_nan_counts:
            print("\n最常产生NaN的任务:")
            sorted_tasks = sorted(self.task_nan_counts.items(), key=lambda x: x[1], reverse=True)
            for task, count in sorted_tasks[:10]:
                print(f"  {task}: {count}次")
        
        # 保存详细报告
        report = {
            "nan_params": param_nan_counts,
            "nan_tasks": self.task_nan_counts,
            "detailed_stats": {k: v for k, v in self.nan_stats.items()}
        }
        
        with open(f"{self.log_dir}/nan_report_{int(time.time())}.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"\n详细报告已保存至: {self.log_dir}/nan_report_{int(time.time())}.json")
        
        # 提供解决建议
        self._provide_solutions(sorted_params)
    
    def _provide_solutions(self, sorted_params):
        """基于分析提供解决方案"""
        print("\n===== 解决建议 =====")
        
        # 检查是否存在模式
        has_pattern = False
        
        # 检查是否主要集中在特定类型的层
        layer_types = defaultdict(int)
        for param, _ in sorted_params:
            if 'conv' in param.lower():
                layer_types['conv'] += 1
            elif 'norm' in param.lower():
                layer_types['norm'] += 1
            elif 'linear' in param.lower() or 'fc' in param.lower():
                layer_types['linear'] += 1
                
        dominant_type = max(layer_types.items(), key=lambda x: x[1])[0] if layer_types else None
        
        if dominant_type and layer_types[dominant_type] > len(sorted_params) * 0.5:
            has_pattern = True
            print(f"1. NaN梯度主要出现在{dominant_type}层，建议:")
            if dominant_type == 'conv':
                print("   - 降低卷积层学习率")
                print("   - 检查卷积层的初始化方法")
                print("   - 考虑在卷积层前添加归一化")
            elif dominant_type == 'norm':
                print("   - 检查归一化层的epsilon值(增大到1e-5或更高)")
                print("   - 考虑使用更稳定的归一化方法(GroupNorm替代BatchNorm)")
                print("   - 检查输入是否包含极端值")
            elif dominant_type == 'linear':
                print("   - 降低全连接层学习率")
                print("   - 添加权重正则化")
                print("   - 检查输入特征是否有极端值")
        
        # 提供通用建议
        print("\n2. 通用解决方案:")
        print("   - 增强梯度裁剪: optimizer.clip_grad_norm_(model.parameters(), max_norm=1.0)")
        print("   - 降低学习率: 当前学习率减半")
        print("   - 使用梯度累积减少批次波动")
        print("   - 检查损失函数中的数值稳定性")
        
        # 代码示例
        print("\n3. 建议的代码修改:")
        print("""
    # 增强梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 平滑损失计算
    def stable_loss(pred, target, epsilon=1e-7):
        pred = torch.clamp(pred, min=epsilon, max=1-epsilon)
        return F.binary_cross_entropy(pred, target)
    
    # 渐进式梯度累积
    accumulation_steps = 4
    optimizer.zero_grad()
    for i in range(accumulation_steps):
        outputs = model(inputs[i])
        loss = loss_fn(outputs, targets[i]) / accumulation_steps
        loss.backward()
    optimizer.step()
        """)
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.registered_hooks:
            hook.remove()
        self.registered_hooks = []
        print("已移除所有NaN梯度监控钩子")
        
    def __del__(self):
        """析构函数，确保钩子被移除"""
        self.remove_hooks()