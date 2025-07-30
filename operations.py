import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any


class BaseOperation(nn.Module):
    """基础操作类，所有ARC操作的父类"""
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def forward(self, grid, params):
        """应用操作到网格"""
        raise NotImplementedError("每个操作子类必须实现forward方法")
    
    def get_parameter_space(self):
        """返回此操作的参数空间描述"""
        raise NotImplementedError("每个操作子类必须定义其参数空间")


class MoveOperation(BaseOperation):
    """移动操作：移动对象或整个网格"""
    def __init__(self):
        super().__init__("move")
    
    def forward(self, grid, params):
        """移动网格中的内容
        
        参数:
            grid: [B, C, H, W] 网格
            params: {'dx': int, 'dy': int} 移动距离
        """
        dx, dy = params['dx'], params['dy']
        B, C, H, W = grid.shape
        result = torch.zeros_like(grid)
        
        # 根据移动方向处理
        if dx >= 0 and dy >= 0:
            result[:, :, dy:, dx:] = grid[:, :, :H-dy, :W-dx]
        elif dx >= 0 and dy < 0:
            result[:, :, :H+dy, dx:] = grid[:, :, -dy:, :W-dx]
        elif dx < 0 and dy >= 0:
            result[:, :, dy:, :W+dx] = grid[:, :, :H-dy, -dx:]
        else:  # dx < 0 and dy < 0
            result[:, :, :H+dy, :W+dx] = grid[:, :, -dy:, -dx:]
        
        return result
    
    def get_parameter_space(self):
        return {
            'dx': {'type': 'int', 'range': [-10, 10]},
            'dy': {'type': 'int', 'range': [-10, 10]}
        }


class RotateOperation(BaseOperation):
    """旋转操作：旋转网格或对象"""
    def __init__(self):
        super().__init__("rotate")
    
    def forward(self, grid, params):
        """旋转网格
        
        参数:
            grid: [B, C, H, W] 网格
            params: {'angle': 90|180|270} 旋转角度，顺时针
        """
        angle = params['angle']
        if angle == 90:
            result = grid.transpose(-1, -2).flip(-2)
        elif angle == 180:
            result = grid.flip(-1).flip(-2)
        elif angle == 270:
            result = grid.transpose(-1, -2).flip(-1)
        else:
            # 无效角度返回原网格
            result = grid
        
        return result
    
    def get_parameter_space(self):
        return {
            'angle': {'type': 'categorical', 'values': [90, 180, 270]}
        }


class FlipOperation(BaseOperation):
    """翻转操作：水平或垂直翻转网格"""
    def __init__(self):
        super().__init__("flip")
    
    def forward(self, grid, params):
        """翻转网格
        
        参数:
            grid: [B, C, H, W] 网格
            params: {'axis': 'horizontal'|'vertical'} 翻转轴
        """
        axis = params['axis']
        if axis == 'horizontal':
            result = grid.flip(-1)  # 水平翻转
        elif axis == 'vertical':
            result = grid.flip(-2)  # 垂直翻转
        else:
            # 无效轴返回原网格
            result = grid
        
        return result
    
    def get_parameter_space(self):
        return {
            'axis': {'type': 'categorical', 'values': ['horizontal', 'vertical']}
        }


class ColorOperation(BaseOperation):
    """颜色操作：更改对象颜色"""
    def __init__(self):
        super().__init__("color")
    
    def forward(self, grid, params):
        """更改颜色
        
        参数:
            grid: [B, C, H, W] one-hot编码的网格
            params: {'from_color': int, 'to_color': int} 颜色变换
        """
        from_color = params['from_color']
        to_color = params['to_color']
        result = grid.clone()
        
        # 在我们的one-hot编码中，颜色就是通道索引
        # 将from_color通道中的1移到to_color通道
        if 0 <= from_color < grid.shape[1] and 0 <= to_color < grid.shape[1]:
            mask = grid[:, from_color] > 0.5
            result[:, from_color][mask] = 0
            result[:, to_color][mask] = 1
        
        return result
    
    def get_parameter_space(self):
        return {
            'from_color': {'type': 'int', 'range': [0, 9]},
            'to_color': {'type': 'int', 'range': [0, 9]}
        }


class FillOperation(BaseOperation):
    """填充操作：填充区域"""
    def __init__(self):
        super().__init__("fill")
    
    def forward(self, grid, params):
        """填充区域
        
        参数:
            grid: [B, C, H, W] 网格
            params: {'color': int, 'mask': [B, 1, H, W]} 填充颜色和区域
        """
        color = params['color']
        mask = params['mask']
        result = grid.clone()
        
        # 在mask区域应用新颜色
        if 0 <= color < grid.shape[1]:
            # 先清除mask区域的所有颜色
            for c in range(grid.shape[1]):
                result[:, c][mask > 0.5] = 0
            
            # 设置新颜色
            result[:, color][mask > 0.5] = 1
        
        return result
    
    def get_parameter_space(self):
        return {
            'color': {'type': 'int', 'range': [0, 9]},
            'mask': {'type': 'tensor', 'shape': [None, 1, None, None]}
        }


class CopyOperation(BaseOperation):
    """复制操作：复制对象或区域"""
    def __init__(self):
        super().__init__("copy")
    
    def forward(self, grid, params):
        """复制对象
        
        参数:
            grid: [B, C, H, W] 网格
            params: {'src': (y1,x1,y2,x2), 'dst': (y,x)} 源区域和目标位置
        """
        src = params['src']
        dst = params['dst']
        result = grid.clone()
        
        # 提取源区域
        y1, x1, y2, x2 = src
        src_height, src_width = y2 - y1, x2 - x1
        src_content = grid[:, :, y1:y2, x1:x2]
        
        # 粘贴到目标位置
        dst_y, dst_x = dst
        if dst_y >= 0 and dst_x >= 0 and dst_y + src_height <= grid.shape[2] and dst_x + src_width <= grid.shape[3]:
            # 清除目标区域原有内容
            result[:, :, dst_y:dst_y+src_height, dst_x:dst_x+src_width] = 0
            
            # 复制内容
            result[:, :, dst_y:dst_y+src_height, dst_x:dst_x+src_width] = src_content
        
        return result
    
    def get_parameter_space(self):
        return {
            'src': {'type': 'tuple', 'elements': 4, 'element_type': 'int'},
            'dst': {'type': 'tuple', 'elements': 2, 'element_type': 'int'}
        }


class OperationLibrary(nn.Module):
    """操作库：定义并编码基本ARC操作"""
    def __init__(self):
        super().__init__()
        
        # 注册所有支持的操作
        self.operations = nn.ModuleDict({
            'move': MoveOperation(),
            'rotate': RotateOperation(),
            'flip': FlipOperation(),
            'color': ColorOperation(),
            'fill': FillOperation(),
            'copy': CopyOperation(),
        })
        
        # 操作嵌入层
        self.operation_embedding = nn.Embedding(len(self.operations), 64)
        
        # 操作序列编码器
        self.sequence_encoder = nn.GRU(64, 128, batch_first=True)
        
        # 操作参数预测器
        self.parameter_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def encode_operations(self, operation_sequence):
        """将操作序列编码为嵌入向量"""
        op_indices = []
        for op in operation_sequence:
            op_type = op['type']
            if op_type in self.operations:
                op_indices.append(list(self.operations.keys()).index(op_type))
            else:
                # 未知操作，使用占位符
                op_indices.append(0)
        
        op_indices_tensor = torch.tensor(op_indices, dtype=torch.long, device=next(self.parameters()).device)
        op_embeddings = self.operation_embedding(op_indices_tensor)
        _, hidden = self.sequence_encoder(op_embeddings.unsqueeze(0))
        return hidden.squeeze(0)
    
    def decode_operations(self, rule_embedding):
        """从规则嵌入解码操作序列"""
        # 简化版：仅预测操作类型和参数
        operation_logits = torch.matmul(rule_embedding, self.operation_embedding.weight.t())
        operation_probs = F.softmax(operation_logits, dim=-1)
        
        # 取概率最高的K个操作
        K = min(3, operation_probs.size(-1))  # 最多3个操作，或者全部操作数
        topk_probs, topk_indices = torch.topk(operation_probs, k=K, dim=-1)
        
        operations = []
        for i in range(K):
            if topk_probs[0, i] > 0.1:  # 概率阈值
                op_idx = topk_indices[0, i].item()
                if op_idx < len(list(self.operations.keys())):
                    op_type = list(self.operations.keys())[op_idx]
                    
                    # 预测参数（示例实现，实际需要为每种操作定制）
                    params = self.parameter_predictor(rule_embedding)
                    
                    operations.append({
                        'type': op_type,
                        'params': params.detach().cpu().numpy(),
                        'probability': topk_probs[0, i].item()
                    })
        
        return operations
    
    def apply_operation(self, grid, operation_name, params):
        """应用指定操作到网格"""
        if operation_name in self.operations:
            return self.operations[operation_name](grid, params)
        return grid  # 未知操作返回原网格