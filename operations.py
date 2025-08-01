import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


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


# === 新增操作类 ===

class CropOperation(BaseOperation):
    """裁剪操作：从网格中裁剪出子区域"""
    def __init__(self):
        super().__init__("crop")

    def forward(self, grid, params):
        """裁剪网格

        参数:
            grid: [B, C, H, W] 网格
            params: {'top': int, 'left': int, 'height': int, 'width': int} 裁剪区域
        """
        top = params['top']
        left = params['left']
        height = params['height']
        width = params['width']

        # 确保裁剪范围有效
        B, C, H, W = grid.shape
        top = max(0, min(top, H-1))
        left = max(0, min(left, W-1))
        height = max(1, min(height, H-top))
        width = max(1, min(width, W-left))

        # 裁剪区域
        result = grid[:, :, top:top+height, left:left+width]

        return result

    def get_parameter_space(self):
        return {
            'top': {'type': 'int', 'range': [0, 29]},
            'left': {'type': 'int', 'range': [0, 29]},
            'height': {'type': 'int', 'range': [1, 30]},
            'width': {'type': 'int', 'range': [1, 30]}
        }


class ExtendOperation(BaseOperation):
    """扩展操作：向指定方向扩展网格"""
    def __init__(self):
        super().__init__("extend")

    def forward(self, grid, params):
        """扩展网格

        参数:
            grid: [B, C, H, W] 网格
            params: {'direction': 'top'|'right'|'bottom'|'left', 'size': int, 'color': int} 扩展方向、大小和颜色
        """
        direction = params['direction']
        size = params['size']
        color = params['color']

        B, C, H, W = grid.shape
        result = grid.clone()

        # 创建扩展区域
        if direction == 'top':
            extension = torch.zeros((B, C, size, W), device=grid.device)
            if 0 <= color < C:
                extension[:, color] = 1
            result = torch.cat([extension, result], dim=2)
        elif direction == 'bottom':
            extension = torch.zeros((B, C, size, W), device=grid.device)
            if 0 <= color < C:
                extension[:, color] = 1
            result = torch.cat([result, extension], dim=2)
        elif direction == 'left':
            extension = torch.zeros((B, C, H, size), device=grid.device)
            if 0 <= color < C:
                extension[:, color] = 1
            result = torch.cat([extension, result], dim=3)
        elif direction == 'right':
            extension = torch.zeros((B, C, H, size), device=grid.device)
            if 0 <= color < C:
                extension[:, color] = 1
            result = torch.cat([result, extension], dim=3)

        return result

    def get_parameter_space(self):
        return {
            'direction': {'type': 'categorical', 'values': ['top', 'right', 'bottom', 'left']},
            'size': {'type': 'int', 'range': [1, 10]},
            'color': {'type': 'int', 'range': [0, 9]}
        }


class BorderFillOperation(BaseOperation):
    """边框填充操作：在对象或网格周围添加边框"""
    def __init__(self):
        super().__init__("border_fill")

    def forward(self, grid, params):
        """添加边框

        参数:
            grid: [B, C, H, W] 网格
            params: {'color': int, 'thickness': int} 边框颜色和厚度
        """
        color = params['color']
        thickness = params['thickness']

        B, C, H, W = grid.shape
        result = grid.clone()

        # 确保颜色和厚度有效
        if not (0 <= color < C) or thickness <= 0:
            return result

        # 添加边框
        for t in range(thickness):
            # 上边框
            result[:, :, t, :] = 0
            result[:, color, t, :] = 1

            # 下边框
            result[:, :, H-1-t, :] = 0
            result[:, color, H-1-t, :] = 1

            # 左边框
            result[:, :, :, t] = 0
            result[:, color, :, t] = 1

            # 右边框
            result[:, :, :, W-1-t] = 0
            result[:, color, :, W-1-t] = 1

        return result

    def get_parameter_space(self):
        return {
            'color': {'type': 'int', 'range': [0, 9]},
            'thickness': {'type': 'int', 'range': [1, 5]}
        }


class CheckerPatternOperation(BaseOperation):
    """棋盘模式操作：创建棋盘格"""
    def __init__(self):
        super().__init__("checker_pattern")

    def forward(self, grid, params):
        """创建棋盘格

        参数:
            grid: [B, C, H, W] 网格
            params: {'color1': int, 'color2': int, 'size': int} 两种颜色和格子大小
        """
        color1 = params['color1']
        color2 = params['color2']
        size = params['size']

        B, C, H, W = grid.shape
        result = torch.zeros_like(grid)

        # 确保颜色有效
        if not (0 <= color1 < C and 0 <= color2 < C):
            return result

        # 创建棋盘格
        for i in range(0, H, size):
            for j in range(0, W, size):
                # 判断当前块的颜色
                color = color1 if ((i // size) + (j // size)) % 2 == 0 else color2

                # 填充当前块
                h_end = min(i + size, H)
                w_end = min(j + size, W)
                result[:, color, i:h_end, j:w_end] = 1

        return result

    def get_parameter_space(self):
        return {
            'color1': {'type': 'int', 'range': [0, 9]},
            'color2': {'type': 'int', 'range': [0, 9]},
            'size': {'type': 'int', 'range': [1, 10]}
        }


class MirrorOperation(BaseOperation):
    """镜像操作：沿对角线或反对角线镜像"""
    def __init__(self):
        super().__init__("mirror")

    def forward(self, grid, params):
        """镜像

        参数:
            grid: [B, C, H, W] 网格
            params: {'axis': 'diagonal'|'antidiagonal'} 镜像轴
        """
        axis = params['axis']

        if axis == 'diagonal':
            # 对角线镜像 (类似于转置)
            result = grid.transpose(-1, -2)
        elif axis == 'antidiagonal':
            # 反对角线镜像
            result = grid.flip(-1).flip(-2).transpose(-1, -2)
        else:
            result = grid

        return result

    def get_parameter_space(self):
        return {
            'axis': {'type': 'categorical', 'values': ['diagonal', 'antidiagonal']}
        }


class LogicalOperation(BaseOperation):
    """逻辑操作：对两个网格进行逻辑运算"""
    def __init__(self):
        super().__init__("logical")

    def forward(self, grid, params):
        """逻辑运算

        参数:
            grid: [B, C, H, W] 主网格
            params: {'operation': 'and'|'or'|'xor', 'grid2': [B, C, H, W]} 操作类型和第二个网格
        """
        operation = params['operation']
        grid2 = params['grid2']

        # 确保两个网格大小一致
        B, C, H, W = grid.shape
        if grid2.shape != grid.shape:
            # 尝试调整大小或填充
            grid2 = F.interpolate(grid2, size=(H, W), mode='nearest')

        # 应用逻辑操作
        if operation == 'and':
            result = grid * grid2  # 元素级乘法实现逻辑与
        elif operation == 'or':
            result = torch.clamp(grid + grid2, 0, 1)  # 元素级加法然后裁剪到0-1实现逻辑或
        elif operation == 'xor':
            result = ((grid + grid2) % 2).to(grid.dtype)  # 元素级加法取模实现异或
        else:
            result = grid

        return result

    def get_parameter_space(self):
        return {
            'operation': {'type': 'categorical', 'values': ['and', 'or', 'xor']},
            'grid2': {'type': 'tensor', 'shape': [None, None, None, None]}
        }


class PatternOperation(BaseOperation):
    """模式操作：检测或应用模式"""
    def __init__(self):
        super().__init__("pattern")

    def forward(self, grid, params):
        """模式操作

        参数:
            grid: [B, C, H, W] 网格
            params: {
                'mode': 'detect'|'apply',
                'pattern': [1, C, PH, PW],
                'target_color': int,  # 仅用于detect模式
                'positions': [(y, x), ...],  # 仅用于apply模式
            }
        """
        mode = params['mode']
        pattern = params['pattern']

        B, C, H, W = grid.shape
        PH, PW = pattern.shape[2], pattern.shape[3]

        if mode == 'detect':
            # 检测模式：寻找匹配模式的位置
            target_color = params.get('target_color', 0)
            result = torch.zeros_like(grid)

            # 对每个可能的位置应用卷积检查是否匹配
            for b in range(B):
                for i in range(H - PH + 1):
                    for j in range(W - PW + 1):
                        # 检查当前位置是否匹配模式
                        window = grid[b, :, i:i+PH, j:j+PW]
                        match = torch.all(torch.abs(window - pattern[0]) < 0.1)

                        if match:
                            # 在匹配位置标记颜色
                            result[b, :, i:i+PH, j:j+PW] = 0
                            result[b, target_color, i:i+PH, j:j+PW] = 1

        elif mode == 'apply':
            # 应用模式：在指定位置应用模式
            positions = params.get('positions', [])
            result = grid.clone()

            for pos in positions:
                y, x = pos
                # 确保位置有效
                if 0 <= y < H - PH + 1 and 0 <= x < W - PW + 1:
                    # 清除目标区域原有内容
                    result[:, :, y:y+PH, x:x+PW] = 0
                    # 应用模式
                    result[:, :, y:y+PH, x:x+PW] = pattern

        else:
            result = grid

        return result

    def get_parameter_space(self):
        return {
            'mode': {'type': 'categorical', 'values': ['detect', 'apply']},
            'pattern': {'type': 'tensor', 'shape': [1, None, None, None]},
            'target_color': {'type': 'int', 'range': [0, 9]},
            'positions': {'type': 'list', 'of': 'tuple'}
        }


class ScaleOperation(BaseOperation):
    """缩放操作：上采样或下采样"""
    def __init__(self):
        super().__init__("scale")

    def forward(self, grid, params):
        """缩放网格

        参数:
            grid: [B, C, H, W] 网格
            params: {
                'mode': 'up'|'down',
                'factor': int,
                'method': 'nearest'|'linear'|'area'  # 插值方法
            }
        """
        mode = params['mode']
        factor = params['factor']
        method = params.get('method', 'nearest')

        B, C, H, W = grid.shape

        if mode == 'up':
            # 上采样
            new_size = (H * factor, W * factor)
            # 对于上采样，nearest通常效果最好，因为我们希望保持离散颜色值
            result = F.interpolate(grid, size=new_size, mode=method, align_corners=False if method != 'nearest' else None)

        elif mode == 'down':
            # 下采样
            new_size = (H // factor, W // factor)
            # 确保新尺寸至少为1
            new_size = (max(1, new_size[0]), max(1, new_size[1]))
            result = F.interpolate(grid, size=new_size, mode=method, align_corners=False if method != 'nearest' else None)

        else:
            result = grid

        return result

    def get_parameter_space(self):
        return {
            'mode': {'type': 'categorical', 'values': ['up', 'down']},
            'factor': {'type': 'int', 'range': [2, 5]},
            'method': {'type': 'categorical', 'values': ['nearest', 'linear', 'area']}
        }


class ConcatOperation(BaseOperation):
    """拼接操作：水平或垂直拼接两个网格"""
    def __init__(self):
        super().__init__("concat")

    def forward(self, grid, params):
        """拼接网格

        参数:
            grid: [B, C, H, W] 主网格
            params: {'direction': 'horizontal'|'vertical', 'grid2': [B, C, H, W]} 拼接方向和第二个网格
        """
        direction = params['direction']
        grid2 = params['grid2']

        # 确保两个网格在相关维度兼容
        B, C, H, W = grid.shape
        B2, C2, H2, W2 = grid2.shape

        if B != B2 or C != C2:
            # 批次大小或通道数不匹配
            return grid

        if direction == 'horizontal':
            # 水平拼接（沿宽度维度），需要确保高度一致
            if H != H2:
                # 调整第二个网格的高度
                grid2 = F.interpolate(grid2, size=(H, W2), mode='nearest')
            result = torch.cat([grid, grid2], dim=3)  # 沿宽度维度拼接

        elif direction == 'vertical':
            # 垂直拼接（沿高度维度），需要确保宽度一致
            if W != W2:
                # 调整第二个网格的宽度
                grid2 = F.interpolate(grid2, size=(H2, W), mode='nearest')
            result = torch.cat([grid, grid2], dim=2)  # 沿高度维度拼接

        else:
            result = grid

        return result

    def get_parameter_space(self):
        return {
            'direction': {'type': 'categorical', 'values': ['horizontal', 'vertical']},
            'grid2': {'type': 'tensor', 'shape': [None, None, None, None]}
        }


class CountColorOperation(BaseOperation):
    """颜色计数操作：计算网格中特定颜色的数量"""
    def __init__(self):
        super().__init__("count_color")

    def forward(self, grid, params):
        """计数特定颜色

        参数:
            grid: [B, C, H, W] 网格
            params: {'color': int, 'target_color': int, 'target_count': int}
            统计color出现的次数，如果次数等于target_count，则将颜色替换为target_color
        """
        color = params['color']
        target_color = params['target_color']
        target_count = params['target_count']

        B, C, H, W = grid.shape
        result = grid.clone()

        # 对每个批次单独处理
        for b in range(B):
            # 计算指定颜色的数量
            color_count = torch.sum(grid[b, color]).item()

            # 如果数量匹配目标计数，则替换为目标颜色
            if int(color_count) == target_count:
                mask = grid[b, color] > 0.5

                # 清除原颜色
                result[b, color][mask] = 0

                # 设置新颜色
                if 0 <= target_color < C:
                    result[b, target_color][mask] = 1

        return result

    def get_parameter_space(self):
        return {
            'color': {'type': 'int', 'range': [0, 9]},
            'target_color': {'type': 'int', 'range': [0, 9]},
            'target_count': {'type': 'int', 'range': [1, 100]}
        }


class ReplaceBackgroundOperation(BaseOperation):
    """背景替换操作：将背景（最常见的颜色）替换为指定颜色"""
    def __init__(self):
        super().__init__("replace_background")

    def forward(self, grid, params):
        """替换背景

        参数:
            grid: [B, C, H, W] 网格
            params: {'target_color': int} 目标颜色
        """
        target_color = params['target_color']

        B, C, H, W = grid.shape
        result = grid.clone()

        for b in range(B):
            # 计算每种颜色的出现频率
            color_counts = []
            for c in range(C):
                color_counts.append((c, torch.sum(grid[b, c]).item()))

            # 找出最常见的颜色（背景色）
            bg_color = max(color_counts, key=lambda x: x[1])[0]

            # 创建背景掩码
            bg_mask = grid[b, bg_color] > 0.5

            # 替换背景
            result[b, bg_color][bg_mask] = 0
            result[b, target_color][bg_mask] = 1

        return result

    def get_parameter_space(self):
        return {
            'target_color': {'type': 'int', 'range': [0, 9]}
        }


class FloodFillOperation(BaseOperation):
    """泛洪填充操作：从种子点开始填充相同颜色区域"""
    def __init__(self):
        super().__init__("flood_fill")

    def forward(self, grid, params):
        """泛洪填充

        参数:
            grid: [B, C, H, W] 网格
            params: {'seed': (y, x), 'target_color': int} 种子点和目标颜色
        """
        seed = params['seed']
        target_color = params['target_color']

        B, C, H, W = grid.shape
        result = grid.clone()

        # 对每个批次单独处理
        for b in range(B):
            y, x = seed

            # 确保种子点有效
            if not (0 <= y < H and 0 <= x < W):
                continue

            # 找出种子点的颜色
            seed_color = -1
            for c in range(C):
                if grid[b, c, y, x] > 0.5:
                    seed_color = c
                    break

            if seed_color == -1:
                continue

            # 创建待处理队列和已访问集合
            queue = [(y, x)]
            visited = set([(y, x)])

            # 填充相同颜色的连通区域
            while queue:
                cy, cx = queue.pop(0)

                # 四个相邻方向
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx

                    # 检查是否在范围内且为相同颜色
                    if (0 <= ny < H and 0 <= nx < W and
                        (ny, nx) not in visited and
                        grid[b, seed_color, ny, nx] > 0.5):
                        queue.append((ny, nx))
                        visited.add((ny, nx))

            # 创建填充掩码
            fill_mask = torch.zeros((H, W), dtype=torch.bool, device=grid.device)
            for fy, fx in visited:
                fill_mask[fy, fx] = True

            # 更改颜色
            result[b, seed_color][fill_mask] = 0
            result[b, target_color][fill_mask] = 1

        return result

    def get_parameter_space(self):
        return {
            'seed': {'type': 'tuple', 'elements': 2, 'element_type': 'int'},
            'target_color': {'type': 'int', 'range': [0, 9]}
        }


class OperationLibrary(nn.Module):
    """操作库：定义并编码基本ARC操作"""
    def __init__(self):
        super().__init__()

        # 注册所有支持的操作
        self.operations = nn.ModuleDict({
            # 基本操作
            'move': MoveOperation(),
            'rotate': RotateOperation(),
            'flip': FlipOperation(),
            'color': ColorOperation(),
            'fill': FillOperation(),
            'copy': CopyOperation(),

            # 新增操作
            'crop': CropOperation(),
            'extend': ExtendOperation(),
            'border_fill': BorderFillOperation(),
            'checker_pattern': CheckerPatternOperation(),
            'mirror': MirrorOperation(),
            'logical': LogicalOperation(),
            'pattern': PatternOperation(),
            'scale': ScaleOperation(),
            'concat': ConcatOperation(),
            'count_color': CountColorOperation(),
            'replace_background': ReplaceBackgroundOperation(),
            'flood_fill': FloodFillOperation(),
        })

        # 操作嵌入层
        self.operation_embedding = nn.Embedding(len(self.operations), 128)

        # 操作序列编码器
        self.sequence_encoder = nn.GRU(128, 128, batch_first=True)

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
        # 计算每个操作的概率
        operation_logits = torch.matmul(rule_embedding, self.operation_embedding.weight.t())
        operation_probs = F.softmax(operation_logits, dim=-1)

        # 取概率最高的K个操作
        K = min(5, operation_probs.size(-1))  # 最多5个操作，或者全部操作数
        topk_probs, topk_indices = torch.topk(operation_probs, k=K, dim=-1)

        operations = []
        for i in range(K):
            if topk_probs[0, i] > 0.1:  # 概率阈值
                op_idx = topk_indices[0, i].item()
                if op_idx < len(list(self.operations.keys())):
                    op_type = list(self.operations.keys())[op_idx]

                    # 预测参数（使用共享参数预测器）
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

    def apply_operation_sequence(self, grid, operation_sequence):
        """按顺序应用多个操作"""
        result = grid
        for op in operation_sequence:
            op_type = op['type']
            params = op['params']
            result = self.apply_operation(result, op_type, params)
        return result