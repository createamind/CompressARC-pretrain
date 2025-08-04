import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from hierarchical_vae import (HierarchicalEncoder, HierarchicalDecoder,
                             VectorQuantizer, ConnectedComponentsModule,
                             RelationalReasoningModule)
from operations import OperationLibrary


class RuleExtractor(nn.Module):
    """规则提取器：分析输入-输出对差异，提取变换规则"""
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.object_diff_encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.rule_generator = nn.Sequential(
            nn.Linear(hidden_dim + 64, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()  # 确保规则嵌入在合理范围
        )

    def forward(self, input_high, output_high, input_objects, output_objects):
        """提取输入-输出对之间的规则"""
        # 高级特征差异分析
        combined_high = torch.cat([input_high, output_high], dim=-1)
        high_features = self.input_encoder(combined_high)

        # 对象级差异分析（简化版）
        input_obj_mean = torch.cat(input_objects, dim=0).mean(dim=0, keepdim=True) if input_objects else torch.zeros(1, 128, device=input_high.device)
        output_obj_mean = torch.cat(output_objects, dim=0).mean(dim=0, keepdim=True) if output_objects else torch.zeros(1, 128, device=input_high.device)

        obj_diff = output_obj_mean - input_obj_mean
        obj_features = self.object_diff_encoder(obj_diff)

        # 生成规则表示
        rule_embedding = self.rule_generator(torch.cat([high_features, obj_features], dim=-1))
        return rule_embedding


class RuleApplier(nn.Module):
    """规则应用器：将提取的规则应用到新输入"""
    def __init__(self, operation_library):
        super().__init__()
        self.operation_library = operation_library

        # 转换网络
        self.transform_network = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # 像素特征转换器
        self.pixel_transformer = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

    def forward(self, objects, pixel_features, rule_embedding, operations=None):
        """应用规则到对象和像素特征"""
        # 根据操作转换对象
        transformed_objects = []
        for obj in objects:
            # 将规则信息与对象结合
            obj_with_rule = torch.cat([obj, rule_embedding.expand(obj.size(0), -1)], dim=-1)
            transformed_obj = self.transform_network(obj_with_rule)
            transformed_objects.append(transformed_obj)

        # 转换像素特征
        # 扩展规则嵌入到像素特征的空间尺寸
        b, c, h, w = pixel_features.shape
        rule_expanded = rule_embedding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        # 连接像素特征和规则
        pixel_with_rule = torch.cat([pixel_features, rule_expanded], dim=1)
        transformed_pixels = self.pixel_transformer(pixel_with_rule)

        return transformed_objects, transformed_pixels


class ObjectComposer(nn.Module):
    """对象组合器：将处理后的对象组合回场景"""
    def __init__(self):
        super().__init__()
        self.composition_network = nn.Sequential(
            nn.Conv2d(64 + 128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )

    def forward(self, objects, pixel_features, grid_size=None):
        """组合对象和像素特征"""
        # 添加调试信息
        print(f"像素特征尺寸: {pixel_features.shape}")

        # 创建对象特征图 - 使用pixel_features的实际尺寸
        batch_size, _, height, width = pixel_features.shape

        object_map = torch.zeros(batch_size, 128, height, width, device=pixel_features.device)

        # 简化：使用平均对象特征填充地图
        if objects:
            obj_mean = torch.cat(objects, dim=0).mean(dim=0, keepdim=True)
            object_map[:] = obj_mean.view(1, 128, 1, 1).expand(-1, -1, height, width)

            # 添加优雅降级机制，确保尺寸匹配
            if pixel_features.shape[-1] != object_map.shape[-1]:
                print(f"调整对象图尺寸: {object_map.shape} -> ({pixel_features.shape[-2]}, {pixel_features.shape[-1]})")
                object_map = F.interpolate(object_map, size=(pixel_features.shape[-2], pixel_features.shape[-1]))

        # 组合像素和对象特征
        combined = torch.cat([pixel_features, object_map], dim=1)
        composed_pixels = self.composition_network(combined)

        print(f"转换后的像素特征尺寸: {composed_pixels.shape}")

        return {
            'pixel_features': composed_pixels,
            'object_features': objects  # 返回原始对象列表，不进行合并
        }


class RuleEvaluator(nn.Module):
    """规则评估器：评估生成输出与目标输出的一致性"""
    def __init__(self):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=1),  # 10类别 x 2（预测和真实）
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, predicted, ground_truth):
        """评估预测输出与真实输出的匹配度"""
        # 将预测和真实分类合并为一个tensor
        combined = torch.cat([predicted, ground_truth], dim=1)
        score = self.evaluator(combined)
        return score

# from enhanced_inference import ExplicitRuleInference










class RuleGuidedVAE(nn.Module):
    """
    规则引导的VAE模型，用于ARC任务
    使用直接的One-hot编码替代像素VQ
    """

    def __init__(
        self,
        grid_size: int = 30,
        num_categories: int = 10,
        pixel_codebook_size: int = 512,  # 保留原参数以兼容接口
        object_codebook_size: int = 256,
        rule_codebook_size: int = 128,
        pixel_dim: int = 64,  # 保留原参数以兼容接口
        object_dim: int = 128,
        relation_dim: int = 64,
        rule_dim: int = 128,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_categories = num_categories
        self.pixel_codebook_size = pixel_codebook_size  # 保留但不使用
        self.object_codebook_size = object_codebook_size
        self.rule_codebook_size = rule_codebook_size
        self.pixel_dim = pixel_dim  # 保留但不使用
        self.object_dim = object_dim
        self.relation_dim = relation_dim
        self.rule_dim = rule_dim

        # 修改编码器：直接处理one-hot输入
        self.encoder = nn.Sequential(
            nn.Conv2d(num_categories, 32, kernel_size=3, padding=1),  # 直接接受one-hot输入
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )

        # 对象级VQ - 使用与原代码兼容的参数
        self.object_quantizer = VectorQuantizer(
            num_embeddings=object_codebook_size,
            embedding_dim=object_dim,
            commitment_cost=0.25
        )

        # 对象级处理
        self.object_encoder = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, object_dim)
        )

        # 规则编码器
        self.rule_encoder = nn.Sequential(
            nn.Linear(object_dim * 2, 256),  # 输入是两个对象表示的拼接
            nn.LeakyReLU(),
            nn.Linear(256, rule_dim)
        )

        # 规则量化器 - 使用与原代码兼容的参数
        self.rule_quantizer = VectorQuantizer(
            num_embeddings=rule_codebook_size,
            embedding_dim=rule_dim,
            commitment_cost=0.25
        )

        # 规则解码器（应用规则）
        self.rule_applier = nn.Sequential(
            nn.Linear(object_dim + rule_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, object_dim)
        )

        # 解码器：从对象表示直接生成网格的类别预测
        self.decoder = nn.Sequential(
            nn.Linear(object_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, grid_size * grid_size * num_categories),  # 直接输出每个像素位置的类别预测
        )

    def extract_rule(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从输入输出对中提取规则表示

        参数:
            input_grid: [B, C, H, W] 输入网格，one-hot编码
            output_grid: [B, C, H, W] 输出网格，one-hot编码

        返回:
            包含规则表示的字典
        """
        # 编码输入和输出网格为中间表示
        input_features = self.encoder(input_grid)
        output_features = self.encoder(output_grid)

        # 扁平化特征
        batch_size = input_grid.shape[0]
        input_features_flat = input_features.view(batch_size, -1)
        output_features_flat = output_features.view(batch_size, -1)

        # 编码为对象级表示
        input_object = self.object_encoder(input_features_flat)
        output_object = self.object_encoder(output_features_flat)

        # 量化对象表示
        input_object_q, input_vq_loss, _ = self.object_quantizer(input_object)  # 适应原代码返回值
        output_object_q, output_vq_loss, _ = self.object_quantizer(output_object)  # 适应原代码返回值

        # 拼接输入和输出对象表示
        combined = torch.cat([input_object_q, output_object_q], dim=1)

        # 提取规则表示
        rule = self.rule_encoder(combined)

        # 量化规则表示
        rule_q, rule_vq_loss, _ = self.rule_quantizer(rule)  # 适应原代码返回值

        # 总的VQ损失
        vq_loss = input_vq_loss + output_vq_loss + rule_vq_loss

        return {
            'input_object': input_object_q,
            'output_object': output_object_q,
            'rule_embedding': rule_q,
            'vq_loss': vq_loss
        }

    def apply_rule(self, input_grid: torch.Tensor, rule: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将提取的规则应用到新的输入网格

        参数:
            input_grid: [B, C, H, W] 输入网格，one-hot编码
            rule: 从extract_rule返回的规则表示

        返回:
            预测的输出网格 [B, grid_cells, num_categories]
        """
        # 编码输入网格
        input_features = self.encoder(input_grid)

        # 扁平化特征
        batch_size = input_grid.shape[0]
        input_features_flat = input_features.view(batch_size, -1)

        # 编码为对象级表示
        input_object = self.object_encoder(input_features_flat)

        # 量化对象表示
        input_object_q, _, _ = self.object_quantizer(input_object)  # 适应原代码返回值

        # 拼接输入对象表示和规则表示
        combined = torch.cat([input_object_q, rule['rule_embedding']], dim=1)

        # 应用规则生成输出对象表示
        output_object = self.rule_applier(combined)

        # 解码为像素类别预测
        pixel_logits_flat = self.decoder(output_object)

        # 重塑为 [batch_size, grid_cells, num_categories] 格式
        grid_cells = self.grid_size * self.grid_size
        pixel_logits = pixel_logits_flat.view(batch_size, grid_cells, self.num_categories)

        return pixel_logits

    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播：提取规则并应用

        参数:
            input_grid: [B, C, H, W] 输入网格，one-hot编码
            output_grid: [B, C, H, W] 输出网格，one-hot编码

        返回:
            预测的输出网格和损失字典
        """
        # 提取规则
        rule = self.extract_rule(input_grid, output_grid)

        # 应用规则
        predicted_output = self.apply_rule(input_grid, rule)

        # 重构损失（交叉熵）
        output_flat = torch.argmax(output_grid, dim=1).reshape(-1)
        recon_loss = F.cross_entropy(
            predicted_output.reshape(-1, self.num_categories),
            output_flat
        )

        # VQ损失
        vq_loss = rule['vq_loss']

        losses = {
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'total_loss': recon_loss + 0.1 * vq_loss
        }

        return predicted_output, losses

    def reconstruct_grid(self, input_grid: torch.Tensor) -> torch.Tensor:
        """
        重建输入网格（自编码）

        参数:
            input_grid: [B, C, H, W] 输入网格，one-hot编码

        返回:
            重建的网格 [B, grid_cells, num_categories]
        """
        # 编码
        features = self.encoder(input_grid)

        # 扁平化特征
        batch_size = input_grid.shape[0]
        features_flat = features.view(batch_size, -1)

        # 编码为对象级表示
        object_repr = self.object_encoder(features_flat)

        # 解码为像素类别预测
        pixel_logits_flat = self.decoder(object_repr)

        # 重塑为 [batch_size, grid_cells, num_categories] 格式
        grid_cells = self.grid_size * self.grid_size
        pixel_logits = pixel_logits_flat.view(batch_size, grid_cells, self.num_categories)

        return pixel_logits






















