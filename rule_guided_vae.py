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
    """整合了规则提取与应用的层次化VAE模型"""
    def __init__(self, grid_size=30, num_categories=10,
                 pixel_codebook_size=512, object_codebook_size=256, rule_codebook_size=128,
                 pixel_dim=64, object_dim=128, relation_dim=64, rule_dim=128):
        super().__init__()




        # 1. 抽象阶段组件
        self.encoder = HierarchicalEncoder(grid_size, num_categories)
        self.object_detector = ConnectedComponentsModule(grid_size, num_categories)
        self.pixel_vq = VectorQuantizer(pixel_codebook_size, pixel_dim, commitment_cost=0.1, decay=0.95)
        self.object_vq = VectorQuantizer(object_codebook_size, object_dim, commitment_cost=0.1, decay=0.95)

        # 2. 操作处理阶段组件
        self.operation_library = OperationLibrary()  # 包含基本操作集
        # self.explicit_inference = ExplicitRuleInference(self.operation_library, rule_dim=128)
        self.rule_extractor = RuleExtractor(input_dim=512, hidden_dim=256, output_dim=rule_dim)
        self.rule_vq = VectorQuantizer(rule_codebook_size, rule_dim, commitment_cost=0.1, decay=0.95)

        # 3. 对象组合与规则应用阶段
        self.rule_applier = RuleApplier(operation_library=self.operation_library)
        self.object_composer = ObjectComposer()

        # 4. 输出生成阶段
        self.decoder = HierarchicalDecoder(grid_size, num_categories, pixel_dim, object_dim, relation_dim)

        # 5. 推理与评估组件
        self.rule_evaluator = RuleEvaluator()

        # 关系推理模块（从原始层次化VAE继承）
        self.relation_module = RelationalReasoningModule(feature_dim=object_dim, output_dim=relation_dim)

        # 投影器（从原始层次化VAE继承）
        self.pixel_projector = nn.Sequential(
            nn.Conv2d(64, pixel_dim, 1),
            nn.GroupNorm(8, pixel_dim),
            nn.LeakyReLU(0.2)
        )

        self.object_projector = nn.Sequential(
            nn.Linear(128, object_dim),
            nn.LayerNorm(object_dim),
            nn.LeakyReLU(0.2)
        )

        # 梯度缩放器
        self._scaler = 0.1

        # 网格尺寸和类别数
        self.grid_size = grid_size
        self.num_categories = num_categories

        self.feature_sizes = {
            'encoder_output': 15,  # 编码器输出特征图大小
            'decoder_input': 15    # 解码器输入特征图大小
        }


    def abstract_scene(self, grid):
        """第一阶段：场景抽象，提取像素和对象表示"""
        batch_size = grid.size(0)

        # 像素级特征
        features = self.encoder(grid)
        pixel_features = self.pixel_projector(features['low'])
        pixel_features = pixel_features * self._scaler  # 应用梯度缩放
        pixel_quantized, pixel_vq_loss, _ = self.pixel_vq(pixel_features)

        # 对象级特征
        object_features_list = []
        object_masks_list = []

        for b in range(batch_size):
            objects, masks = self.object_detector(grid[b:b+1])
            object_features_list.append(objects)
            object_masks_list.append(masks)

        # 处理对象特征
        processed_obj_features = []
        for obj_feat in object_features_list:
            # 使用对象投影器处理每个对象
            if obj_feat and obj_feat[0].shape[0] > 0:  # 有对象
                proj_feats = [self.object_projector(feat) for feat in obj_feat]
                processed_obj_features.append(proj_feats)
            else:  # 无对象情况
                processed_obj_features.append([torch.zeros(1, self.object_dim, device=grid.device)])

        # 量化对象特征
        quantized_objects_list = []
        object_vq_loss_total = 0

        for batch_idx, batch_objs in enumerate(processed_obj_features):
            batch_quantized = []
            for obj in batch_objs:
                obj_scaled = obj * self._scaler  # 应用梯度缩放
                obj_quantized, obj_vq_loss, _ = self.object_vq(obj_scaled.unsqueeze(1))
                batch_quantized.append(obj_quantized.squeeze(1))
                object_vq_loss_total += obj_vq_loss
            quantized_objects_list.append(batch_quantized)

        # 计算关系特征
        relation_features_list = []
        for batch_objs in object_features_list:
            if batch_objs and batch_objs[0].shape[0] > 0:
                relation = self.relation_module(batch_objs)
            else:
                relation = torch.zeros(1, relation_dim, device=grid.device)
            relation_features_list.append(relation * self._scaler)  # 应用梯度缩放

        relation_features = torch.cat(relation_features_list, dim=0)

        # 返回抽象表示
        return {
            'pixel_features': pixel_quantized,
            'object_features': quantized_objects_list,
            'relation_features': relation_features,
            'masks': object_masks_list,
            'high_features': features['high'],
            'mid_features': features['mid'],
            'pixel_vq_loss': pixel_vq_loss,
            'object_vq_loss': object_vq_loss_total / batch_size if batch_size > 0 else 0
        }

    def extract_rule(self, input_grid, output_grid):
        """第二阶段：从输入-输出对中提取规则"""
        # 抽象输入和输出
        input_abstract = self.abstract_scene(input_grid)
        output_abstract = self.abstract_scene(output_grid)

        # 分析差异和变换
        rule_embedding = self.rule_extractor(
            input_abstract['high_features'],
            output_abstract['high_features'],
            [obj[0] for obj in input_abstract['object_features']],  # 简化：每批次取第一个对象
            [obj[0] for obj in output_abstract['object_features']]
        )

        # 量化规则表示
        rule_embedding = rule_embedding * self._scaler  # 应用梯度缩放
        rule_quantized, rule_vq_loss, rule_indices = self.rule_vq(rule_embedding)

        # 提取具体操作序列
        operations = self.operation_library.decode_operations(rule_quantized)

        # 计算总VQ损失
        total_vq_loss = (
            input_abstract['pixel_vq_loss'] +
            input_abstract['object_vq_loss'] +
            output_abstract['pixel_vq_loss'] +
            output_abstract['object_vq_loss'] +
            rule_vq_loss
        )

        return {
            'rule_embedding': rule_quantized,
            'operations': operations,
            'indices': rule_indices,
            'vq_loss': total_vq_loss
        }


    def apply_rule(self, input_grid, rule):
        """第三阶段：应用规则到输入以生成预测输出"""
        # 抽象输入
        input_abstract = self.abstract_scene(input_grid)

        # 提取各层次特征
        batch_size = input_grid.size(0)
        device = input_grid.device

        # 应用规则转换
        transformed_objects_list = []
        transformed_pixels_list = []
        for b in range(batch_size):
            # 对每个批次应用规则
            batch_objects = input_abstract['object_features'][b]
            batch_rule = rule['rule_embedding'][b:b+1]

            transformed_objects, transformed_pixels = self.rule_applier(
                batch_objects,
                input_abstract['pixel_features'][b:b+1],
                batch_rule,
                rule['operations'] if 'operations' in rule else None
            )
            transformed_objects_list.append(transformed_objects)
            transformed_pixels_list.append(transformed_pixels)

        # 处理批量输出
        output_batch = []
        for b in range(batch_size):
            # 关键修复：将对象列表转换为解码器需要的张量格式
            # HierarchicalDecoder期望形状为[batch_size=1, num_features]的张量

            # 检查是否有对象
            if transformed_objects_list[b] and len(transformed_objects_list[b]) > 0:
                # 将所有对象特征平均合并为一个特征向量
                object_tensor = torch.stack([
                    obj.mean(0) if obj.dim() > 1 else obj
                    for obj in transformed_objects_list[b]
                ]).mean(0, keepdim=True)  # [1, feature_dim]
            else:
                # 如果没有对象，创建全零张量
                object_tensor = torch.zeros(1, self.object_dim, device=device)

            # 调用解码器
            output = self.decoder(
                transformed_pixels_list[b],    # [1, channels, H, W]
                object_tensor,                # [1, feature_dim]
                input_abstract['relation_features'][b:b+1],  # [1, relation_dim]
                input_abstract['high_features'][b:b+1]       # [1, high_dim]
            )
            output_batch.append(output)

        # 合并批次结果
        final_output = torch.cat(output_batch, dim=0)
        return final_output




    def evaluate_rule(self, predicted_output, ground_truth):
        """第四阶段：评估规则应用结果与目标的一致性"""
        return self.rule_evaluator(predicted_output, ground_truth)

    def forward(self, input_grid, output_grid=None, test_input=None, mode='reconstruction'):
        """综合前向传播函数"""
        if mode == 'rule_extraction' and output_grid is not None:
            # 训练模式：仅提取规则
            rule = self.extract_rule(input_grid, output_grid)
            return rule

        elif mode == 'rule_application' and test_input is not None and 'rule_embedding' in input_grid:
            # 应用给定规则到测试输入
            predicted_output = self.apply_rule(test_input, input_grid)
            return predicted_output

        elif mode == 'full_task' and output_grid is not None and test_input is not None:
            # 完整模式：提取规则并应用
            rule = self.extract_rule(input_grid, output_grid)
            predicted_output = self.apply_rule(test_input, rule)
            return predicted_output, rule

        else:
            # 默认模式：与当前VAE兼容的重构
            abstract = self.abstract_scene(input_grid)

            # 解码重建
            reconstruction = self.decoder(
                abstract['pixel_features'],
                [obj[0] for obj in abstract['object_features']],  # 简化：每批次取第一个对象
                abstract['relation_features'],
                abstract['high_features']
            )

            # 计算总VQ损失
            vq_loss = abstract['pixel_vq_loss'] + abstract['object_vq_loss']

            return reconstruction, None, None, None, vq_loss, None