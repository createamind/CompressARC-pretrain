import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import math
from typing import List, Dict, Tuple, Any

class ConnectedComponentsModule(nn.Module):
    """提取并编码图像中的连通对象"""
    def __init__(self, grid_size=30, num_categories=10, connectivity=8, max_objects=20):
        super().__init__()
        self.grid_size = grid_size
        self.num_categories = num_categories
        self.connectivity = connectivity  # 4或8连通
        self.max_objects = max_objects  # 每个图像最大对象数

        # 对象特征提取器
        self.object_encoder = nn.Sequential(
            nn.Linear(4, 32),  # 输入：[x_min, y_min, x_max, y_max]
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 颜色特征编码器
        self.color_encoder = nn.Embedding(num_categories, 32)

        # 大小特征编码器
        self.size_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # 组合所有特征
        self.feature_combiner = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def extract_components(self, x_batch):
        """从批次中提取连通组件"""
        batch_size = x_batch.size(0)
        device = x_batch.device
        all_objects = []

        for b in range(batch_size):
            # 获取单个样本并转换为numpy以进行连通组件分析
            x = x_batch[b].argmax(dim=0).cpu().numpy()  # [grid_size, grid_size]

            objects = []
            # 对每个颜色单独处理
            for color in range(1, self.num_categories):  # 跳过背景色(0)
                # 创建二值掩码
                mask = (x == color).astype(np.uint8)
                if np.sum(mask) == 0:
                    continue  # 跳过不存在的颜色

                # 提取连通组件
                if self.connectivity == 4:
                    structure = [[0,1,0],[1,1,1],[0,1,0]]
                else:  # 8-连通
                    structure = [[1,1,1],[1,1,1],[1,1,1]]

                labeled, num_components = ndimage.label(mask, structure=structure)

                # 处理每个组件
                for comp_idx in range(1, num_components+1):
                    comp_mask = (labeled == comp_idx)
                    if np.sum(comp_mask) < 2:  # 跳过太小的组件
                        continue

                    # 计算边界框
                    rows, cols = np.where(comp_mask)
                    x_min, y_min = np.min(cols), np.min(rows)
                    x_max, y_max = np.max(cols), np.max(rows)

                    # 记录对象信息
                    objects.append({
                        'color': color,
                        'bbox': [x_min/self.grid_size, y_min/self.grid_size,
                                x_max/self.grid_size, y_max/self.grid_size],  # 归一化坐标
                        'size': np.sum(comp_mask)/(self.grid_size*self.grid_size),  # 归一化大小
                        'mask': torch.from_numpy(comp_mask).to(device)
                    })

            # 限制每个图像的对象数量
            if len(objects) > self.max_objects:
                # 按对象大小排序，保留最大的
                objects.sort(key=lambda obj: obj['size'], reverse=True)
                objects = objects[:self.max_objects]

            all_objects.append(objects)

        return all_objects

    def forward(self, x):
        # 提取连通组件
        batch_objects = self.extract_components(x)

        # 编码所有对象
        batch_size = len(batch_objects)
        batch_features = []
        obj_masks = []

        for b in range(batch_size):
            objects = batch_objects[b]
            if not objects:
                # 如果没有对象，使用零向量
                batch_features.append(torch.zeros(1, 128, device=x.device))
                obj_masks.append(None)
                continue

            obj_features = []
            masks = []
            for obj in objects:
                # 编码边界框
                bbox_tensor = torch.tensor(obj['bbox'], dtype=torch.float, device=x.device)
                bbox_features = self.object_encoder(bbox_tensor)

                # 编码颜色
                color_tensor = torch.tensor(obj['color'], dtype=torch.long, device=x.device)
                color_features = self.color_encoder(color_tensor)

                # 编码大小
                size_tensor = torch.tensor([obj['size']], dtype=torch.float, device=x.device)
                size_features = self.size_encoder(size_tensor)

                # 组合特征
                combined = torch.cat([bbox_features, color_features, size_features], dim=0)
                obj_features.append(self.feature_combiner(combined))
                masks.append(obj['mask'])

            # 将所有对象特征堆叠起来
            if obj_features:
                obj_features = torch.stack(obj_features)
            else:
                obj_features = torch.zeros(1, 128, device=x.device)

            batch_features.append(obj_features)
            obj_masks.append(masks)

        return batch_features, obj_masks


class RelationalReasoningModule(nn.Module):
    # 保持不变但使用输出维度变量...
    def __init__(self, feature_dim=128, output_dim=128):
        super().__init__()
        self.output_dim = output_dim

        # 关系编码器
        self.relation_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        # 全局关系聚合
        self.global_aggregator = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, object_features):
        """
        输入: 对象特征列表，每个批次样本包含若干对象
        输出: 关系表示
        """
        batch_relations = []

        for objects in object_features:
            # 如果对象数量不足，则返回零向量
            if objects.size(0) <= 1:
                batch_relations.append(torch.zeros(1, self.output_dim, device=objects.device))
                continue

            # 计算所有对象对之间的关系
            n_objects = objects.size(0)
            relations = []

            for i in range(n_objects):
                for j in range(i+1, n_objects):
                    # 连接两个对象的特征
                    obj_pair = torch.cat([objects[i], objects[j]], dim=0)
                    # 编码它们的关系
                    relation = self.relation_encoder(obj_pair.unsqueeze(0))
                    relations.append(relation)

            if relations:
                relations = torch.cat(relations, dim=0)
                # 聚合所有关系
                aggregated = self.global_aggregator(torch.mean(relations, dim=0, keepdim=True))
                batch_relations.append(aggregated)
            else:
                batch_relations.append(torch.zeros(1, self.output_dim, device=objects.device))

        # 堆叠所有批次样本的关系表示
        return torch.cat(batch_relations, dim=0)





class VectorQuantizer(nn.Module):
    """
    向量量化器 - 将连续向量映射到离散的编码本
    增加了数值稳定性措施
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # 使用更保守的初始化
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 使用正态分布初始化而非均匀分布
        self.embedding.weight.data.normal_(0, 0.02)

        # 使用指数移动平均(EMA)来更新编码本
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())

        # 添加安全检查标志
        self.safe_mode = True

    def forward(self, inputs):
        # 输入形状: [B, C, H, W] 或 [B, C]
        input_shape = inputs.shape

        # 添加特征归一化以增加稳定性
        inputs = F.normalize(inputs, p=2, dim=1) * math.sqrt(self.embedding_dim)

        # 扁平化空间维度以用于计算
        if len(input_shape) == 4:
            # 卷积特征情况 [B, C, H, W]
            flat_input = inputs.permute(0, 2, 3, 1).contiguous()
            flat_input = flat_input.view(-1, self.embedding_dim)
        elif len(input_shape) == 3:
            # 可变长序列情况 [B, N, C]
            flat_input = inputs.reshape(-1, self.embedding_dim)
        else:
            # 批量向量情况 [B, C]
            flat_input = inputs

        # 为了数值稳定性，使用归一化的权重
        normalized_embeddings = F.normalize(self.embedding.weight, p=2, dim=1) * math.sqrt(self.embedding_dim)

        # 使用更稳定的距离计算方法（欧氏距离平方的数值稳定版本）
        # 避免显式计算平方项，而是直接使用矩阵乘法
        similarity = torch.matmul(flat_input, normalized_embeddings.t())

        # 找到最近的编码索引（最高相似度）
        encoding_indices = torch.argmax(similarity, dim=1).unsqueeze(1)

        # 使用one-hot进行编码
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化: 从编码本中查找最近的编码
        quantized = torch.matmul(encodings, self.embedding.weight)

        # 在训练时使用EMA更新编码本
        if self.training:
            # 更新编码本的ema统计信息
            self._ema_update(flat_input, encodings)

        # 计算损失 - 增加梯度缩放以增加稳定性
        q_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        e_latent_loss = F.mse_loss(quantized, flat_input.detach())

        # 使用梯度缩放防止损失过大
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 添加损失值检查
        if self.safe_mode and (torch.isnan(loss) or torch.isinf(loss) or loss > 1e5):
            print(f"警告: VQ损失异常! q_loss={q_latent_loss.item():.4f}, e_loss={e_latent_loss.item():.4f}")
            # 在异常情况下返回小损失值以避免训练崩溃
            loss = torch.tensor(0.1, device=inputs.device, requires_grad=True)

        # 直通估计器 (straight-through estimator)
        quantized = flat_input + (quantized - flat_input).detach()

        # 恢复原始形状
        if len(input_shape) == 4:
            quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
        elif len(input_shape) == 3:
            quantized = quantized.view(input_shape[0], input_shape[1], self.embedding_dim)

        return quantized, loss, encoding_indices

    def _ema_update(self, flat_input, encodings):
        """使用EMA更新编码本 - 添加保护措施"""
        # 计算每个编码的使用频率
        batch_mean = torch.mean(encodings, dim=0)

        # 添加梯度保护
        with torch.no_grad():
            # 更新集群大小
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * batch_mean

            # 拉普拉斯平滑，防止某些编码永不使用
            n = torch.sum(self.ema_cluster_size)
            weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

            # 计算每个编码的新中心
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            # 更新编码本权重，使用梯度裁剪防止极端值
            updated_weights = self.ema_w / weights.unsqueeze(1).clamp(min=1e-5)

            # 防止权重爆炸
            norm = torch.norm(updated_weights, dim=1, keepdim=True)
            normalized = updated_weights / norm.clamp(min=1e-5)
            self.embedding.weight.data = normalized * math.sqrt(self.embedding_dim)

    # 在VectorQuantizer类中添加此方法
    def get_codebook_indices(self, inputs):
        """返回最近编码的索引，用于监控编码本使用情况"""
        # 与forward相同的前处理
        input_shape = inputs.shape

        # 扁平化空间维度以用于计算
        if len(input_shape) == 4:
            flat_input = inputs.permute(0, 2, 3, 1).contiguous()
            flat_input = flat_input.view(-1, self.embedding_dim)
        elif len(input_shape) == 3:
            flat_input = inputs.reshape(-1, self.embedding_dim)
        else:
            flat_input = inputs

        # 计算距离并找到最近的编码
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.embedding.weight.t())

        # 返回最近的编码索引
        return torch.argmin(distances, dim=1)


# [ConnectedComponentsModule和RelationalReasoningModule类保持不变]

class HierarchicalEncoder(nn.Module):
    """分层次特征编码器 - 增加稳定性"""
    def __init__(self, grid_size=30, num_categories=10):
        super().__init__()

        # 计算下采样后的特征图尺寸
        self.level1_size = grid_size // 2
        self.level2_size = (self.level1_size + 2*1 - 3) // 2 + 1

        print(f"编码器特征图尺寸: {grid_size} -> {self.level1_size} -> {self.level2_size}")

        # 低级特征提取：像素级模式
        self.low_level_encoder = nn.Sequential(
            nn.Conv2d(num_categories, 64, 3, padding=1),
            nn.GroupNorm(8, 64),  # 使用GroupNorm代替LayerNorm提高稳定性
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 降采样到 15x15
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
        )

        # 中级特征提取：局部结构
        self.mid_level_encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 降采样到 8x8
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2),
        )

        # 高级特征提取：全局概念
        self.high_level_encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),  # 自适应池化到固定大小4x4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # 提取层次化特征
        low_features = self.low_level_encoder(x)
        mid_features = self.mid_level_encoder(low_features)
        high_features = self.high_level_encoder(mid_features)

        return {
            'low': low_features,
            'mid': mid_features,
            'high': high_features
        }



class HierarchicalDecoder(nn.Module):
    """修复版增强分层次解码器 - 确保输出格式兼容"""
    def __init__(self, grid_size=30, num_categories=10, pixel_dim=64, object_dim=128, relation_dim=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_categories = num_categories
        self.object_dim = object_dim

        # 保持原有的尺寸参数
        self.pool_size = 4
        self.level2_size = 8  # 从4x4 -> 8x8
        self.level1_size = 16  # 从8x8 -> 16x16

        print(f"修复解码器特征图尺寸: 4x4 -> {self.level2_size}x{self.level2_size} -> {self.level1_size}x{self.level1_size} -> {grid_size}x{grid_size}")

        # 高级特征处理 - 恢复与原始解码器相似的结构
        self.high_processor = nn.Sequential(
            nn.Linear(512 + relation_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        # 中级特征解码 - 从4x4到8x8
        self.mid_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 4, stride=2, padding=1),  # 256->192
            nn.GroupNorm(16, 192),
            nn.LeakyReLU(0.2)
        )

        # 低级特征解码 - 从8x8到16x16
        self.low_decoder = nn.Sequential(
            nn.ConvTranspose2d(192 + object_dim, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2)
        )

        # 最终解码 - 从16x16到30x30，确保输出 [batch, num_categories, H, W]
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + pixel_dim, 96, 4, stride=2, padding=1),
            nn.GroupNorm(16, 96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, num_categories, 3, padding=1)
        )

        # 简化的对象特征处理
        self.object_adapter = nn.Linear(object_dim, object_dim)

    def forward(self, pixel_features, object_features, relation_features, high_features):
        batch_size = high_features.shape[0]

        # 1. 处理高级特征与关系特征
        combined_high = torch.cat([high_features, relation_features], dim=1)
        high_processed = self.high_processor(combined_high)
        high_processed = high_processed.view(batch_size, 256, 4, 4)  # 重要：确保正确的形状

        # 2. 解码中级特征
        mid_decoded = self.mid_decoder(high_processed)  # [batch, 192, 8, 8]

        # 3. 处理对象特征 - 使用简化且可靠的方法
        # 创建统一大小的对象特征图
        object_map = torch.zeros(batch_size, self.object_dim,
                                self.level2_size, self.level2_size,
                                device=high_features.device)

        # 处理对象特征
        if isinstance(object_features, torch.Tensor) and object_features.dim() > 1:
            # 对于张量，简单处理并扩展
            obj_feat = self.object_adapter(object_features.reshape(-1, self.object_dim))
            obj_feat = obj_feat.view(batch_size, -1, self.object_dim)  # [batch, objects, dim]

            # 取每个批次的平均特征
            mean_obj = obj_feat.mean(dim=1)  # [batch, dim]

            # 扩展到空间维度
            for b in range(batch_size):
                object_map[b] = mean_obj[b].view(-1, 1, 1).expand(-1, self.level2_size, self.level2_size)

        # 4. 合并中级特征和对象特征
        mid_with_objects = torch.cat([mid_decoded, object_map], dim=1)

        # 5. 解码低级特征
        low_decoded = self.low_decoder(mid_with_objects)  # [batch, 128, 16, 16]

        # 6. 将像素特征上采样
        pixel_upsampled = F.interpolate(pixel_features,
                                       size=(self.level1_size, self.level1_size),
                                       mode='bilinear',
                                       align_corners=False)

        # 7. 合并低级特征和像素特征
        low_with_pixels = torch.cat([low_decoded, pixel_upsampled], dim=1)

        # 8. 最终解码 - 确保输出正确形状
        output = self.final_decoder(low_with_pixels)

        # 剪裁到正确大小
        output = output[:, :, :self.grid_size, :self.grid_size]

        # 打印形状以确认
        # print(f"解码器输出形状: {output.shape}")

        return output









class ObjectOrientedHierarchicalVAE(nn.Module):
    def __init__(self, grid_size=30, num_categories=10,
                 pixel_codebook_size=512, object_codebook_size=256, relation_codebook_size=128,
                 pixel_dim=64, object_dim=128, relation_dim=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_categories = num_categories
        self.grid_cells = grid_size * grid_size
        self.pixel_dim = pixel_dim
        self.object_dim = object_dim
        self.relation_dim = relation_dim

        # 分层编码器
        self.hierarchical_encoder = HierarchicalEncoder(grid_size, num_categories)
        self.level1_size = self.hierarchical_encoder.level1_size  # 15

        # 对象连通性提取模块
        self.connectivity_module = ConnectedComponentsModule(
            grid_size=grid_size,
            num_categories=num_categories,
            connectivity=8  # 使用8连通
        )

        # 关系推理模块
        self.relation_module = RelationalReasoningModule(feature_dim=128, output_dim=relation_dim)

        # 分离的潜在空间 - 使用更小的commitment_cost增加稳定性
        # self.pixel_vq = VectorQuantizer(
        #     num_embeddings=pixel_codebook_size,
        #     embedding_dim=pixel_dim,
        #     commitment_cost=0.1
        # )
        # 在hierarchical_vae.py中
        self.pixel_vq = VectorQuantizer(
            num_embeddings=pixel_codebook_size,
            embedding_dim=pixel_dim,
            commitment_cost=0.1,
            decay=0.95  # 降低从0.99到0.95，增加编码本更新速度
        )
        self.object_vq = VectorQuantizer(
            num_embeddings=object_codebook_size,
            embedding_dim=object_dim,
            commitment_cost=0.1
        )
        self.relation_vq = VectorQuantizer(
            num_embeddings=relation_codebook_size,
            embedding_dim=relation_dim,
            commitment_cost=0.1
        )

        # 潜在空间投影
        self.pixel_projector = nn.Sequential(
            nn.Conv2d(64, pixel_dim, 1),
            nn.GroupNorm(8, pixel_dim),  # 使用GroupNorm增加稳定性
            nn.LeakyReLU(0.2)
        )

        self.object_projector = nn.Sequential(
            nn.Linear(128, object_dim),
            nn.LayerNorm(object_dim),
            nn.LeakyReLU(0.2)
        )

        # 层次化解码器
        self.decoder = HierarchicalDecoder(
            grid_size=grid_size,
            num_categories=num_categories,
            pixel_dim=pixel_dim,
            object_dim=object_dim,
            relation_dim=relation_dim
        )

        # 梯度缩放器
        self._scaler = 0.1

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 提取层次特征
        features = self.hierarchical_encoder(x)

        # 2. 提取对象级特征
        object_features, object_masks = self.connectivity_module(x)

        # 处理object_features为批量张量
        processed_obj_features = []
        for obj_feat in object_features:
            # 使用对象投影器处理每个对象
            if obj_feat.shape[0] > 1:  # 有多个对象
                proj_feats = self.object_projector(obj_feat)
            else:  # 单个对象或零对象的情况
                proj_feats = torch.zeros(1, self.object_dim, device=x.device)
            # 使用平均池化聚合多个对象特征，并应用梯度缩放
            processed_obj_features.append(proj_feats.mean(0) * self._scaler)

        batch_obj_features = torch.stack(processed_obj_features)

        # 3. 提取关系特征
        relation_features = self.relation_module(object_features)

        # 应用梯度缩放
        relation_features = relation_features * self._scaler

        # 4. 投影到适合量化的空间
        pixel_proj = self.pixel_projector(features['low'])
        pixel_proj = pixel_proj * self._scaler  # 缩放以增加稳定性

        # 5. 应用向量量化
        try:
            pixel_quantized, pixel_vq_loss, _ = self.pixel_vq(pixel_proj)
            object_quantized, object_vq_loss, _ = self.object_vq(batch_obj_features.unsqueeze(1))
            relation_quantized, relation_vq_loss, _ = self.relation_vq(relation_features)
        except Exception as e:
            print(f"向量量化出错: {e}")
            # 提供安全的后备方案
            pixel_quantized = pixel_proj
            object_quantized = batch_obj_features.unsqueeze(1)
            relation_quantized = relation_features
            pixel_vq_loss = torch.tensor(0.1, device=x.device)
            object_vq_loss = torch.tensor(0.1, device=x.device)
            relation_vq_loss = torch.tensor(0.1, device=x.device)

        # 6. 解码综合信息
        object_quantized = object_quantized.squeeze(1)  # 移除添加的维度

        # 检查异常值
        if torch.isnan(pixel_quantized).any() or torch.isinf(pixel_quantized).any():
            print("警告: 检测到NaN/Inf值，使用原始特征替代")
            pixel_quantized = pixel_proj
        if torch.isnan(object_quantized).any() or torch.isinf(object_quantized).any():
            object_quantized = batch_obj_features
        if torch.isnan(relation_quantized).any() or torch.isinf(relation_quantized).any():
            relation_quantized = relation_features

        reconstruction = self.decoder(
            pixel_quantized,
            object_quantized,
            relation_quantized,
            features['high']
        )

        # 7. 计算总VQ损失 - 使用更平衡的权重
        total_vq_loss = pixel_vq_loss + 0.5*object_vq_loss + 0.5*relation_vq_loss

        # 检测并防止NaN/Inf损失
        if torch.isnan(total_vq_loss) or torch.isinf(total_vq_loss):
            print("警告: VQ损失是NaN/Inf，替换为小常数")
            total_vq_loss = torch.tensor(0.1, device=x.device, requires_grad=True)

        # 返回重建结果和VQ损失
        return reconstruction, None, None, None, total_vq_loss, None