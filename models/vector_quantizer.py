import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


class VectorQuantizer(nn.Module):
    """
    GPU优化的向量量化器 - 使用指数移动平均(EMA)更新codebook
    包含多种数值稳定性增强措施
    """
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 commitment_cost=0.25,
                 decay=0.99,
                 epsilon=1e-5,
                 use_ema=True,
                 threshold_ema_dead_code=0):
        super().__init__()
        
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.threshold_ema_dead_code = threshold_ema_dead_code
        
        # 使用更好的初始化方法
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight, gain=1.0)
        
        # EMA相关缓冲区
        if use_ema:
            self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
            self._decay = decay
            self._epsilon = epsilon
            # 标记是否已初始化
            self.register_buffer('_initialized', torch.tensor(0, dtype=torch.bool))
        
        # 用于调试: 跟踪编码本使用情况
        self.register_buffer('_codebook_usage', torch.zeros(num_embeddings))
        self.register_buffer('_total_usage', torch.tensor(0, dtype=torch.long))

    def _tile_inputs(self, inputs):
        """处理不同形状的输入"""
        if len(inputs.shape) == 4:  # [B, C, H, W]
            inputs = inputs.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        input_shape = inputs.shape
        
        # 保留原始形状以便重建
        flat_input = inputs.view(-1, self._embedding_dim)
        
        return flat_input, input_shape
    
    def _safe_distances(self, flat_input):
        """计算输入和编码本向量之间的欧氏距离(数值稳定版)"""
        # 计算 (x - y)^2 = x^2 - 2xy + y^2
        x_squared = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        y_squared = torch.sum(self.embedding.weight ** 2, dim=1)
        two_xy = 2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # 添加小epsilon确保数值稳定性
        distances = x_squared - two_xy + y_squared.unsqueeze(0) + 1e-8
        return distances
    
    def _get_encoding_indices(self, distances):
        """获取最近的编码向量索引"""
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def _update_ema(self, flat_input, encodings):
        """使用EMA更新编码本"""
        with torch.no_grad():
            # 确保第一批次正确初始化
            if not self._initialized and self.training:
                # 使用K-means++启发式初始化
                if flat_input.shape[0] >= self._num_embeddings:
                    # 随机选择第一个中心点
                    indices = torch.zeros(self._num_embeddings, dtype=torch.long, device=flat_input.device)
                    indices[0] = torch.randint(0, flat_input.shape[0], (1,), device=flat_input.device)
                    
                    # 选择其余中心点
                    for i in range(1, self._num_embeddings):
                        # 计算每个点到已选中心点的最小距离
                        centers = flat_input[indices[:i]]
                        dists = torch.cdist(flat_input, centers, p=2.0)
                        min_dists, _ = torch.min(dists, dim=1)
                        
                        # 概率与距离平方成正比
                        weights = min_dists ** 2
                        if torch.sum(weights) > 0:
                            weights /= torch.sum(weights)
                            # 处理NaN
                            weights = torch.where(torch.isnan(weights), 
                                                torch.zeros_like(weights), 
                                                weights)
                            if torch.sum(weights) > 0:
                                idx = torch.multinomial(weights, 1)
                                indices[i] = idx
                            else:
                                indices[i] = torch.randint(0, flat_input.shape[0], (1,), device=flat_input.device)
                        else:
                            indices[i] = torch.randint(0, flat_input.shape[0], (1,), device=flat_input.device)
                    
                    # 设置初始编码本
                    self.embedding.weight.data.copy_(flat_input[indices])
                    
                else:
                    # 如果批次太小，使用批次中的样本并重复
                    samples = flat_input.repeat(self._num_embeddings // flat_input.shape[0] + 1, 1)
                    self.embedding.weight.data.copy_(samples[:self._num_embeddings])
                    
                # 初始化EMA计数器
                self._ema_cluster_size.data.fill_(1)
                self._initialized.fill_(True)
                
                logging.info(f"VQ编码本已使用K-means++启发式初始化")
            
            # 标准EMA更新
            encodings_one_hot = F.one_hot(encodings, self._num_embeddings).float()
            
            # 更新簇大小
            new_cluster_size = encodings_one_hot.sum(0)
            self._ema_cluster_size.data.mul_(self._decay).add_(
                new_cluster_size, alpha=(1 - self._decay)
            )
            
            # 更新嵌入权重和
            new_embeddings_sum = torch.matmul(encodings_one_hot.t(), flat_input)
            self._ema_w.data.mul_(self._decay).add_(
                new_embeddings_sum, alpha=(1 - self._decay)
            )
            
            # 处理"死亡"编码 (长期不用的编码)
            if self.threshold_ema_dead_code > 0:
                n_total = new_cluster_size.sum().item()
                unused_codes = torch.where(new_cluster_size < self.threshold_ema_dead_code)[0]
                n_dead = len(unused_codes)
                
                if n_dead > 0 and n_total > 0:
                    # 为死亡编码分配最常用编码的样本
                    most_used_codes = torch.topk(new_cluster_size, k=min(n_dead + 1, self._num_embeddings))[1]
                    for i, dead_idx in enumerate(unused_codes):
                        # 从最常用编码样本中重新初始化
                        if i < len(most_used_codes):
                            used_idx = most_used_codes[i]
                            used_samples = flat_input[encodings == used_idx]
                            if len(used_samples) > 0:
                                # 添加随机噪声以打破对称性
                                noise = torch.randn_like(used_samples[0]) * 0.1
                                self.embedding.weight.data[dead_idx] = used_samples[0] + noise
                                logging.debug(f"重新初始化死亡编码 {dead_idx}")
            
            # 归一化嵌入权重
            n = self._ema_cluster_size.sum()
            cluster_size = (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            
            # 使用EMA更新嵌入权重
            normalized_embeddings = self._ema_w / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(normalized_embeddings)
    
    def _track_codebook_usage(self, encodings):
        """跟踪编码本使用情况用于调试"""
        with torch.no_grad():
            unique_encodings = torch.unique(encodings)
            new_usage = torch.bincount(encodings, minlength=self._num_embeddings)
            self._codebook_usage += new_usage
            self._total_usage += len(encodings)

    def forward(self, inputs):
        """
        向量量化前向传播
        
        参数:
            inputs: 输入张量 [B, D] 或 [B, C, H, W]
            
        返回:
            quantized: 量化后的张量 (与输入相同形状)
            loss: VQ损失
            perplexity: 编码本熵
            encodings: 编码索引
            encoding_indices: 编码向量索引
        """
        # 处理输入
        flat_input, input_shape = self._tile_inputs(inputs)

        # 计算距离并获取最近编码向量
        distances = self._safe_distances(flat_input)
        encoding_indices = self._get_encoding_indices(distances)
        
        # 跟踪编码本使用情况
        self._track_codebook_usage(encoding_indices)
        
        # 量化查找
        quantized = self.embedding(encoding_indices).view(input_shape)
        
        # 训练期间更新编码本(如果使用EMA)
        if self.training and self.use_ema:
            self._update_ema(flat_input, encoding_indices)
        
        # 计算损失
        q_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='mean')
        e_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='mean')
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # 检查损失是否有异常值
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"VQ损失异常! q_loss={q_latent_loss.item()}, e_loss={e_latent_loss.item()}")
            loss = torch.tensor(0.1, device=inputs.device, requires_grad=True)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算使用编码向量的多样性
        avg_probs = torch.histc(encoding_indices.float(), 
                              bins=self._num_embeddings, 
                              min=0, 
                              max=self._num_embeddings-1) / encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 创建独热编码
        encodings = F.one_hot(encoding_indices, self._num_embeddings).float()
        
        # 恢复原始形状(如果需要)
        if len(inputs.shape) == 4:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
        return quantized, loss, perplexity, encodings, encoding_indices
        
    def get_codebook_stats(self):
        """返回编码本使用统计信息用于调试"""
        if self._total_usage == 0:
            usage_percent = torch.zeros_like(self._codebook_usage)
        else:
            usage_percent = self._codebook_usage.float() / self._total_usage.float() * 100
        
        used_codes = torch.sum(self._codebook_usage > 0).item()
        total_codes = self._num_embeddings
        
        stats = {
            'used_codes': used_codes,
            'total_codes': total_codes,
            'used_percent': used_codes / total_codes * 100,
            'code_usage': self._codebook_usage.cpu().numpy(),
            'code_usage_percent': usage_percent.cpu().numpy()
        }
        
        return stats