import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VectorQuantizer(nn.Module):
    """
    向量量化器 - 将连续向量映射到离散的编码本
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 初始化编码本 (codebook)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # 输入形状: [B, C, H, W]
        input_shape = inputs.shape
        
        # 扁平化空间维度以用于计算
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算与编码本中每个编码的欧氏距离
        distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_input, self.embedding.weight.t())
        
        # 找到最近的编码索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # 使用one-hot进行编码
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化: 从编码本中查找最近的编码
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)
        
        # 计算损失
        q_latent_loss = F.mse_loss(quantized.detach(), inputs)
        e_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器 (straight-through estimator)
        # 在前向传播中使用量化值，但在反向传播中使用原始输入的梯度
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[0], -1)

class DiscreteConv2d(nn.Module):
    """
    离散卷积层 - 使用量化权重和量化激活进行卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # 量化权重的编码本
        self.num_weight_levels = 16  # 权重的离散级别数
        
    def forward(self, x):
        # 量化权重（简化版，无需额外的VectorQuantizer）
        weight = self.conv.weight
        # 简单的权重离散化方法
        weight_abs_mean = torch.mean(torch.abs(weight))
        weight_scaled = weight / (weight_abs_mean * 2)  # 缩放到大约 [-0.5, 0.5] 范围
        weight_quantized = torch.round(weight_scaled * (self.num_weight_levels-1)) / (self.num_weight_levels-1)
        
        # 使用量化权重进行卷积
        out = F.conv2d(x, weight_quantized, self.conv.bias, 
                      self.conv.stride, self.conv.padding)
        
        # 应用阶跃函数进行激活离散化
        out_scaled = torch.sigmoid(out)
        discretized = torch.round(out_scaled * 9) / 9  # 离散到10个级别
        
        # 直通估计器
        result = out + (discretized - out_scaled).detach()
        
        return result

class DiscreteVAE(nn.Module):
    def __init__(self, grid_size=30, num_categories=10, hidden_dim=512, latent_dim=256, 
                codebook_size=512, embedding_dim=64, use_discrete_conv=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_categories = num_categories
        self.grid_cells = grid_size * grid_size
        self.input_dim = self.grid_cells * num_categories
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.use_discrete_conv = use_discrete_conv
        
        # 卷积编码器
        if use_discrete_conv:
            self.feature_extractor = nn.Sequential(
                DiscreteConv2d(num_categories, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                DiscreteConv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                DiscreteConv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(num_categories, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        
        # 全连接编码器部分
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * (grid_size // 8) * (grid_size // 8), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 潜变量参数
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 向量量化层
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_cost=0.25
        )
        
        # 解码器
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_cells * num_categories)
        )
        
    def encode(self, x):
        """
        将输入编码为潜在表示
        x形状: [batch_size, num_categories, grid_size, grid_size]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 全连接编码
        h = self.encoder_fc(features)
        
        # 生成潜变量参数
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """使用重参数化技巧从潜变量分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def quantize(self, z):
        """量化潜在表示"""
        # 调整形状以适应VectorQuantizer
        z_reshaped = z.unsqueeze(2).unsqueeze(3)  # [B, latent_dim, 1, 1]
        
        # 向量量化
        quantized, vq_loss, indices = self.vector_quantizer(z_reshaped)
        
        # 调整回原始形状
        quantized = quantized.squeeze(3).squeeze(2)  # [B, latent_dim]
        
        return quantized, vq_loss, indices
    
    def decode(self, z):
        """
        解码潜在表示
        z形状: [batch_size, latent_dim]
        """
        # 全连接解码
        output = self.decoder_fc(z)
        
        # 调整形状为[batch_size, grid_cells, num_categories]
        output = output.view(-1, self.grid_cells, self.num_categories)
        
        return output
    
    def forward(self, x):
        """
        完整前向传播
        x形状: [batch_size, num_categories, grid_size, grid_size]
        """
        # 编码到潜变量
        mu, logvar = self.encode(x)
        
        # 采样潜变量
        z = self.reparameterize(mu, logvar)
        
        # 量化潜变量（离散化）
        quantized, vq_loss, indices = self.quantize(z)
        
        # 解码
        reconstruction = self.decode(quantized)
        
        return reconstruction, mu, logvar, quantized, vq_loss, indices
    
    def reconstruct(self, x, num_steps=1):
        """
        重构输入，支持多步细化
        x形状: [batch_size, num_categories, grid_size, grid_size]
        """
        current_x = x
        
        for _ in range(num_steps):
            # 前向传播
            with torch.no_grad():
                recon, _, _, _, _, _ = self.forward(current_x)
                
                # 获取每个位置的最可能类别
                indices = torch.argmax(recon, dim=2)  # [batch_size, grid_cells]
                indices = indices.reshape(-1, self.grid_size, self.grid_size)
                
                # 创建新的one-hot输入
                new_x = torch.zeros_like(current_x)
                for b in range(indices.size(0)):
                    for i in range(self.grid_size):
                        for j in range(self.grid_size):
                            category = indices[b, i, j]
                            new_x[b, category, i, j] = 1.0
                
                current_x = new_x
        
        return indices