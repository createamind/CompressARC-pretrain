import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from models.vector_quantizer import VectorQuantizer


class ResidualBlock(nn.Module):
    """残差块 - 提高模型深度而不增加梯度问题"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = in_channels == out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if not self.same_channels:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.same_channels:
            skip = x
        else:
            skip = self.conv_skip(x)

        x = self.conv_block(x)
        return F.relu(x + skip, inplace=True)


class Encoder(nn.Module):
    """用于小型ARC网格的高效编码器"""
    def __init__(self, in_channels=10, latent_dim=64, hidden_dims=[32, 64]):
        super().__init__()

        modules = []

        # 初始卷积层
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )

        # 添加残差块并逐渐增加通道数
        for i in range(len(hidden_dims) - 1):
            modules.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1])
            )

        # 最终卷积层以生成latent表示
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1),
                nn.BatchNorm2d(latent_dim)
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """用于小型ARC网格的高效解码器"""
    def __init__(self, out_channels=10, latent_dim=64, hidden_dims=[64, 32]):
        super().__init__()

        modules = []

        # 初始卷积层
        modules.append(
            nn.Sequential(
                nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )

        # 添加残差块并逐渐减少通道数
        for i in range(len(hidden_dims) - 1):
            modules.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1])
            )

        # 最终卷积层以生成像素类别logits
        modules.append(
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1)
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)


class ArcGridVQVAE(nn.Module):
    """
    ARC任务的网格重建VQVAE模型

    特点:
    - 支持不同大小的网格
    - 离散向量量化
    - 支持10类颜色像素
    - 为后续对象识别和规则提取提供基础
    """
    def __init__(
        self,
        num_categories=10,
        latent_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        hidden_dims=[32, 64],
        use_ema=True
    ):
        super().__init__()

        self.num_categories = num_categories
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = Encoder(
            in_channels=num_categories,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )

        # 向量量化器
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            use_ema=use_ema,
            threshold_ema_dead_code=2  # 激活死码检测
        )

        # 解码器
        self.decoder = Decoder(
            out_channels=num_categories,
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims))
        )

    def encode(self, x):
        """编码输入网格"""
        # 确保输入是one-hot编码
        if x.dtype == torch.long:
            # 将类别索引转换为one-hot
            x_onehot = F.one_hot(x, self.num_categories).float()
            # 调整维度顺序为[B, C, H, W]
            if len(x.shape) == 3:  # [B, H, W]
                x_onehot = x_onehot.permute(0, 3, 1, 2)
            else:  # [H, W]
                x_onehot = x_onehot.permute(2, 0, 1).unsqueeze(0)
        else:
            # 假设已经是one-hot编码形式[B, C, H, W]
            x_onehot = x

        # 编码
        encoded = self.encoder(x_onehot)
        return encoded

    def decode(self, z):
        """解码潜在表示"""
        # 解码为类别logits
        decoded = self.decoder(z)
        return decoded

    def quantize(self, encoded):
        """向量量化编码后的表示"""
        quantized, vq_loss, perplexity, encodings, encoding_indices = self.vector_quantizer(encoded)
        return quantized, vq_loss, perplexity, encodings, encoding_indices

    def forward(self, x):
        """完整的前向传播"""
        # 编码
        encoded = self.encode(x)

        # 量化
        quantized, vq_loss, perplexity, encodings, encoding_indices = self.quantize(encoded)

        # 解码
        decoded = self.decode(quantized)

        return decoded, vq_loss, perplexity

    # 修改ArcGridVQVAE的forward方法
    def forward(self, x, mask=None):
        """
        完整的前向传播，考虑填充掩码

        参数:
            x: 输入网格 [B, H, W] 或 [B, C, H, W]
            mask: 可选，表示哪些部分是原始数据(True)，哪些是填充(False)
        """
        # 编码
        encoded = self.encode(x)

        # 量化
        quantized, vq_loss, perplexity, encodings, encoding_indices = self.quantize(encoded)

        # 解码
        decoded = self.decode(quantized)

        # 如果提供了掩码，只计算原始数据部分的损失
        if mask is not None and mask.dim() == 3:  # [B, H, W]
            # 调整掩码形状以匹配解码输出
            if decoded.dim() == 4:  # [B, C, H, W]
                # 扩展掩码以匹配通道维度
                mask = mask.unsqueeze(1).expand(-1, decoded.size(1), -1, -1)

        return decoded, vq_loss, perplexity

    def reconstruct(self, x):
        """重建输入网格(不计算损失)"""
        with torch.no_grad():
            # 编码
            encoded = self.encode(x)

            # 量化
            quantized, _, _, _, _ = self.quantize(encoded)

            # 解码
            decoded = self.decode(quantized)

            # 返回类别预测
            if x.dtype == torch.long:
                # 返回类别索引
                return torch.argmax(decoded, dim=1)
            else:
                # 返回logits
                return decoded

    def get_codebook_usage(self):
        """获取编码本使用统计"""
        return self.vector_quantizer.get_codebook_stats()