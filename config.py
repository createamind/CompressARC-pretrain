"""
ARC VQVAE 配置文件
"""

# 数据配置
DATA_CONFIG = {
    'data_path': 'data/training',  # ARC训练数据路径
    'max_grid_size': 30,           # 最大网格大小
    'pad_to_size': None,           # 设置为整数值以将所有网格填充到相同大小，None表示不填充
    'val_split': 0.1               # 验证集比例
}

# 模型配置
MODEL_CONFIG = {
    'num_categories': 10,          # ARC网格中的颜色类别数
    'latent_dim': 64,              # 潜在表示维度
    'num_embeddings': 512,         # VQ编码本大小
    'commitment_cost': 0.25,       # VQ承诺损失权重
    'decay': 0.99,                 # EMA更新率
    'hidden_dims': [32, 64],       # 编码器/解码器中的隐藏维度
    'use_ema': True                # 是否使用EMA更新编码本
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 64,              # 批量大小
    'num_workers': 4,              # 数据加载线程数
    'learning_rate': 3e-4,         # 学习率
    'weight_decay': 1e-6,          # 权重衰减率
    'epochs': 100,                 # 训练轮次
    'save_interval': 5,            # 每隔多少轮保存一次模型
    'eval_interval': 1,            # 每隔多少轮评估一次
    'vis_interval': 5,             # 每隔多少轮可视化一次
    'lr_scheduler': 'cosine',      # 学习率调度器类型: 'step', 'cosine', 'plateau'
    'warmup_epochs': 5,            # 学习率预热轮次
    'grad_clip': 1.0,              # 梯度裁剪阈值
}

# 输出配置
OUTPUT_CONFIG = {
    'checkpoint_dir': 'checkpoints',  # 检查点保存目录
    'log_dir': 'logs',                # 日志保存目录
    'vis_dir': 'visualizations',      # 可视化保存目录
}

# 调试配置
DEBUG_CONFIG = {
    'log_level': 'INFO',              # 日志级别: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'debug_batches': 5,               # 调试模式下处理的批次数
    'debug_mode': False,              # 是否启用调试模式
    'profile': False,                 # 是否启用性能分析
}