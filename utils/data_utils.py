import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging


class ARCGridDataset(Dataset):
    """ARC网格数据集 - 提取所有任务中的网格进行重建训练"""
    def __init__(self, data_path, transform=None, max_grid_size=30, pad_to_size=None):
        self.data_path = data_path
        self.transform = transform
        self.max_grid_size = max_grid_size
        self.pad_to_size = pad_to_size
        
        # 加载所有网格
        self.grids = []
        self.grid_sizes = []
        self.task_ids = []
        self.grid_types = []
        
        logging.info(f"加载ARC任务数据从: {data_path}")
        self._load_data()
        logging.info(f"共加载 {len(self.grids)} 个网格, 大小范围: {min(self.grid_sizes)}-{max(self.grid_sizes)}")
    
    def _load_data(self):
        """从ARC JSON文件加载网格数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
        
        # 如果是目录则加载所有JSON文件
        if os.path.isdir(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith('.json'):
                    self._load_task_file(os.path.join(self.data_path, filename))
        else:
            # 否则加载单个JSON文件
            self._load_task_file(self.data_path)
    
    def _load_task_file(self, json_path):
        """加载单个ARC任务文件"""
        try:
            with open(json_path, 'r') as f:
                task = json.load(f)
            
            task_id = os.path.basename(json_path).replace('.json', '')
            
            # 加载训练示例
            for i, train in enumerate(task.get('train', [])):
                # 输入网格
                input_grid = np.array(train['input'])
                if max(input_grid.shape) <= self.max_grid_size:
                    self.grids.append(input_grid)
                    self.grid_sizes.append(input_grid.shape)
                    self.task_ids.append(f"{task_id}_train{i}_input")
                    self.grid_types.append("train_input")
                
                # 输出网格
                output_grid = np.array(train['output'])
                if max(output_grid.shape) <= self.max_grid_size:
                    self.grids.append(output_grid)
                    self.grid_sizes.append(output_grid.shape)
                    self.task_ids.append(f"{task_id}_train{i}_output")
                    self.grid_types.append("train_output")
            
            # 加载测试示例
            if 'test' in task:
                for i, test in enumerate(task['test']):
                    # 输入网格
                    input_grid = np.array(test['input'])
                    if max(input_grid.shape) <= self.max_grid_size:
                        self.grids.append(input_grid)
                        self.grid_sizes.append(input_grid.shape)
                        self.task_ids.append(f"{task_id}_test{i}_input")
                        self.grid_types.append("test_input")
                    
                    # 输出网格
                    if 'output' in test:
                        output_grid = np.array(test['output'])
                        if max(output_grid.shape) <= self.max_grid_size:
                            self.grids.append(output_grid)
                            self.grid_sizes.append(output_grid.shape)
                            self.task_ids.append(f"{task_id}_test{i}_output")
                            self.grid_types.append("test_output")
        
        except Exception as e:
            logging.error(f"加载任务文件出错 {json_path}: {e}")
    
    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        """获取单个网格"""
        grid = self.grids[idx]
        grid_size = self.grid_sizes[idx]
        task_id = self.task_ids[idx]
        grid_type = self.grid_types[idx]
        
        # 转换为张量
        grid_tensor = torch.tensor(grid, dtype=torch.long)
        
        # 填充到固定大小(如果需要)
        if self.pad_to_size is not None:
            h, w = grid.shape
            pad_h = max(0, self.pad_to_size - h)
            pad_w = max(0, self.pad_to_size - w)
            if pad_h > 0 or pad_w > 0:
                # 用0填充(ARC中0是背景色)
                grid_tensor = F.pad(grid_tensor, (0, pad_w, 0, pad_h), "constant", 0)
        
        # 应用转换
        if self.transform:
            grid_tensor = self.transform(grid_tensor)
        
        # 返回带元数据的样本
        return {
            'grid': grid_tensor, 
            'original_shape': grid_size,
            'task_id': task_id,
            'grid_type': grid_type
        }


def create_arc_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, 
                          max_grid_size=30, pad_to_size=None, val_split=0.1):
    """创建ARC网格数据加载器"""
    # 创建数据集
    dataset = ARCGridDataset(
        data_path, 
        max_grid_size=max_grid_size,
        pad_to_size=pad_to_size
    )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)
    
    val_size = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=True  # 加速CPU->GPU传输
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset