import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ARCTaskDataset(Dataset):
    """ARC任务数据集，以任务为单位加载"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.tasks = []
        self.load_data()

    def load_data(self):
        print(f"Loading ARC tasks from {self.data_path}...")
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    task = {
                        'id': filename,
                        'train': [],
                        'test': {}
                    }

                    # 处理训练样例
                    for train_ex in task_data['train']:
                        input_grid = self._preprocess_grid(train_ex['input'])
                        output_grid = self._preprocess_grid(train_ex['output'])

                        # 转换为one-hot编码
                        input_onehot = self._to_onehot(input_grid)
                        output_onehot = self._to_onehot(output_grid)

                        task['train'].append((input_onehot, output_onehot))

                    # 处理测试样例
                    test_input = self._preprocess_grid(task_data['test'][0]['input'])
                    test_output = self._preprocess_grid(task_data['test'][0]['output'])
                    task['test']['input'] = self._to_onehot(test_input)
                    task['test']['output'] = self._to_onehot(test_output)

                    self.tasks.append(task)

        print(f"Loaded {len(self.tasks)} ARC tasks")

    def _preprocess_grid(self, grid):
        """预处理网格：统一格式"""
        grid_np = np.array(grid)
        return torch.tensor(grid_np, dtype=torch.long)

    def _to_onehot(self, grid):
        """转换网格为one-hot编码"""
        h, w = grid.shape
        one_hot = torch.zeros(10, 30, 30)  # 固定大小，10个类别

        # 复制数据到fixed-size数组
        for i in range(min(h, 30)):
            for j in range(min(w, 30)):
                one_hot[grid[i, j], i, j] = 1

        return one_hot

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def collate_arc_tasks(batch):
    """自定义collate函数，处理不同大小的任务"""
    # batch是一个包含多个任务的列表
    return batch  # 简单返回任务列表，不做额外处理

def get_arc_dataset(data_path):
    """只加载数据集，不创建DataLoader"""
    dataset = ARCTaskDataset(data_path)
    return dataset

def get_arc_dataloader(data_path, batch_size=4, shuffle=True, num_workers=2):
    """创建ARC任务数据加载器"""
    dataset = ARCTaskDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_arc_tasks
    )
    return loader, dataset