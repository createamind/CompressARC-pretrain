import torch
import torch.nn as nn
import numpy as np

class ExplicitRuleInference(nn.Module):
    """明确的规则推理机制，增强RuleGuidedVAE"""
    def __init__(self, operation_library, rule_dim=128, device='cuda'):
        super().__init__()
        self.operation_library = operation_library
        self.rule_dim = rule_dim
        self.device = device
        
        # 规则解析器 - 将潜在规则表示转换为明确操作序列
        self.rule_parser = nn.Sequential(
            nn.Linear(rule_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # 操作类型和参数预测器
        self.op_predictor = nn.Linear(512, len(operation_library.operations))
        self.param_predictor = nn.ModuleDict({
            op_name: self._create_param_predictor(op.get_parameter_space())
            for op_name, op in operation_library.operations.items()
        })
        
        # 规则验证器 - 评估操作序列的有效性
        self.rule_validator = nn.Sequential(
            nn.Linear(rule_dim + 128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def _create_param_predictor(self, param_space):
        """为特定参数空间创建参数预测器"""
        output_size = 0
        for param, spec in param_space.items():
            if spec['type'] == 'int':
                output_size += 2  # 范围的最小值和最大值
            elif spec['type'] == 'categorical':
                output_size += len(spec['values'])
            elif spec['type'] == 'tensor':
                # 张量参数需要专门处理
                continue
                
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    
    def parse_rule(self, rule_embedding):
        """将规则嵌入解析为显式操作序列"""
        features = self.rule_parser(rule_embedding)
        
        # 预测操作类型
        op_logits = self.op_predictor(features)
        op_probs = torch.softmax(op_logits, dim=-1)
        
        # 获取最可能的操作
        top_k = 3  # 考虑前k个最可能的操作
        top_ops, top_indices = torch.topk(op_probs, k=top_k, dim=-1)
        
        operations = []
        op_names = list(self.operation_library.operations.keys())
        
        for k in range(top_k):
            if top_ops[0, k] < 0.1:  # 概率阈值
                continue
                
            op_idx = top_indices[0, k].item()
            op_name = op_names[op_idx]
            
            # 为此操作预测参数
            param_predictor = self.param_predictor[op_name]
            param_values = param_predictor(features)
            
            # 解析参数
            param_space = self.operation_library.operations[op_name].get_parameter_space()
            parsed_params = self._parse_parameters(param_values, param_space)
            
            operations.append({
                'type': op_name,
                'params': parsed_params,
                'confidence': top_ops[0, k].item()
            })
        
        return operations
    
    def _parse_parameters(self, param_values, param_space):
        """将神经网络输出解析为操作参数"""
        parsed = {}
        index = 0
        
        for param_name, spec in param_space.items():
            if spec['type'] == 'int':
                # 预测范围内的整数
                min_val, max_val = spec['range']
                range_size = max_val - min_val + 1
                
                # 使用softmax获取离散概率
                logits = param_values[0, index:index+range_size]
                probs = torch.softmax(logits, dim=-1)
                value = min_val + torch.argmax(probs).item()
                
                parsed[param_name] = value
                index += range_size
            
            elif spec['type'] == 'categorical':
                # 选择类别
                num_options = len(spec['values'])
                logits = param_values[0, index:index+num_options]
                probs = torch.softmax(logits, dim=-1)
                option_idx = torch.argmax(probs).item()
                
                parsed[param_name] = spec['values'][option_idx]
                index += num_options
                
        return parsed
    
    def apply_explicit_rules(self, input_grid, operations):
        """显式应用操作序列到输入网格"""
        result = input_grid.clone()
        
        for op in operations:
            op_type = op['type']
            params = op['params']
            
            if op_type in self.operation_library.operations:
                # 应用操作
                result = self.operation_library.operations[op_type](result, params)
            
        return result
    
    def validate_rule(self, rule_embedding, input_grid, output_grid):
        """验证规则对于给定的输入-输出对的有效性"""
        # 提取输入和输出的特征
        batch_size = input_grid.size(0)
        input_features = torch.mean(input_grid.view(batch_size, -1), dim=1)
        output_features = torch.mean(output_grid.view(batch_size, -1), dim=1)
        
        # 计算输入和输出的差异
        diff_features = output_features - input_features
        
        # 评估规则与差异的匹配度
        validation_input = torch.cat([rule_embedding, diff_features], dim=1)
        validity_score = self.rule_validator(validation_input)
        
        return validity_score