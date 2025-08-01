import torch
import types

def fix_decoder_output_format(model):
    """修复解码器的输出格式，恢复原始行为"""
    print("应用解码器输出格式修复...")

    # 保存原始前向传播方法
    original_forward = model.decoder.forward

    # 定义新的前向传播方法，恢复正确的输出格式
    def fixed_forward(self, pixel_features, object_features, relation_features, high_features):
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
        pixel_upsampled = torch.nn.functional.interpolate(pixel_features,
                                      size=(self.level1_size, self.level1_size),
                                      mode='bilinear',
                                      align_corners=False)

        # 7. 合并低级特征和像素特征
        low_with_pixels = torch.cat([low_decoded, pixel_upsampled], dim=1)

        # 8. 最终解码
        output = self.final_decoder(low_with_pixels)

        # 9. 裁剪到网格大小
        output = output[:, :, :self.grid_size, :self.grid_size]

        # 10. 关键修复：恢复原始输出格式转换
        # 将 [batch, categories, H, W] 转换为 [batch, H*W, categories]
        output = output.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_categories)

        return output

    # 替换解码器的前向传播方法
    model.decoder.forward = types.MethodType(fixed_forward, model.decoder)
    
    # 修复应用规则函数以处理不同尺寸
    original_apply_rule = model.apply_rule
    
    def fixed_apply_rule(self, input_grid, rule):
        # 获取原始规则应用结果
        result = original_apply_rule(self, input_grid, rule)
        
        # 如果输出形状不匹配输入，进行修复
        if isinstance(result, torch.Tensor):
            input_shape = input_grid.shape
            if len(input_shape) >= 4 and len(result.shape) >= 3:
                # 确保输出与输入具有相同的空间尺寸
                input_h, input_w = input_shape[2:4]
                
                # 如果是 [batch, pixels, categories] 格式
                if result.dim() == 3 and result.shape[1] == self.grid_size * self.grid_size:
                    # 重塑为空间格式，裁剪，再展平
                    result = result.reshape(result.shape[0], self.grid_size, self.grid_size, -1)
                    result = result[:, :input_h, :input_w, :]  # 裁剪
                    result = result.reshape(result.shape[0], input_h * input_w, -1)  # 展平
                
        return result
    
    # 替换应用规则方法
    model.apply_rule = types.MethodType(fixed_apply_rule, model)
    
    # 修复可视化函数以正确处理模型输出
    def fix_visualization(task_id, input_grid, target_grid, predicted_grid, save_path):
        """修复后的可视化函数，正确处理模型输出格式"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 转换输入和输出的one-hot为颜色网格
        input_colors = torch.argmax(input_grid, dim=0).cpu().numpy()
        output_colors = torch.argmax(target_grid, dim=0).cpu().numpy()
        
        # 获取真实网格尺寸（非填充尺寸）
        h, w = input_colors.shape
        
        # 处理预测输出的不同格式
        if isinstance(predicted_grid, torch.Tensor):
            if predicted_grid.dim() == 3 and predicted_grid.shape[0] == 10:
                # 已经是正确格式 [C, H, W]
                pred_colors = torch.argmax(predicted_grid, dim=0).cpu().numpy()
            elif predicted_grid.dim() == 2:
                # 如果是 [H*W, C] 或其他2D格式
                if predicted_grid.shape[1] == 10:
                    pred_colors = torch.argmax(predicted_grid, dim=1).reshape(h, w).cpu().numpy()
                else:
                    # 尝试自适应识别格式
                    pred_colors = predicted_grid.cpu().numpy()
            else:
                # 其他格式，尝试适配
                pred_colors = torch.argmax(predicted_grid.view(-1, 10), dim=1).reshape(h, w).cpu().numpy()
        else:
            # 非张量，可能是NumPy数组
            pred_colors = np.array([[0]])
            
        # 确保裁剪到正确尺寸
        if pred_colors.shape[0] > h or pred_colors.shape[1] > w:
            pred_colors = pred_colors[:h, :w]
        
        # 绘制网格
        axes[0].imshow(input_colors, vmin=0, vmax=9)
        axes[0].set_title('Input')
        axes[0].grid(True, color='black', linewidth=0.5)
        
        axes[1].imshow(output_colors, vmin=0, vmax=9)
        axes[1].set_title('Expected Output')
        axes[1].grid(True, color='black', linewidth=0.5)
        
        axes[2].imshow(pred_colors, vmin=0, vmax=9)
        axes[2].set_title(f'Predicted Output {pred_colors.shape}')
        axes[2].grid(True, color='black', linewidth=0.5)
        
        plt.suptitle(f'Task: {task_id}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    
    print("解码器输出格式修复完成！")
    return fix_visualization