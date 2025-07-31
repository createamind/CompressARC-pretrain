import torch

def handle_tuple_output(model_output):
    """
    处理模型返回的元组输出，提取预测结果
    
    Args:
        model_output: 模型返回的元组
        
    Returns:
        预测输出张量，形状应为 [batch_size, num_categories, height, width]
    """
    print(f"模型返回了元组，长度为 {len(model_output)}")
    
    # 打印元组中每个元素的信息，以帮助识别
    for i, item in enumerate(model_output):
        if isinstance(item, torch.Tensor):
            print(f"  元素 {i}: 张量，形状 = {item.shape}")
        else:
            print(f"  元素 {i}: {type(item).__name__}")
    
    # 尝试找到正确的预测输出
    # 通常第一个元素是主要预测结果
    prediction = None
    
    # 对每个张量元素应用启发式规则找出预测结果
    for item in model_output:
        if not isinstance(item, torch.Tensor):
            continue
            
        # 如果是4D张量且第二维是10（类别数），这很可能是我们要找的
        if item.dim() == 4 and item.shape[1] == 10:
            prediction = item
            print(f"找到可能的预测张量，形状: {item.shape}")
            break
    
    # 如果找不到合适的张量，使用第一个张量
    if prediction is None:
        for item in model_output:
            if isinstance(item, torch.Tensor):
                prediction = item
                print(f"使用第一个可用张量作为预测，形状: {item.shape}")
                break
    
    # 如果仍然没有找到任何张量，创建一个空白预测
    if prediction is None:
        print("警告: 未找到有效的预测张量，使用零张量代替")
        prediction = torch.zeros(1, 10, 30, 30)
    
    return prediction
