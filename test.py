import torch
import numpy as np
import json
import os
import argparse
import time
import itertools
from model import DiscreteVAE
from discrete_vae import DiscreteVAE as FullDiscreteVAE
from torch.cuda.amp import autocast
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

def check_gpu(gpu_id=None):
    """检查并详细报告GPU状态"""
    if not torch.cuda.is_available():
        print("警告: 未检测到CUDA，将使用CPU进行推理 (速度会非常慢)")
        return False, "cpu", None
    
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("警告: 虽然CUDA可用，但未找到可用的GPU设备，将使用CPU推理")
        return False, "cpu", None
    
    # 打印所有可用GPU的详细信息
    print(f"检测到 {device_count} 个GPU:")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        print(f"  GPU {i}: {gpu_name} ({total_memory:.2f} GB)")
    
    # 如果指定了GPU ID，优先使用指定的GPU
    if gpu_id is not None:
        if gpu_id >= device_count:
            print(f"警告: 指定的GPU ID {gpu_id} 超出可用GPU数量范围，将使用默认GPU")
            device_id = 0
        else:
            device_id = gpu_id
            print(f"使用指定的GPU {device_id}")
    else:
        # 默认使用第一个GPU
        device_id = 0
    
    device = f"cuda:{device_id}"
    
    # 测试内存分配
    try:
        test_tensor = torch.zeros((100, 100), device=device)
        del test_tensor
        print(f"成功在 {device} 上分配测试张量")
    except Exception as e:
        print(f"警告: GPU内存分配测试失败: {e}")
        print("将尝试继续使用GPU，但可能会遇到问题")
    
    return True, device, device_id

def load_model(model_path, device='cuda', use_discrete_vae=True, 
              codebook_size=512, embedding_dim=64, use_residual=True):
    """Load trained model from checkpoint"""
    if use_discrete_vae:
        # 加载完全离散VAE
        model = FullDiscreteVAE(
            grid_size=30,
            num_categories=10,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
    else:
        # 加载标准VAE
        model = DiscreteVAE(use_residual=use_residual)
    
    # 处理状态字典格式
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 打印GPU内存使用情况
    if device.startswith("cuda"):
        gpu_id = int(device.split(":")[-1])
        allocated_mem = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
        print(f"模型已加载到GPU，内存使用: {allocated_mem:.2f}GB / {total_mem:.2f}GB")
    
    return model

def preprocess_grid(grid):
    """Convert grid to one-hot encoding for model input"""
    h, w = grid.shape
    one_hot = np.zeros((10, 30, 30))  # Fixed size for all grids
    
    # Copy original grid data to fixed-size array
    for i in range(min(h, 30)):
        for j in range(min(w, 30)):
            one_hot[grid[i, j], i, j] = 1
    
    # Convert to tensor and add batch dimension
    return torch.tensor(one_hot, dtype=torch.float).unsqueeze(0)

def postprocess_grid(output, original_shape):
    """Extract grid of original shape from model output"""
    h, w = original_shape
    h = min(h, 30)
    w = min(w, 30)
    
    # Output shape is [batch_size, grid_size, grid_size]
    output = output.squeeze().cpu().numpy()
    
    # Extract grid of original shape
    return output[:h, :w]

def ensemble_prediction(model, input_tensor, num_steps_list=[1, 3, 5], device='cuda', use_discrete_vae=True):
    """Generate ensemble predictions with different numbers of inference steps"""
    with torch.no_grad():
        predictions = []
        
        for steps in num_steps_list:
            output = model.reconstruct(input_tensor.to(device), num_steps=steps)
            predictions.append(output)
        
        # Stack predictions and take mode (most common value) at each position
        stacked = torch.stack(predictions)
        voted, _ = torch.mode(stacked, dim=0)
        
        return voted

def test_time_train(model, train_examples, epochs=10, lr=0.0005, device='cuda', use_amp=True, use_discrete_vae=True,
                   codebook_size=512, embedding_dim=64):
    """Fine-tune model on task-specific examples with improved training"""
    # Create a copy of the model for fine-tuning
    if use_discrete_vae:
        fine_tuned_model = FullDiscreteVAE(
            grid_size=30,
            num_categories=10,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
    else:
        fine_tuned_model = DiscreteVAE(use_residual=True)
    
    fine_tuned_model.load_state_dict({k: v.cpu().clone() for k, v in model.state_dict().items()})
    fine_tuned_model.to(device)
    fine_tuned_model.train()
    
    # Prepare optimizer with weight decay
    optimizer = torch.optim.AdamW(fine_tuned_model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Process training examples
    for epoch in range(epochs):
        epoch_loss = 0
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Preprocess input and output
            input_tensor = preprocess_grid(input_grid).to(device)
            target_tensor = torch.tensor(output_grid, dtype=torch.long).to(device)
            
            # Create target grid (padded to 30x30)
            h, w = output_grid.shape
            padded_target = torch.zeros((30, 30), dtype=torch.long, device=device)
            padded_target[:min(h, 30), :min(w, 30)] = target_tensor[:min(h, 30), :min(w, 30)]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if use_amp:
                with autocast():
                    if use_discrete_vae:
                        recon_batch, mu, logvar, quantized, vq_loss, indices = fine_tuned_model(input_tensor)
                        
                        # 计算自定义损失，专注于从输入预测输出
                        flat_target = padded_target.reshape(-1)
                        recon_loss = 0
                        for i in range(900):  # 30x30网格
                            recon_loss += F.cross_entropy(recon_batch[0, i, :].unsqueeze(0), flat_target[i].unsqueeze(0))
                        recon_loss = recon_loss / 900
                        
                        # 在测试时训练中也包含VQ损失
                        loss = recon_loss + 0.1 * vq_loss
                    else:
                        recon_batch, mu, logvar = fine_tuned_model(input_tensor)
                        
                        # 扁平化目标并计算损失
                        flat_target = padded_target.reshape(-1)
                        recon_loss = 0
                        for i in range(900):  # 30x30网格
                            recon_loss += F.cross_entropy(recon_batch[0, i, :].unsqueeze(0), flat_target[i].unsqueeze(0))
                        recon_loss = recon_loss / 900
                        
                        # 测试时训练中不使用KL项
                        loss = recon_loss
                
                # 使用缩放器的反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fine_tuned_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准前向传播
                if use_discrete_vae:
                    recon_batch, mu, logvar, quantized, vq_loss, indices = fine_tuned_model(input_tensor)
                    
                    # 计算自定义损失
                    flat_target = padded_target.reshape(-1)
                    recon_loss = 0
                    for i in range(900):  # 30x30网格
                        recon_loss += F.cross_entropy(recon_batch[0, i, :].unsqueeze(0), flat_target[i].unsqueeze(0))
                    recon_loss = recon_loss / 900
                    
                    # 包含VQ损失
                    loss = recon_loss + 0.1 * vq_loss
                else:
                    recon_batch, mu, logvar = fine_tuned_model(input_tensor)
                    
                    # 计算自定义损失
                    flat_target = padded_target.reshape(-1)
                    recon_loss = 0
                    for i in range(900):  # 30x30网格
                        recon_loss += F.cross_entropy(recon_batch[0, i, :].unsqueeze(0), flat_target[i].unsqueeze(0))
                    recon_loss = recon_loss / 900
                    
                    # 不使用KL项
                    loss = recon_loss
                
                # 标准反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fine_tuned_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
        
        if epochs > 1:  # 只在多个轮次时打印进度
            print(f"测试时训练 epoch {epoch+1}/{epochs}, 损失: {epoch_loss:.4f}")
    
    fine_tuned_model.eval()
    return fine_tuned_model

def evaluate_task(model, task_path, num_inference_steps=5, test_time_epochs=10, device='cuda', 
                 use_amp=True, use_discrete_vae=True, codebook_size=512, embedding_dim=64):
    """Evaluate model on a single ARC task with test-time training and ensemble prediction"""
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test-time training on this specific task if there are training examples
    if task['train'] and test_time_epochs > 0:
        print(f"对 {len(task['train'])} 个示例进行测试时训练...")
        model = test_time_train(
            model, task['train'], 
            epochs=test_time_epochs, 
            device=device,
            use_amp=use_amp,
            use_discrete_vae=use_discrete_vae,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
    
    # Evaluate on test examples
    correct = 0
    total = len(task['test'])
    
    # 存储任务级别的像素准确率数据
    pixel_accuracies = []
    
    for test_example in task['test']:
        input_grid = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        
        # Process input
        input_tensor = preprocess_grid(input_grid).to(device)
        
        # Use ensemble prediction for better results
        step_options = [1, num_inference_steps, num_inference_steps * 2]
        with torch.no_grad():
            output_grid = ensemble_prediction(
                model, 
                input_tensor, 
                num_steps_list=step_options, 
                device=device,
                use_discrete_vae=use_discrete_vae
            )
        
        # Post-process output
        predicted_output = postprocess_grid(output_grid, expected_output.shape)
        
        # Check if output matches expected
        is_correct = np.array_equal(predicted_output, expected_output)
        if is_correct:
            correct += 1
        
        # 计算像素级准确率
        total_pixels = expected_output.size
        correct_pixels = np.sum(predicted_output == expected_output)
        pixel_accuracy = correct_pixels / total_pixels
        pixel_accuracies.append(pixel_accuracy)
    
    task_accuracy = correct / total if total > 0 else 0
    avg_pixel_accuracy = np.mean(pixel_accuracies) if pixel_accuracies else 0
    
    return task_accuracy, avg_pixel_accuracy

def evaluate_all_tasks(model_path, data_dir, num_inference_steps=5, test_time_epochs=10, 
                      use_amp=True, use_discrete_vae=True, codebook_size=512, embedding_dim=64,
                      gpu_id=None):
    """Evaluate model on all tasks in a directory"""
    # 检查GPU状态
    has_gpu, device_str, detected_gpu_id = check_gpu(gpu_id)
    device = torch.device(device_str)
    
    # 只有在GPU上才使用混合精度
    use_amp = use_amp and has_gpu
    
    # 加载模型
    model = load_model(
        model_path, 
        device, 
        use_discrete_vae=use_discrete_vae,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim
    )
    
    total_tasks = 0
    total_correct = 0
    all_pixel_accuracies = []
    task_results = {}
    
    # 创建结果目录
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.dirname(model_path)
    results_dir = os.path.join(model_dir, f"evaluation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    
    # 获取所有任务文件
    task_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    for filename in tqdm(task_files, desc="评估任务"):
        task_path = os.path.join(data_dir, filename)
        task_start_time = time.time()
        
        task_accuracy, pixel_accuracy = evaluate_task(
            model, task_path, 
            num_inference_steps=num_inference_steps,
            test_time_epochs=test_time_epochs,
            device=device,
            use_amp=use_amp,
            use_discrete_vae=use_discrete_vae,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim
        )
        
        task_time = time.time() - task_start_time
        all_pixel_accuracies.append(pixel_accuracy)
        
        task_results[filename] = {
            "task_accuracy": task_accuracy,
            "pixel_accuracy": pixel_accuracy,
            "eval_time": task_time
        }
        
        print(f"任务 {filename}: 准确率={task_accuracy:.2f}, 像素准确率={pixel_accuracy:.4f}, 耗时={task_time:.2f}s")
        
        total_tasks += 1
        total_correct += task_accuracy
    
    overall_accuracy = total_correct / total_tasks if total_tasks > 0 else 0
    avg_pixel_accuracy = np.mean(all_pixel_accuracies) if all_pixel_accuracies else 0
    total_time = time.time() - start_time
    
    print(f"\n{total_tasks}个任务的总体准确率: {overall_accuracy:.4f}")
    print(f"平均像素准确率: {avg_pixel_accuracy:.4f}")
    print(f"总评估时间: {total_time:.2f}s, 每任务平均: {total_time/total_tasks:.2f}s")
    
    # 保存结果到文件
    model_type = "discrete" if use_discrete_vae else "standard"
    results_path = os.path.join(results_dir, f'{model_type}_eval_steps{num_inference_steps}_tt{test_time_epochs}.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model_path': model_path,
            'model_type': model_type,
            'overall_accuracy': float(overall_accuracy),
            'average_pixel_accuracy': float(avg_pixel_accuracy),
            'total_tasks': total_tasks,
            'inference_steps': num_inference_steps,
            'test_time_epochs': test_time_epochs,
            'codebook_size': codebook_size if use_discrete_vae else None,
            'embedding_dim': embedding_dim if use_discrete_vae else None,
            'task_results': task_results,
            'evaluation_time': total_time,
            'device': device_str,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"结果已保存到 {results_path}")
    
    return overall_accuracy, avg_pixel_accuracy, task_results

def parameter_scan(model_path, data_dir, use_discrete_vae=True, codebook_size=512, embedding_dim=64, gpu_id=None):
    """
    进行参数扫描实验，测试不同的inference_steps和test_time_epochs组合
    """
    # 检查GPU状态
    has_gpu, device_str, detected_gpu_id = check_gpu(gpu_id)
    
    # 创建实验结果目录
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.dirname(model_path)
    scan_dir = os.path.join(model_dir, f"param_scan_{timestamp}")
    os.makedirs(scan_dir, exist_ok=True)
    
    # 定义要尝试的参数
    steps_options = [1, 3, 5, 10]
    tt_epochs_options = [0, 5, 10, 20]
    
    # 准备结果收集
    results = []
    
    print(f"开始参数扫描: inference_steps {steps_options}, test_time_epochs {tt_epochs_options}")
    print(f"使用设备: {device_str}")
    
    # 遍历所有参数组合
    for steps, tt_epochs in itertools.product(steps_options, tt_epochs_options):
        print(f"\n===== 测试参数: steps={steps}, tt_epochs={tt_epochs} =====")
        
        # 评估当前参数组合
        overall_acc, pixel_acc, task_results = evaluate_all_tasks(
            model_path,
            data_dir,
            num_inference_steps=steps,
            test_time_epochs=tt_epochs,
            use_amp=has_gpu,  # 只在GPU上使用AMP
            use_discrete_vae=use_discrete_vae,
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            gpu_id=gpu_id
        )
        
        # 记录结果
        results.append({
            'inference_steps': steps,
            'test_time_epochs': tt_epochs,
            'overall_accuracy': float(overall_acc),
            'pixel_accuracy': float(pixel_acc),
            'task_results': task_results
        })
        
        # 保存当前结果
        with open(os.path.join(scan_dir, f'scan_steps{steps}_tt{tt_epochs}.json'), 'w') as f:
            json.dump(results[-1], f, indent=2)
    
    # 生成结果摘要
    summary_df = pd.DataFrame([
        {
            'inference_steps': r['inference_steps'], 
            'test_time_epochs': r['test_time_epochs'],
            'overall_accuracy': r['overall_accuracy'],
            'pixel_accuracy': r['pixel_accuracy']
        } 
        for r in results
    ])
    
    # 保存为CSV
    summary_df.to_csv(os.path.join(scan_dir, 'param_scan_summary.csv'), index=False)
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    
    # 准确率热力图
    plt.subplot(2, 1, 1)
    heatmap_data = summary_df.pivot(
        index='test_time_epochs', 
        columns='inference_steps', 
        values='overall_accuracy'
    )
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Overall Accuracy')
    plt.title('Task Accuracy by Parameter Combination')
    plt.xlabel('Inference Steps')
    plt.ylabel('Test-Time Training Epochs')
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}', 
                    ha='center', va='center', color='white')
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    # 像素准确率热力图
    plt.subplot(2, 1, 2)
    heatmap_data = summary_df.pivot(
        index='test_time_epochs', 
        columns='inference_steps', 
        values='pixel_accuracy'
    )
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Pixel Accuracy')
    plt.title('Pixel Accuracy by Parameter Combination')
    plt.xlabel('Inference Steps')
    plt.ylabel('Test-Time Training Epochs')
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}', 
                    ha='center', va='center', color='white')
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    plt.tight_layout()
    plt.savefig(os.path.join(scan_dir, 'param_scan_heatmap.png'))
    
    print(f"参数扫描完成，结果保存在: {scan_dir}")
    print(summary_df)
    
    # 返回最佳参数
    best_overall_idx = summary_df['overall_accuracy'].argmax()
    best_pixel_idx = summary_df['pixel_accuracy'].argmax()
    
    best_overall = summary_df.iloc[best_overall_idx]
    best_pixel = summary_df.iloc[best_pixel_idx]
    
    print(f"\n最佳任务准确率参数: steps={best_overall['inference_steps']}, tt_epochs={best_overall['test_time_epochs']}, accuracy={best_overall['overall_accuracy']:.4f}")
    print(f"最佳像素准确率参数: steps={best_pixel['inference_steps']}, tt_epochs={best_pixel['test_time_epochs']}, accuracy={best_pixel['pixel_accuracy']:.4f}")
    
    return summary_df, scan_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VAE models on ARC tasks")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to ARC test data directory")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--tt_epochs", type=int, default=10, help="Test-time training epochs (0 to disable)")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--standard_vae", action="store_true", help="Use standard VAE instead of discrete VAE")
    parser.add_argument("--codebook_size", type=int, default=512, help="Codebook size for discrete VAE")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension for discrete VAE")
    parser.add_argument("--param_scan", action="store_true", help="Perform parameter scanning experiment")
    parser.add_argument("--gpu", type=int, default=None, help="Specific GPU to use (e.g. 0, 1, etc)")
    
    args = parser.parse_args()
    
    if args.param_scan:
        # 执行参数扫描实验
        parameter_scan(
            args.model,
            args.data,
            use_discrete_vae=not args.standard_vae,
            codebook_size=args.codebook_size,
            embedding_dim=args.embedding_dim,
            gpu_id=args.gpu
        )
    else:
        # 正常评估单个参数设置
        evaluate_all_tasks(
            args.model, 
            args.data, 
            args.steps, 
            args.tt_epochs, 
            not args.no_amp,
            not args.standard_vae,
            args.codebook_size,
            args.embedding_dim,
            args.gpu
        )