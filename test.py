import torch
import numpy as np
import json
import os
import argparse
import time
from model import DiscreteVAE
from discrete_vae import DiscreteVAE as FullDiscreteVAE
from torch.cuda.amp import autocast
import torch.nn.functional as F

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
        output_grid = postprocess_grid(output_grid, expected_output.shape)
        
        # Check if output matches expected
        if np.array_equal(output_grid, expected_output):
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate_all_tasks(model_path, data_dir, num_inference_steps=5, test_time_epochs=10, 
                      use_amp=True, use_discrete_vae=True, codebook_size=512, embedding_dim=64):
    """Evaluate model on all tasks in a directory"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path, 
        device, 
        use_discrete_vae=use_discrete_vae,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim
    )
    
    total_tasks = 0
    total_correct = 0
    task_results = {}
    
    start_time = time.time()
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            task_path = os.path.join(data_dir, filename)
            task_start_time = time.time()
            
            accuracy = evaluate_task(
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
            task_results[filename] = accuracy
            
            print(f"任务 {filename}: 准确率 = {accuracy:.2f} (耗时 {task_time:.2f}s)")
            
            total_tasks += 1
            total_correct += accuracy
    
    overall_accuracy = total_correct / total_tasks if total_tasks > 0 else 0
    total_time = time.time() - start_time
    
    print(f"\n{total_tasks} 个任务的总体准确率: {overall_accuracy:.4f}")
    print(f"总评估时间: {total_time:.2f}s, 每任务平均: {total_time/total_tasks:.2f}s")
    
    # 保存结果到文件
    model_type = "discrete" if use_discrete_vae else "standard"
    results_path = os.path.join(os.path.dirname(model_path), f'{model_type}_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model_path': model_path,
            'model_type': model_type,
            'overall_accuracy': overall_accuracy,
            'total_tasks': total_tasks,
            'inference_steps': num_inference_steps,
            'test_time_epochs': test_time_epochs,
            'codebook_size': codebook_size if use_discrete_vae else None,
            'embedding_dim': embedding_dim if use_discrete_vae else None,
            'task_results': task_results,
            'evaluation_time': total_time
        }, f, indent=2)
    
    print(f"结果已保存到 {results_path}")
    
    return overall_accuracy

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
    
    args = parser.parse_args()
    
    evaluate_all_tasks(
        args.model, 
        args.data, 
        args.steps, 
        args.tt_epochs, 
        not args.no_amp,
        not args.standard_vae,
        args.codebook_size,
        args.embedding_dim
    )