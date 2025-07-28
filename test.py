import torch
import numpy as np
import json
import os
from model import DiscreteVAE

def load_model(model_path):
    model = DiscreteVAE()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

def preprocess_grid(grid):
    """Convert grid to one-hot encoding and pad if necessary"""
    h, w = grid.shape
    one_hot = np.zeros((10, h, w))
    for i in range(h):
        for j in range(w):
            one_hot[grid[i, j], i, j] = 1
    
    # Pad to 30x30 if needed
    if h < 30 or w < 30:
        padded = np.zeros((10, 30, 30))
        padded[:, :h, :w] = one_hot
        one_hot = padded
        
    # Convert to tensor
    return torch.tensor(one_hot, dtype=torch.float).unsqueeze(0)

def postprocess_output(output, original_shape):
    """Convert model output back to grid format with original shape"""
    h, w = original_shape
    output = output.squeeze().detach().cpu().numpy()
    if output.shape[0] == 30 and output.shape[1] == 30:
        output = output[:h, :w]
    return output

def evaluate_arc_task(model, task_path, num_inference_steps=1):
    """Evaluate model on a single ARC task"""
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    device = next(model.parameters()).device
    
    # Evaluate on test examples
    correct = 0
    total = len(task['test'])
    
    for test_example in task['test']:
        input_grid = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        
        input_tensor = preprocess_grid(input_grid).to(device)
        
        # Apply model with specified number of inference steps
        with torch.no_grad():
            reconstructed = model.reconstruct(input_tensor, num_steps=num_inference_steps)
        
        # Post-process output
        output_grid = postprocess_output(reconstructed, expected_output.shape)
        
        # Check if output matches expected
        if np.array_equal(output_grid, expected_output):
            correct += 1
    
    accuracy = correct / total
    return accuracy

def evaluate_all_tasks(model_path, data_dir, num_inference_steps=1):
    """Evaluate model on all ARC tasks in directory"""
    model = load_model(model_path)
    
    total_tasks = 0
    total_correct = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            task_path = os.path.join(data_dir, filename)
            accuracy = evaluate_arc_task(model, task_path, num_inference_steps)
            print(f"Task {filename}: Accuracy = {accuracy:.2f}")
            
            total_tasks += 1
            total_correct += accuracy
    
    overall_accuracy = total_correct / total_tasks
    print(f"\nOverall Accuracy across {total_tasks} tasks: {overall_accuracy:.4f}")
    
    return overall_accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DiscreteVAE on ARC tasks")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to ARC test data")
    parser.add_argument("--steps", type=int, default=1, help="Number of inference steps")
    
    args = parser.parse_args()
    
    evaluate_all_tasks(args.model, args.data, args.steps)