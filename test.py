import torch
import numpy as np
import json
import os
import argparse
from model import DiscreteVAE

def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    model = DiscreteVAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
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

def evaluate_task(model, task_path, num_inference_steps=1, device='cuda'):
    """Evaluate model on a single ARC task"""
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    # Test-time training on this specific task
    if task['train']:
        model = test_time_train(model, task['train'], device=device)
    
    # Evaluate on test examples
    correct = 0
    total = len(task['test'])
    
    for test_example in task['test']:
        input_grid = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        
        # Process input
        input_tensor = preprocess_grid(input_grid).to(device)
        
        # Generate output with specified inference steps
        with torch.no_grad():
            output_grid = model.reconstruct(input_tensor, num_steps=num_inference_steps)
        
        # Post-process output
        output_grid = postprocess_grid(output_grid, expected_output.shape)
        
        # Check if output matches expected
        if np.array_equal(output_grid, expected_output):
            correct += 1
    
    accuracy = correct / total
    return accuracy

def test_time_train(model, train_examples, epochs=10, lr=0.001, device='cuda'):
    """Fine-tune model on task-specific examples"""
    # Create a copy of the model for fine-tuning
    fine_tuned_model = DiscreteVAE()
    fine_tuned_model.load_state_dict({k: v.cpu().clone() for k, v in model.state_dict().items()})
    fine_tuned_model.to(device)
    fine_tuned_model.train()
    
    # Prepare optimizer
    optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=lr)
    
    # Process training examples
    for epoch in range(epochs):
        epoch_loss = 0
        
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            # Preprocess input and output
            input_tensor = preprocess_grid(input_grid).to(device)
            target_tensor = torch.tensor(output_grid, dtype=torch.long).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = fine_tuned_model(input_tensor)
            
            # Compute loss - we want to predict output_grid from input_grid
            h, w = output_grid.shape
            h_min, w_min = min(h, 30), min(w, 30)
            target_flat = target_tensor.reshape(-1)[:h_min*w_min]
            
            # Extract relevant part of reconstruction
            indices = torch.arange(h_min*w_min).to(device)
            relevant_recon = recon_batch[0, indices, :]
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(relevant_recon, target_flat)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Test-time training epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    fine_tuned_model.eval()
    return fine_tuned_model

def evaluate_all_tasks(model_path, data_dir, num_inference_steps=1, test_time_training=True):
    """Evaluate model on all tasks in a directory"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    total_tasks = 0
    total_correct = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            task_path = os.path.join(data_dir, filename)
            accuracy = evaluate_task(model, task_path, num_inference_steps, device)
            
            print(f"Task {filename}: Accuracy = {accuracy:.2f}")
            
            total_tasks += 1
            total_correct += accuracy
    
    overall_accuracy = total_correct / total_tasks if total_tasks > 0 else 0
    print(f"\nOverall Accuracy across {total_tasks} tasks: {overall_accuracy:.4f}")
    
    return overall_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DiscreteVAE on ARC tasks")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to ARC test data directory")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--no_tt", action="store_true", help="Disable test-time training")
    
    args = parser.parse_args()
    
    evaluate_all_tasks(args.model, args.data, args.steps, not args.no_tt)