import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from model import DiscreteVAE
import torch.nn.functional as F

def load_base_model(model_path):
    """Load the base model that was trained on all tasks"""
    model = DiscreteVAE()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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

def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    """Loss function for VAE"""
    # Reconstruction loss (Cross-entropy for categorical data)
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)
    recon_flat = reconstruction.view(batch_size, -1, 10)
    recon_loss = F.cross_entropy(recon_flat, x_flat)
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size
    
    return recon_loss + beta * kld

def test_time_train(model, train_examples, epochs=10, lr=0.001):
    """Fine-tune model on task-specific training examples"""
    # Create a copy of the model for fine-tuning
    fine_tuned_model = DiscreteVAE()
    fine_tuned_model.load_state_dict(model.state_dict())
    device = next(model.parameters()).device
    fine_tuned_model.to(device)
    fine_tuned_model.train()
    
    # Prepare training data
    input_tensors = []
    target_tensors = []
    
    for example in train_examples:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        input_tensor = preprocess_grid(input_grid).to(device)
        output_tensor = torch.tensor(output_grid, dtype=torch.long).to(device)
        
        input_tensors.append(input_tensor)
        target_tensors.append(output_tensor)
    
    # Set up optimizer
    optimizer = optim.Adam(fine_tuned_model.parameters(), lr=lr)
    
    # Fine-tuning loop
    for epoch in range(epochs):
        total_loss = 0
        
        for i in range(len(input_tensors)):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = fine_tuned_model(input_tensors[i])
            loss = vae_loss(recon_batch, target_tensors[i], mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(input_tensors)
        print(f"Test-time training epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    fine_tuned_model.eval()
    return fine_tuned_model

def evaluate_arc_task(base_model, task_path, fine_tune_epochs=10, inference_steps=5):
    """Evaluate model on a single ARC task with test-time training and multiple inference steps"""
    with open(task_path, 'r') as f:
        task = json.load(f)
    
    device = next(base_model.parameters()).device
    
    # First, fine-tune model on the task's training examples
    fine_tuned_model = test_time_train(base_model, task['train'], epochs=fine_tune_epochs)
    
    # Then evaluate on test examples using multiple inference steps
    correct = 0
    total = len(task['test'])
    
    for test_example in task['test']:
        input_grid = np.array(test_example['input'])
        expected_output = np.array(test_example['output'])
        
        input_tensor = preprocess_grid(input_grid).to(device)
        
        # Apply model with specified number of inference steps
        with torch.no_grad():
            reconstructed = fine_tuned_model.reconstruct(input_tensor, num_steps=inference_steps)
        
        # Post-process output
        output_grid = postprocess_output(reconstructed, expected_output.shape)
        
        # Check if output matches expected
        if np.array_equal(output_grid, expected_output):
            correct += 1
    
    accuracy = correct / total
    return accuracy

def evaluate_all_tasks(model_path, data_dir, fine_tune_epochs=10, inference_steps=5):
    """Evaluate model on all ARC tasks with the combined strategy"""
    base_model = load_base_model(model_path)
    
    total_tasks = 0
    total_correct = 0
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            task_path = os.path.join(data_dir, filename)
            accuracy = evaluate_arc_task(base_model, task_path, 
                                         fine_tune_epochs=fine_tune_epochs, 
                                         inference_steps=inference_steps)
            
            print(f"Task {filename}: Accuracy = {accuracy:.2f}")
            
            total_tasks += 1
            total_correct += accuracy
    
    overall_accuracy = total_correct / total_tasks
    print(f"\nOverall Accuracy across {total_tasks} tasks: {overall_accuracy:.4f}")
    
    return overall_accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DiscreteVAE on ARC tasks with test-time training")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to ARC test data")
    parser.add_argument("--tt_epochs", type=int, default=10, help="Test-time training epochs")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    
    args = parser.parse_args()
    
    evaluate_all_tasks(args.model, args.data, args.tt_epochs, args.steps)