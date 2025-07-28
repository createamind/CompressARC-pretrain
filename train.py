import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
import json
import torch.nn.functional as F
from model import DiscreteVAE
from utils import compute_task_complexity, grid_augmentation
import time
from collections import defaultdict

# Dataset class with curriculum learning capability
class ARCDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []  # List of (file_path, complexity) tuples
        self.task_data = {}  # Cache for task data
        self.complexity_percentile = 1.0  # Default: use all samples
        self.current_indices = []  # Indices of samples to use in current curriculum stage
        self.load_data()
    
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        start_time = time.time()
        
        # Load all tasks and compute complexities
        all_tasks = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    self.task_data[file_path] = task_data
                    complexity = compute_task_complexity(task_data)
                    all_tasks.append((file_path, complexity, task_data))
        
        # Sort by complexity
        all_tasks.sort(key=lambda x: x[1])
        
        # Convert tasks to grid samples
        for file_path, complexity, task_data in all_tasks:
            for train in task_data['train']:
                input_grid = np.array(train['input'])
                self.samples.append((input_grid, complexity, file_path))
        
        # Initialize current indices to all samples
        self.current_indices = list(range(len(self.samples)))
        
        print(f"Loaded {len(self.samples)} examples from {len(all_tasks)} tasks")
        print(f"Complexity range: {self.samples[0][1]:.2f} - {self.samples[-1][1]:.2f}")
        print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    def set_complexity_threshold(self, percentile):
        """Set curriculum threshold to use only samples below the given percentile"""
        self.complexity_percentile = percentile
        if percentile >= 1.0:
            self.current_indices = list(range(len(self.samples)))
        else:
            # Calculate complexity threshold based on percentile
            sorted_complexities = sorted([sample[1] for sample in self.samples])
            threshold_idx = int(percentile * len(sorted_complexities))
            threshold = sorted_complexities[threshold_idx]
            
            # Update indices
            self.current_indices = [
                i for i, (_, complexity, _) in enumerate(self.samples) 
                if complexity <= threshold
            ]
        
        print(f"Curriculum updated: Using {len(self.current_indices)}/{len(self.samples)} samples ({percentile*100:.0f}%)")
    
    def __len__(self):
        return len(self.current_indices)
    
    def __getitem__(self, idx):
        # Map index to current curriculum subset
        real_idx = self.current_indices[idx]
        grid, _, _ = self.samples[real_idx]
        
        # Apply data augmentation
        if np.random.random() < 0.5:  # 50% chance of augmentation
            grid = grid_augmentation(grid)
        
        # Convert grid to one-hot encoding
        h, w = grid.shape
        one_hot = np.zeros((10, 30, 30))  # Fixed size for all grids
        
        # Copy original grid data to fixed-size array
        for i in range(min(h, 30)):
            for j in range(min(w, 30)):
                one_hot[grid[i, j], i, j] = 1
        
        # Create target grid (padded to 30x30)
        target_grid = np.zeros((30, 30), dtype=np.long)
        target_grid[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        
        # Convert to tensors
        return torch.tensor(one_hot, dtype=torch.float), torch.tensor(target_grid, dtype=torch.long)

# Loss function for discrete VAE with KL annealing
def vae_loss(reconstruction, x, mu, logvar, beta_weight, current_step, total_steps):
    """
    Computes VAE loss with categorical reconstruction loss and KL divergence
    
    Args:
        reconstruction: tensor of shape [batch_size, grid_cells, num_categories]
        x: tensor of shape [batch_size, grid_height, grid_width]
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta_weight: maximum KL weight
        current_step: current training step
        total_steps: total training steps for annealing
        
    Returns:
        Total loss (reconstruction + beta * KL)
    """
    batch_size = x.size(0)
    
    # Flatten target to [batch_size, grid_cells]
    x_flat = x.reshape(batch_size, -1)
    
    # Compute cross-entropy loss
    recon_loss = 0
    for i in range(x_flat.size(1)):  # For each position
        logits = reconstruction[:, i, :]  # [batch_size, num_categories]
        target = x_flat[:, i]            # [batch_size]
        recon_loss += F.cross_entropy(logits, target)
    
    # Average over positions
    recon_loss = recon_loss / x_flat.size(1)
    
    # KL divergence with annealing
    # Gradually increase beta from 0.01 to beta_weight over first 30% of training
    annealing_factor = min(1.0, current_step / (total_steps * 0.3))
    beta = 0.01 + (beta_weight - 0.01) * annealing_factor
    
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size
    
    # Total loss
    total_loss = recon_loss + beta * kld
    
    return total_loss, recon_loss, beta * kld

def train_model(data_path, save_dir, epochs=100, batch_size=32, learning_rate=1e-3, 
               beta=1.0, accumulation_steps=4, use_amp=True, use_residual=True):
    """
    Train the DiscreteVAE model with curriculum learning and other optimizations
    
    Args:
        data_path: path to directory containing ARC task json files
        save_dir: directory to save model checkpoints
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        beta: weight for KL divergence term in loss
        accumulation_steps: number of steps to accumulate gradients
        use_amp: whether to use automatic mixed precision
        use_residual: whether to use residual connections in model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = ARCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=True, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = DiscreteVAE(use_residual=use_residual)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if use_amp else None
    
    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Using automatic mixed precision: {use_amp}")
    print(f"Using residual connections: {use_residual}")
    print(f"Using gradient accumulation: {accumulation_steps} steps")
    
    # Calculate total steps for KL annealing
    total_steps = epochs * (len(dataloader) // accumulation_steps)
    current_step = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        # Update curriculum at specific points
        if epoch < epochs * 0.3:
            dataset.set_complexity_threshold(0.3)  # Start with simpler 30% of tasks
        elif epoch < epochs * 0.6:
            dataset.set_complexity_threshold(0.7)  # Move to 70% of tasks
        else:
            dataset.set_complexity_threshold(1.0)  # Use all tasks
        
        # Reset optimizer
        optimizer.zero_grad()
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass with optional mixed precision
            if use_amp:
                with autocast():
                    recon_batch, mu, logvar = model(data)
                    loss, recon_term, kl_term = vae_loss(
                        recon_batch, target, mu, logvar, beta, current_step, total_steps)
                
                # Backward pass with scaler
                scaler.scale(loss / accumulation_steps).backward()
            else:
                # Standard forward pass
                recon_batch, mu, logvar = model(data)
                loss, recon_term, kl_term = vae_loss(
                    recon_batch, target, mu, logvar, beta, current_step, total_steps)
                
                # Standard backward pass
                (loss / accumulation_steps).backward()
            
            # Track losses
            total_loss += loss.item()
            recon_loss_sum += recon_term.item()
            kl_loss_sum += kl_term.item()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if use_amp:
                    # Step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard step
                    optimizer.step()
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Increment step counter for annealing
                current_step += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(dataloader)}, '
                     f'Loss: {loss.item():.4f}, Recon: {recon_term.item():.4f}, '
                     f'KL: {kl_term.item():.4f}')
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Compute average losses
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_loss_sum / len(dataloader)
        avg_kl = kl_loss_sum / len(dataloader)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f'Epoch: {epoch+1}/{epochs}, Time: {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}, '
              f'Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DiscreteVAE on ARC tasks")
    parser.add_argument("--data", type=str, default="/home/zdx/github/VSAHDC/ARC-AGI-2/data/training", 
                        help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence term")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no_residual", action="store_true", help="Disable residual connections")
    parser.add_argument("--bg_threshold", type=int, default=40, help="Background color threshold percentage")
    
    args = parser.parse_args()
    
    train_model(
        args.data, 
        args.save_dir, 
        args.epochs, 
        args.batch_size, 
        args.lr, 
        args.beta,
        args.accumulation_steps,
        not args.no_amp,
        not args.no_residual
    )