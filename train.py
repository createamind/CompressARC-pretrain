import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import torch.nn.functional as F
from model import DiscreteVAE

# Dataset class for ARC tasks
class ARCDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        # Load data from the specified path
        import json
        for filename in os.listdir(data_path):
            if filename.endswith('.json'):
                with open(os.path.join(data_path, filename), 'r') as f:
                    task = json.load(f)
                    for train in task['train']:
                        input_grid = np.array(train['input'])
                        self.data.append(input_grid)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        grid = self.data[idx]
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

# Loss function for discrete VAE
def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    """
    Computes VAE loss with categorical reconstruction loss and KL divergence
    
    Args:
        reconstruction: tensor of shape [batch_size, grid_cells, num_categories]
        x: tensor of shape [batch_size, grid_height, grid_width]
        mu: mean of latent distribution
        logvar: log variance of latent distribution
        beta: weight for KL divergence term
        
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
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size
    
    # Total loss
    total_loss = recon_loss + beta * kld
    
    return total_loss

def train_model(data_path, save_dir, epochs=100, batch_size=32, learning_rate=1e-3, beta=1.0):
    """
    Train the DiscreteVAE model
    
    Args:
        data_path: path to directory containing ARC task json files
        save_dir: directory to save model checkpoints
        epochs: number of training epochs
        batch_size: batch size for training
        learning_rate: learning rate for optimizer
        beta: weight for KL divergence term in loss
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = ARCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = DiscreteVAE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Compute loss
            loss = vae_loss(recon_batch, target, mu, logvar, beta)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Compute average loss for epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
    
    args = parser.parse_args()
    
    train_model(args.data, args.save_dir, args.epochs, args.batch_size, args.lr, args.beta)