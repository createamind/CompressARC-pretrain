import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
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
        one_hot = np.zeros((10, h, w))
        for i in range(h):
            for j in range(w):
                one_hot[grid[i, j], i, j] = 1

        # Pad both one_hot and target grid to 30x30
        padded_one_hot = np.zeros((10, 30, 30))
        padded_grid = np.zeros((30, 30), dtype=np.long)

        # Copy original data to padded arrays
        padded_one_hot[:, :h, :w] = one_hot
        padded_grid[:h, :w] = grid

        # Convert to tensor
        return torch.tensor(padded_one_hot, dtype=torch.float), torch.tensor(padded_grid, dtype=torch.long)

# Loss function for discrete VAE
def vae_loss(reconstruction, x, mu, logvar, beta=1.0):
    # Reconstruction loss (Cross-entropy for categorical data)
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)
    recon_flat = reconstruction.view(batch_size, -1, 10)
    recon_loss = F.cross_entropy(recon_flat, x_flat)

    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / batch_size

    return recon_loss + beta * kld

def train_model(data_path, save_dir, epochs=100, batch_size=32, learning_rate=1e-3):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = ARCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DiscreteVAE()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, target, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')

        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
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
    parser.add_argument("--data", type=str, default="/home/zdx/github/VSAHDC/ARC-AGI-2/data/training", help="Path to ARC training data")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train_model(args.data, args.save_dir, args.epochs, args.batch_size, args.lr)