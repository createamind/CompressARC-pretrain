import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiscreteVAE(nn.Module):
    def __init__(self, grid_size=30, num_categories=10, hidden_dim=512, latent_dim=256):
        super(DiscreteVAE, self).__init__()
        
        self.grid_size = grid_size
        self.num_categories = num_categories
        self.grid_cells = grid_size * grid_size
        self.input_dim = self.grid_cells * num_categories
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.grid_cells * num_categories)
        )
    
    def encode(self, x):
        """
        Encode input grid to latent representation
        x shape: [batch_size, num_categories, grid_size, grid_size]
        """
        # Reshape to [batch_size, input_dim]
        x = x.view(x.size(0), -1)
        
        # Pass through encoder
        h = self.encoder(x)
        
        # Get mu and logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent space using reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent representation to grid logits
        Returns shape: [batch_size, grid_cells, num_categories]
        """
        # Pass through decoder to get flattened logits
        logits_flat = self.decoder(z)
        
        # Reshape to [batch_size, grid_cells, num_categories]
        logits = logits_flat.view(-1, self.grid_cells, self.num_categories)
        
        return logits
    
    def forward(self, x):
        """
        Complete forward pass
        x shape: [batch_size, num_categories, grid_size, grid_size]
        Returns:
            - reconstruction: [batch_size, grid_cells, num_categories]
            - mu
            - logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def reconstruct(self, x, num_steps=1):
        """
        Reconstruct input with multiple refinement steps
        x shape: [batch_size, num_categories, grid_size, grid_size]
        """
        current_x = x
        for _ in range(num_steps):
            reconstruction, _, _ = self.forward(current_x)
            
            # Convert reconstruction logits to one-hot
            probs = F.softmax(reconstruction, dim=-1)
            indices = torch.argmax(probs, dim=-1)  # [batch_size, grid_cells]
            
            # Create new one-hot input for next iteration
            one_hot = torch.zeros_like(current_x)
            for b in range(indices.size(0)):
                for i in range(self.grid_cells):
                    # Convert flat index to 2D position
                    row = i // self.grid_size
                    col = i % self.grid_size
                    category = indices[b, i]
                    one_hot[b, category, row, col] = 1.0
            
            current_x = one_hot
            
        return indices.reshape(-1, self.grid_size, self.grid_size)