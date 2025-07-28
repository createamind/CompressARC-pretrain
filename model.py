import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiscreteVAE(nn.Module):
    def __init__(self, input_dim=30*30, hidden_dim=512, latent_dim=256, num_categories=10):
        super(DiscreteVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent representation
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * num_categories)
        )
        
        self.input_dim = input_dim
        self.num_categories = num_categories
    
    def encode(self, x):
        x = x.view(-1, self.input_dim)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        output = self.decoder(z)
        output = output.view(-1, self.input_dim, self.num_categories)
        return output
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar
    
    def reconstruct(self, x, num_steps=1):
        """Reconstruct input with multiple refinement steps"""
        current_x = x
        for _ in range(num_steps):
            output, _, _ = self.forward(current_x)
            # Convert output logits to categorical distribution and sample
            probs = F.softmax(output, dim=2)
            sampled = torch.argmax(probs, dim=2)
            current_x = sampled
        return current_x