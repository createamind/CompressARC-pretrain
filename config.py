import torch

# --- General ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Path to the ARC dataset
ARC_PATH = 'arc-prize-2024'
# Path to save/load the shared model weights
SHARED_WEIGHTS_PATH = 'shared_arc_model.pth'

# --- Model ---
# Number of symbols in the vocabulary
V_DIM = 12
# Embedding dimension for symbols
E_DIM = 256
# Dimension of the latent space
L_DIM = 256
# Number of attention heads
N_HEADS = 8
# Number of encoder/decoder blocks
N_BLOCKS = 4
# Dropout rate
DROPOUT = 0.1

# --- Training ---
# Batch size for training
BATCH_SIZE = 16
# Learning rate
LEARNING_RATE = 1e-4
# Number of training epochs for each individual task
# In the original repo, this was 2000.
# For shared training, you might want to adjust this.
# A lower number like 50-100 might be sufficient per task.
EPOCHS_PER_TASK = 100

# --- Preprocessing ---
# Maximum grid height
MAX_H = 30
# Maximum grid width
MAX_W = 30