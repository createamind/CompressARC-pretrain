import torch
from tqdm import tqdm
import config
import random

# ==============================================================================
# Functions copied directly from the original `preprocessing.py` to ensure they exist
# This resolves all previous ImportError issues.
# ==============================================================================

def pad_grid(grid, max_h, max_w):
    """Pads a grid with 0s to make it max_h x max_w."""
    h, w = len(grid), len(grid[0])
    padded_grid = [[0] * max_w for _ in range(max_h)]
    for r in range(h):
        for c in range(w):
            padded_grid[r][c] = grid[r][c]
    return padded_grid

def rot_augment(grid):
    """Rotates a grid by 0, 90, 180, or 270 degrees."""
    k = random.randint(0, 3)
    return [list(row) for row in torch.tensor(grid).rot90(k, [0, 1]).tolist()]

def flip_augment(grid):
    """Flips a grid horizontally or vertically."""
    if random.random() < 0.5:
        return [list(row) for row in torch.tensor(grid).flip(0).tolist()]
    if random.random() < 0.5:
        return [list(row) for row in torch.tensor(grid).flip(1).tolist()]
    return grid

# ==============================================================================
# Our custom, importable functions that use the helpers above
# ==============================================================================

def preprocess_task_for_sharing(task_data, max_h, max_w):
    """
    Processes a single task's data into padded grids.
    This function is now self-contained within lib.py.
    """
    train_pairs = task_data.get('train', [])
    test_pairs = task_data.get('test', [])

    train_in_grids, train_out_grids = [], []
    for pair in train_pairs:
        train_in_grids.append(pad_grid(pair['input'], max_h, max_w))
        train_out_grids.append(pad_grid(pair['output'], max_h, max_w))

    test_in_grids, test_out_grids = [], []
    for pair in test_pairs:
        test_in_grids.append(pad_grid(pair['input'], max_h, max_w))
        if 'output' in pair:
            test_out_grids.append(pad_grid(pair['output'], max_h, max_w))

    return train_in_grids, train_out_grids, test_in_grids, test_out_grids

def training_loop(model, optimizer, train_in, train_out, epochs, batch_size, device, task_id="N/A", disable_tqdm=False):
    """
    An importable version of the training loop.
    This function is self-contained within lib.py.
    """
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    iterator = range(epochs)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Training Task {task_id}", leave=False)

    for _ in iterator:
        # Simple batching by creating a random sample for each step
        # This is more robust for tasks with very few examples
        indices = torch.randint(0, len(train_in), (batch_size,))
        
        batch_in_list = [train_in[i] for i in indices]
        batch_out_list = [train_out[i] for i in indices]

        batch_in = torch.tensor(batch_in_list, dtype=torch.long, device=device)
        batch_out = torch.tensor(batch_out_list, dtype=torch.long, device=device)

        optimizer.zero_grad()
        pred, _ = model(batch_in)
        loss = loss_fn(pred, batch_out)
        loss.backward()
        optimizer.step()

        if not disable_tqdm:
            iterator.set_postfix(loss=loss.item())