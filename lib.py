import torch
from tqdm import tqdm
import config
from preprocessing import pad_grid, rot_augment, flip_augment

def preprocess_task_for_sharing(task_data, max_h, max_w):
    """
    A modified and importable version of the preprocessing logic.
    It processes a single task's data into padded grids.
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
    """
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    iterator = range(epochs)
    if not disable_tqdm:
        iterator = tqdm(iterator, desc=f"Training Task {task_id}", leave=False)

    for epoch in iterator:
        # Simple batching by slicing
        for i in range(0, len(train_in), batch_size):
            batch_in = torch.tensor(train_in[i:i+batch_size], dtype=torch.long, device=device)
            batch_out = torch.tensor(train_out[i:i+batch_size], dtype=torch.long, device=device)

            optimizer.zero_grad()
            pred, _ = model(batch_in)
            loss = loss_fn(pred, batch_out)
            loss.backward()
            optimizer.step()

        if not disable_tqdm:
            iterator.set_postfix(loss=loss.item())
