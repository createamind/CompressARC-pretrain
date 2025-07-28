import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import copy

from arc_compressor import ARCCompressor
# 修正：从新的 lib.py 导入可重用函数
from lib import preprocess_task_for_sharing, training_loop
import config

def evaluate_with_shared_weights(tasks_file_path, output_dir="evaluation_outputs", fine_tune_steps=100):
    """
    Loads shared weights, fine-tunes on each task's train pairs, and evaluates.
    """
    print(f"Using device: {config.DEVICE}")
    print(f"Test-time fine-tuning steps per task: {fine_tune_steps}")
    os.makedirs(output_dir, exist_ok=True)

    base_model = ARCCompressor(
        v_dim=config.V_DIM,
        e_dim=config.E_DIM,
        l_dim=config.L_DIM,
        n_heads=config.N_HEADS,
        n_blocks=config.N_BLOCKS,
        dropout=config.DROPOUT,
        device=config.DEVICE
    ).to(config.DEVICE)

    try:
        base_model.load_state_dict(torch.load(config.SHARED_WEIGHTS_PATH, map_location=config.DEVICE))
        print(f"Successfully loaded shared weights from '{config.SHARED_WEIGHTS_PATH}'")
    except FileNotFoundError:
        print(f"Error: Weight file not found at '{config.SHARED_WEIGHTS_PATH}'")
        print("Please run `train_shared.py` first to train and save the model.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    try:
        with open(tasks_file_path, 'r') as f:
            evaluation_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Evaluation tasks file not found at '{tasks_file_path}'")
        return

    task_ids = list(evaluation_tasks.keys())
    print(f"Found {len(task_ids)} tasks for evaluation. Starting fine-tuning and prediction...")

    for task_id in tqdm(task_ids, desc="Evaluating All Tasks"):
        task_data = evaluation_tasks[task_id]
        task_output_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_output_dir, exist_ok=True)

        task_model = copy.deepcopy(base_model)
        
        # 修正：使用 lib.py 中的预处理函数
        train_in_grids, train_out_grids, test_in_grids, _ = preprocess_task_for_sharing(task_data, config.MAX_H, config.MAX_W)

        if fine_tune_steps > 0 and train_in_grids:
            optimizer = optim.Adam(task_model.parameters(), lr=config.LEARNING_RATE / 10)
            # 修正：使用 lib.py 中的训练循环函数进行微调
            training_loop(
                model=task_model,
                optimizer=optimizer,
                train_in=train_in_grids,
                train_out=train_out_grids,
                epochs=fine_tune_steps,
                batch_size=config.BATCH_SIZE,
                device=config.DEVICE,
                task_id=f"{task_id}_finetune",
                disable_tqdm=True
            )

        task_model.eval()
        
        for i, test_input_grid in enumerate(test_in_grids):
            input_tensor = torch.tensor([test_input_grid], dtype=torch.long, device=config.DEVICE)

            with torch.no_grad():
                predicted_grid_tensor, _ = task_model(input_tensor)
            
            predicted_grid = predicted_grid_tensor.argmax(dim=1).squeeze(0).cpu().numpy().tolist()

            output_file_path = os.path.join(task_output_dir, f"prediction_{i}.json")
            with open(output_file_path, 'w') as f:
                json.dump(predicted_grid, f)

    print(f"\nEvaluation complete. Predictions saved in '{output_dir}' directory.")