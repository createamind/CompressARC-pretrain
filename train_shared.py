import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import random

from arc_compressor import ARCCompressor
# 修正：导入正确的函数名
from preprocessing import preprocess_tasks
from train import train_model
import config

# 导入评估函数
from evaluate_shared import evaluate_with_shared_weights

def train_shared_weights():
    """
    Trains a single model on all ARC training tasks with shared weights.
    """
    print(f"Using device: {config.DEVICE}")

    model = ARCCompressor(
        v_dim=config.V_DIM,
        e_dim=config.E_DIM,
        l_dim=config.L_DIM,
        n_heads=config.N_HEADS,
        n_blocks=config.N_BLOCKS,
        dropout=config.DROPOUT,
        device=config.DEVICE
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    training_tasks_path = os.path.join(config.ARC_PATH, 'arc-agi_training_challenges.json')
    
    try:
        with open(training_tasks_path, 'r') as f:
            training_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training tasks file not found at '{training_tasks_path}'")
        print("Please ensure the ARC dataset is available and the `ARC_PATH` in config.py is correct.")
        return

    task_ids = list(training_tasks.keys())
    random.shuffle(task_ids)

    print(f"Found {len(task_ids)} training tasks. Starting training...")

    for task_id in tqdm(task_ids, desc="Training on all tasks"):
        task_data = training_tasks[task_id]
        
        try:
            # 修正：使用正确的函数名 preprocess_tasks 并传入所需参数
            input_grids, output_grids, _, _ = preprocess_tasks(task_data, config.MAX_H, config.MAX_W)
            
            if not input_grids:
                tqdm.write(f"Skipping task {task_id}: No valid training pairs found after preprocessing.")
                continue

            train_model(
                model=model,
                optimizer=optimizer,
                train_in=input_grids,
                train_out=output_grids,
                epochs=config.EPOCHS_PER_TASK,
                batch_size=config.BATCH_SIZE,
                device=config.DEVICE,
                task_id=task_id,
                disable_tqdm=True # Disable inner loop tqdm to keep output clean
            )

        except Exception as e:
            tqdm.write(f"Error processing task {task_id}: {e}")
            continue

    try:
        torch.save(model.state_dict(), config.SHARED_WEIGHTS_PATH)
        print(f"\nTraining complete. Shared model weights saved to '{config.SHARED_WEIGHTS_PATH}'")
    except Exception as e:
        print(f"\nError saving model weights: {e}")


if __name__ == '__main__':
    print("--- Starting Shared Weights Training ---")
    train_shared_weights()
    print("--- Shared Weights Training Finished ---")

    print("\n" + "="*50 + "\n")

    print("--- Starting Evaluation with Fine-Tuning ---")
    
    TEST_TIME_STEPS = 200 

    eval_file = os.path.join(config.ARC_PATH, 'arc-agi_solutions.json')
    if not os.path.exists(eval_file):
         print(f"'{eval_file}' not found. Evaluating on training set as a demonstration.")
         eval_file = os.path.join(config.ARC_PATH, 'arc-agi_training_challenges.json')

    evaluate_with_shared_weights(eval_file, fine_tune_steps=TEST_TIME_STEPS)
    print("--- Evaluation Finished ---")