import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import random

from arc_compressor import ARCCompressor
from preprocessing import preprocess_arc_task
from train import train_model
import config

# 新增导入：从我们之前创建的评估脚本中导入评估函数
from evaluate_shared import evaluate_with_shared_weights

def train_shared_weights():
    """
    Trains a single model on all ARC training tasks with shared weights.
    """
    print(f"Using device: {config.DEVICE}")

    # 1. Initialize a single model
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
    
    # 2. Find all training task files
    training_tasks_path = os.path.join(config.ARC_PATH, 'arc-agi_training_challenges.json')
    
    try:
        with open(training_tasks_path, 'r') as f:
            training_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Training tasks file not found at '{training_tasks_path}'")
        print("Please ensure the ARC dataset is available and the `ARC_PATH` in config.py is correct.")
        return

    task_ids = list(training_tasks.keys())
    random.shuffle(task_ids) # Shuffle tasks to avoid any order bias

    print(f"Found {len(task_ids)} training tasks. Starting training...")

    # 3. Loop through all tasks and train the model
    for task_id in tqdm(task_ids, desc="Training on all tasks"):
        task_data = training_tasks[task_id]
        
        try:
            input_grids, output_grids, _, _ = preprocess_arc_task(task_data)
            
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
                task_id=task_id
            )

        except Exception as e:
            tqdm.write(f"Error processing task {task_id}: {e}")
            continue

    # 4. Save the final model weights
    try:
        torch.save(model.state_dict(), config.SHARED_WEIGHTS_PATH)
        print(f"\nTraining complete. Shared model weights saved to '{config.SHARED_WEIGHTS_PATH}'")
    except Exception as e:
        print(f"\nError saving model weights: {e}")


if __name__ == '__main__':
    # --- 步骤 1: 执行训练 ---
    print("--- Starting Shared Weights Training ---")
    train_shared_weights()
    print("--- Shared Weights Training Finished ---")

    print("\n" + "="*50 + "\n")

    # --- 步骤 2: 自动开始评估 ---
    print("--- Starting Evaluation with Fine-Tuning ---")
    
    # 控制测试时微调的步数（训练周期数）
    # 设置为 0 将禁用微调，行为和我们改动的第一版相同
    # 设置为正数（例如 200）将在每个任务上微调 200 步以提升准确性
    TEST_TIME_STEPS = 200

    # 确定评估文件路径
    eval_file = os.path.join(config.ARC_PATH, 'arc-agi_solutions.json')
    if not os.path.exists(eval_file):
         print(f"'{eval_file}' not found. Evaluating on training set as a demonstration.")
         eval_file = os.path.join(config.ARC_PATH, 'arc-agi_training_challenges.json')

    # 调用评估函数
    evaluate_with_shared_weights(eval_file, fine_tune_steps=TEST_TIME_STEPS)
    print("--- Evaluation Finished ---")