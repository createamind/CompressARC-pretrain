import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import copy

from arc_compressor import ARCCompressor
from preprocessing import preprocess_arc_task, pad_grid
from train import train_model
import config

def evaluate_with_shared_weights(tasks_file_path, output_dir="evaluation_outputs", fine_tune_steps=100):
    """
    Loads the shared weights model, fine-tunes it on each task's training examples,
    and then evaluates it on the test examples.
    
    Args:
        tasks_file_path (str): Path to the JSON file containing tasks for evaluation.
        output_dir (str): Directory to save the prediction outputs.
        fine_tune_steps (int): The number of training epochs to fine-tune the model on each task's
                               training examples before making a prediction. Set to 0 to disable.
    """
    print(f"Using device: {config.DEVICE}")
    print(f"Test-time fine-tuning steps per task: {fine_tune_steps}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize the base model architecture
    base_model = ARCCompressor(
        v_dim=config.V_DIM,
        e_dim=config.E_DIM,
        l_dim=config.L_DIM,
        n_heads=config.N_HEADS,
        n_blocks=config.N_BLOCKS,
        dropout=config.DROPOUT,
        device=config.DEVICE
    ).to(config.DEVICE)

    # 2. Load the saved shared weights into the base model
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

    # 3. Load the tasks for evaluation
    try:
        with open(tasks_file_path, 'r') as f:
            evaluation_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Evaluation tasks file not found at '{tasks_file_path}'")
        return

    task_ids = list(evaluation_tasks.keys())
    print(f"Found {len(task_ids)} tasks for evaluation. Starting fine-tuning and prediction...")

    # 4. Loop through tasks, fine-tune, and generate predictions
    with torch.no_grad():
        for task_id in tqdm(task_ids, desc="Evaluating tasks"):
            task_data = evaluation_tasks[task_id]
            task_output_dir = os.path.join(output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)

            # For each task, start with a fresh copy of the base model
            task_model = copy.deepcopy(base_model)
            
            # Preprocess the task's training examples for fine-tuning
            train_in_grids, train_out_grids, _, _ = preprocess_arc_task(task_data)

            # A. Fine-tune the model on this task's training pairs (if steps > 0 and pairs exist)
            if fine_tune_steps > 0 and train_in_grids:
                task_model.train() # Set model to training mode for fine-tuning
                optimizer = optim.Adam(task_model.parameters(), lr=config.LEARNING_RATE / 10) # Use a smaller LR for fine-tuning

                # The train_model function is reused for fine-tuning
                train_model(
                    model=task_model,
                    optimizer=optimizer,
                    train_in=train_in_grids,
                    train_out=train_out_grids,
                    epochs=fine_tune_steps,
                    batch_size=config.BATCH_SIZE,
                    device=config.DEVICE,
                    task_id=f"{task_id}_finetune",
                    disable_tqdm=True # Disable inner progress bar
                )

            # B. Generate predictions using the (potentially) fine-tuned model
            task_model.eval() # Set model to evaluation mode for prediction
            num_test_pairs = len(task_data.get('test', []))

            for i in range(num_test_pairs):
                test_input_grid = task_data['test'][i]['input']
                
                padded_input = pad_grid(test_input_grid, config.MAX_H, config.MAX_W)
                input_tensor = torch.tensor([padded_input], dtype=torch.long, device=config.DEVICE)

                with torch.no_grad():
                    predicted_grid_tensor, _ = task_model(input_tensor)
                
                predicted_grid = predicted_grid_tensor.argmax(dim=1).squeeze(0).cpu().numpy().tolist()

                output_file_path = os.path.join(task_output_dir, f"prediction_{i}.json")
                with open(output_file_path, 'w') as f:
                    json.dump(predicted_grid, f)

    print(f"\nEvaluation complete. Predictions saved in '{output_dir}' directory.")

if __name__ == '__main__':
    # You can now control the number of fine-tuning steps from here
    # Set to 0 to perform zero-shot evaluation (no test-time training)
    # Set to a positive number (e.g., 100, 500) to enable test-time fine-tuning
    TEST_TIME_STEPS = 200 

    eval_file = os.path.join(config.ARC_PATH, 'arc-agi_solutions.json')
    if not os.path.exists(eval_file):
         print(f"'{eval_file}' not found. Evaluating on training set as a demonstration.")
         eval_file = os.path.join(config.ARC_PATH, 'arc-agi_training_challenges.json')

    evaluate_with_shared_weights(eval_file, fine_tune_steps=TEST_TIME_STEPS)