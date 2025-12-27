"""
Top-level script to orchestrate a full experimental run:
1. Train the agent.
2. Evaluate the best checkpoint against all baselines over multiple seeds.
"""
import os
import subprocess
import argparse
import yaml

def run_command(command: str):
    """Executes a shell command and prints its output."""
    print(f"\n--- Running Command ---\n{command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")

def main(config_path: str, num_eval_seeds: int):
    """Runs the full train-then-evaluate pipeline."""
    # --- 1. Training Phase ---
    train_command = f"python tiny_d2d_rl_saba/train.py --config {config_path}"
    run_command(train_command)
    
    # --- 2. Evaluation Phase ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run_name = f"{config['agent']['name']}_seed{config['training']['seed']}"
    model_dir = os.path.join(config['training']['results_dir'], run_name)
    model_path = os.path.join(model_dir, config['training']['model_filename'])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found after training: {model_path}")

    eval_command = (f"python tiny_d2d_rl_saba/evaluate.py --config {config_path} "
                    f"--model-path {model_path} "
                    f"--num-seeds {num_eval_seeds}")
    run_command(eval_command)
    
    print("\n--- Experiment Complete ---")
    print(f"All results, logs, and plots are saved in: {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full D2D-RL experiment.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to the experiment configuration file.")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds for final evaluation.")
    args = parser.parse_args()
    main(args.config, args.num_seeds)