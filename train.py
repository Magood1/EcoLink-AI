# train.py
"""
Main training script for D2D resource allocation agents.
This script handles configuration loading, robust device selection, training loop,
periodic evaluation, and saving of the best-performing model.
"""
import argparse
import yaml
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import random

from src.config_schema import ExperimentConfig
from src.envs.d2d_env import D2DEnv
from src.agents.dqn_agent import DQNAgent

def set_seed(seed: int):
    """Sets the random seed across all relevant libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(env: D2DEnv, agent: DQNAgent, num_episodes: int, base_seed: int) -> float:
    """Evaluates the agent's performance over several episodes with a deterministic policy."""
    all_rewards = []
    for i in range(num_episodes):
        state, _ = env.reset(seed=base_seed + i)
        done = False
        episode_reward = 0.0
        while not done:
            # Epsilon = 0.0 for deterministic evaluation
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
        all_rewards.append(episode_reward)
    return float(np.mean(all_rewards))

def train(config: ExperimentConfig):
    """Main training loop orchestrating the environment and agent."""
    # --- 1. Setup and Initialization ---

    # Robustly select the training device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        config.training.device = "cpu"

    set_seed(config.training.seed)
    env = D2DEnv(config.env)
    agent = DQNAgent(env.observation_space, env.action_space, config)

    run_name = f"{config.agent.name}_seed{config.training.seed}"
    log_dir = os.path.join(config.training.results_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"--- Starting Training ---")
    print(f"Device: {config.training.device}")
    print(f"Log directory: {log_dir}")

    # --- 2. Training Loop ---
    eps_start = config.agent.epsilon_start
    eps_end = config.agent.epsilon_end
    eps_decay_steps = config.agent.epsilon_decay_steps

    global_step = 0
    best_eval_reward = -float('inf')

    for episode in tqdm(range(config.training.num_episodes), desc="Training Progress"):
        state, _ = env.reset(seed=config.training.seed + episode)
        
        for step in range(config.env.episode_length):
            epsilon = np.interp(global_step, [0, eps_decay_steps], [eps_start, eps_end])
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            global_step += 1
            if done:
                break
        
        # --- 3. Periodic Evaluation and Checkpointing ---
        if (episode + 1) % config.training.eval_freq == 0:
            eval_reward = evaluate(
                env, agent, config.training.eval_episodes, base_seed=10000 + episode
            )
            
            tqdm.write(
                f"Episode {episode+1} | Step {global_step} | Eval Reward: {eval_reward:.2f} | Epsilon: {epsilon:.3f}"
            )

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                model_path = os.path.join(log_dir, config.training.model_filename)
                torch.save(agent.policy_net.state_dict(), model_path)
                tqdm.write(f"** New best model saved with reward: {best_eval_reward:.2f} **")

    print(f"--- Training Finished ---")
    print(f"Best evaluation reward achieved: {best_eval_reward:.2f}")
    final_model_path = os.path.join(log_dir, config.training.model_filename)
    print(f"Best model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for D2D resource allocation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate and parse the configuration using the Pydantic schema for type safety
    try:
        config = ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)
        
    train(config)