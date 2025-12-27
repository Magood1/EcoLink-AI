"""
Final evaluation script for thesis-level results.
Loads a trained agent, evaluates it and baselines over multiple seeds,
calculates detailed communication metrics, prints a summary table,
and generates a comparative plot.
This version includes robust, hardware-aware device selection and model loading.
"""
# evaluate.py
"""
Final, production-ready evaluation script for thesis-level results.
This version correctly evaluates all baselines, including the stateful Proportional Fair scheduler,
calculates detailed communication metrics, and generates a comprehensive set of plots.
"""
import argparse
import yaml
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

from src.config_schema import ExperimentConfig
from src.envs.d2d_env import D2DEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.baselines import run_pf_evaluation # Import the dedicated PF evaluation function
from src.utils.metrics import calculate_energy_efficiency
from src.envs.physics import db_to_linear

def run_stateless_evaluation(env: D2DEnv, policy_func: Callable, num_episodes: int, base_seed: int) -> pd.DataFrame:
    """
    Runs an evaluation loop for stateless policies (Random, Greedy, and trained DQN agents).
    A stateless policy's action depends only on the current state.
    """
    results = []
    policy_name = getattr(policy_func, '__name__', 'policy')
    
    for i in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
        state, _ = env.reset(seed=base_seed + i)
        done = False
        
        episode_reward, episode_ddp_throughputs, episode_ddp_powers, episode_cue_outages = 0.0, [], [], 0
        num_steps = 0

        while not done:
            action = policy_func(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Collect metrics from the 'info' dictionary returned by the environment
            episode_reward += reward
            episode_ddp_throughputs.append(sum(info["ddp_throughputs"]))
            episode_ddp_powers.append(np.sum(info["ddp_powers"]))
            qos_thr_linear = db_to_linear(env.config.reward_config.qos_threshold_db)
            episode_cue_outages += sum(1 for sinr in info["cue_sinrs"] if sinr < qos_thr_linear)
            num_steps += 1
        
        results.append({
            "reward": episode_reward,
            "throughput_mbps": np.mean(episode_ddp_throughputs) / 1e6,
            "energy_efficiency_bits_per_joule": calculate_energy_efficiency(np.sum(episode_ddp_throughputs), np.sum(episode_ddp_powers)),
            "cue_outage_rate": episode_cue_outages / (num_steps * env.config.num_cues),
        })
        
    return pd.DataFrame(results)

def main(config: ExperimentConfig, model_path: str, num_seeds: int):
    """Main function to orchestrate the final evaluation suite."""
    device = torch.device("cpu") # Evaluation is fast enough on CPU
    print(f"Using device: {device}")
    
    env = D2DEnv(config.env)
    
    # --- 1. Define All Policies to Evaluate ---
    
    # DQN Agent Policy
    agent = DQNAgent(env.observation_space, env.action_space, config)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    dqn_policy = lambda state: agent.select_action(state, epsilon=0.0)
    dqn_policy.__name__ = "DQN_Agent"
    
    # Baselines Policies
    random_policy = lambda state: env.action_space.sample()
    random_policy.__name__ = "Random_Policy"
    
    greedy_action = (config.env.power_levels - 1) * np.ones(config.env.num_ddps, dtype=int)
    greedy_policy = lambda state: greedy_action
    greedy_policy.__name__ = "Greedy_Policy"

    # --- 2. Run All Evaluations ---
    all_results_df = pd.DataFrame()
    
    # Evaluate stateless policies
    stateless_policies = [dqn_policy, greedy_policy, random_policy]
    for i, policy in enumerate(stateless_policies):
        df = run_stateless_evaluation(env, policy, num_seeds, base_seed=10000 * (i + 1))
        df['agent'] = policy.__name__
        all_results_df = pd.concat([all_results_df, df], ignore_index=True)
        
    # **CRITICAL FIX**: Evaluate the stateful Proportional Fair policy using its dedicated function
    pf_df = run_pf_evaluation(env, num_seeds, base_seed=40000)
    pf_df['agent'] = "Proportional_Fair"
    all_results_df = pd.concat([all_results_df, pf_df], ignore_index=True)

    # --- 3. Print and Save Summary ---
    summary = all_results_df.groupby('agent').mean()
    summary_std = all_results_df.groupby('agent').std()
    
    print("\n--- Final Evaluation Summary (Mean Values) ---")
    print(summary)
    
    log_dir = os.path.dirname(model_path)
    summary.to_csv(os.path.join(log_dir, "final_evaluation_summary.csv"), index=False)
    
    # --- 4. Generate and Save Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Performance Comparison Across All Policies", fontsize=16)
    
    # Ensure a consistent order for plotting
    agent_order = ["DQN_Agent", "Proportional_Fair", "Greedy_Policy", "Random_Policy"]
    
    metrics_to_plot = list(summary.columns)
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes.flatten()[i]
        means = summary.loc[agent_order, metric]
        stds = summary_std.loc[agent_order, metric]
        
        means.plot(kind='bar', yerr=stds, ax=ax, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(log_dir, "final_performance_comparison.png")
    plt.savefig(plot_path)
    print(f"\nFinal comparison plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN agent against all baselines.")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved model (.pth file).")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of random seeds for evaluation.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ExperimentConfig(**config_dict)
        
    main(config, args.model_path, args.num_seeds)