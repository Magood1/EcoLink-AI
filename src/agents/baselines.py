"""
Baseline policies for comparison against DRL agents.
Includes Random, Greedy, and a stateful Proportional Fair (PF) scheduler.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from src.envs.d2d_env import D2DEnv
from src.utils.metrics import calculate_energy_efficiency
from src.envs.physics import db_to_linear

class ProportionalFairScheduler:
    """
    A stateful scheduler that implements a simplified Proportional Fair baseline.
    It allocates power to the D2D user that has the best instantaneous channel
    quality relative to its own historical average.
    """
    def __init__(self, num_ddps: int, alpha: float = 0.1):
        self.num_ddps = num_ddps
        self.alpha = alpha  # Smoothing factor for the moving average throughput
        self.avg_throughputs = np.ones(num_ddps) * 1e-6  # Initialize with a small value

    def reset(self):
        """Resets the historical average for a new episode."""
        self.avg_throughputs.fill(1e-6)

    def select_action(self, env: D2DEnv) -> np.ndarray:
        """Selects the best power level for each DDP based on the PF metric."""
        best_actions = np.zeros(self.num_ddps, dtype=int)
        for i in range(self.num_ddps):
            best_metric = -1.0
            best_power_idx = 0
            # Iterate through all possible power levels to find the best one for this agent
            for p_idx, power in enumerate(env.ddp_power_options):
                # Create a temporary power vector to simulate this single choice
                temp_powers = np.zeros(self.num_ddps)
                temp_powers[i] = power
                
                # Estimate throughput for this specific action
                throughputs, _ = env._calculate_throughputs(temp_powers)
                potential_rate = throughputs[i]
                
                # The PF metric: potential_rate / historical_average_rate
                metric = potential_rate / self.avg_throughputs[i]
                
                if metric > best_metric:
                    best_metric = metric
                    best_power_idx = p_idx
            best_actions[i] = best_power_idx
        return best_actions
    
    def update(self, throughputs: List[float]):
        """Updates the historical average throughputs using an exponential moving average."""
        self.avg_throughputs = (1 - self.alpha) * self.avg_throughputs + self.alpha * np.array(throughputs)

def run_pf_evaluation(env: D2DEnv, num_episodes: int, base_seed: int) -> pd.DataFrame:
    """
    A dedicated evaluation loop for the stateful Proportional Fair scheduler.
    It correctly handles the scheduler's internal state across steps in an episode.
    """
    pf_scheduler = ProportionalFairScheduler(env.config.num_ddps)
    results = []
    
    for i in tqdm(range(num_episodes), desc="Evaluating Proportional_Fair"):
        pf_scheduler.reset()
        state, _ = env.reset(seed=base_seed + i)
        done = False
        
        episode_reward = 0.0
        episode_ddp_throughputs, episode_ddp_powers, episode_cue_outages = [], [], 0
        num_steps = 0

        while not done:
            action = pf_scheduler.select_action(env)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # **Crucial**: Update PF scheduler's internal state with the latest throughputs
            pf_scheduler.update(info["ddp_throughputs"])

            # Collect metrics for this step
            episode_reward += reward
            episode_ddp_throughputs.append(sum(info["ddp_throughputs"]))
            episode_ddp_powers.append(np.sum(info["ddp_powers"]))
            qos_thr_linear = db_to_linear(env.config.reward_config.qos_threshold_db)
            episode_cue_outages += sum(1 for sinr in info["cue_sinrs"] if sinr < qos_thr_linear)
            num_steps += 1
        
        # Aggregate and store results for the completed episode
        results.append({
            "reward": episode_reward,
            "throughput_mbps": np.mean(episode_ddp_throughputs) / 1e6,
            "energy_efficiency_bits_per_joule": calculate_energy_efficiency(np.sum(episode_ddp_throughputs), np.sum(episode_ddp_powers)),
            "cue_outage_rate": episode_cue_outages / (num_steps * env.config.num_cues),
        })
        
    return pd.DataFrame(results)