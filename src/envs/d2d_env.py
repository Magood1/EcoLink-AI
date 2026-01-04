# src/envs/d2d_env.py
"""
Final, robust Gymnasium environment for D2D resource allocation.
Optimized with Vectorized Operations for High Scalability.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple

from src.config_schema import EnvConfig
from src.envs.physics import (
    path_loss_noman2024,
    calculate_noise_watts,
    calculate_sinr,
    shannon_throughput,
    db_to_linear,
)

class D2DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        self.noise_watts = calculate_noise_watts(self.config.noise_dbm_per_hz, self.config.bandwidth)
        
        # Pre-calculate linear power limits
        self.cue_p_max_watts = db_to_linear(self.config.cue_tx_power_max_dbm - 30)
        self.ddp_p_max_watts = db_to_linear(self.config.ddp_tx_power_max_dbm - 30)
        
        # Action & Observation Spaces
        self.ddp_power_options = np.linspace(0, self.ddp_p_max_watts, self.config.power_levels)
        self.action_space = spaces.MultiDiscrete([self.config.power_levels] * self.config.num_ddps)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.num_ddps, 3), dtype=np.float32)
        
        # State variables
        self.current_step = 0
        self.bs_pos = np.array([self.config.area_size / 2, self.config.area_size / 2])
        self.cue_pos = None
        self.ddp_tx_pos = None
        self.ddp_rx_pos = None

        # Pre-compute CUE mapping indices for vectorization (0, 1, 2, 3, 0, 1...)
        self.cue_indices = np.arange(self.config.num_ddps) % self.config.num_cues

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. Random Placement
        self.cue_pos = self.np_random.random((self.config.num_cues, 2)) * self.config.area_size
        self.ddp_tx_pos = self.np_random.random((self.config.num_ddps, 2)) * self.config.area_size
        
        # 2. D2D Rx Placement (within 50m radius)
        radii = self.np_random.random(self.config.num_ddps) * 50
        angles = self.np_random.random(self.config.num_ddps) * 2 * np.pi
        offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        self.ddp_rx_pos = np.clip(self.ddp_tx_pos + offsets, 0, self.config.area_size)
        
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        
        # Convert discrete actions to power values
        ddp_tx_powers = self.ddp_power_options[action] # Shape: (num_ddps,)
        
        # Vectorized calculation
        all_throughputs, cue_sinrs = self._calculate_throughputs_vectorized(ddp_tx_powers)
        
        # Aggregation
        total_ddp_throughput = np.sum(all_throughputs)
        total_ddp_power_consumed = np.sum(ddp_tx_powers)
        
        # Reward
        reward = self._calculate_reward(total_ddp_throughput, total_ddp_power_consumed, cue_sinrs)
        
        terminated = False
        truncated = self.current_step >= self.config.episode_length
        
        info = {
            "ddp_throughputs": all_throughputs.tolist(), # Convert back to list for compatibility
            "ddp_powers": ddp_tx_powers.tolist(),
            "cue_sinrs": cue_sinrs.tolist(),
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_throughputs_vectorized(self, ddp_tx_powers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fully vectorized SINR and Throughput calculation. 
        Replaces nested loops with matrix operations.
        """
        # --- 1. Distances (Matrix Operations) ---
        # Shape: (num_ddps, 1, 2) - (1, num_ddps, 2) -> (num_ddps, num_ddps)
        # d_ddp_tx_rx_matrix[i, j] = distance from Tx 'i' to Rx 'j'
        d_ddp_tx_rx_matrix = np.linalg.norm(self.ddp_tx_pos[:, None, :] - self.ddp_rx_pos[None, :, :], axis=2) / 1000.0
        
        # Diagonals are the signal links (Tx 'i' to Rx 'i')
        d_signal = np.diag(d_ddp_tx_rx_matrix)
        
        # BS Distances
        d_ddp_tx_bs = np.linalg.norm(self.ddp_tx_pos - self.bs_pos, axis=1) / 1000.0
        d_cue_bs = np.linalg.norm(self.cue_pos - self.bs_pos, axis=1) / 1000.0
        
        # CUE to D2D Rx Distances (Shape: num_cues, num_ddps)
        d_cue_ddp_rx = np.linalg.norm(self.cue_pos[:, None, :] - self.ddp_rx_pos[None, :, :], axis=2) / 1000.0

        # --- 2. Channel Gains (Linear) ---
        # Note: path_loss_noman2024 works element-wise on numpy arrays thanks to numpy broadcasting
        g_ddp_tx_rx_matrix = db_to_linear(-path_loss_noman2024(d_ddp_tx_rx_matrix, is_d2d=True))
        g_signal = np.diag(g_ddp_tx_rx_matrix)
        
        g_ddp_tx_bs = db_to_linear(-path_loss_noman2024(d_ddp_tx_bs, is_d2d=False))
        g_cue_bs = db_to_linear(-path_loss_noman2024(d_cue_bs, is_d2d=False))
        g_cue_ddp_rx = db_to_linear(-path_loss_noman2024(d_cue_ddp_rx, is_d2d=False))

        # --- 3. Interference & SINR Calculation ---
        
        # A. CUE SINR Calculation
        # Interference at BS from ALL D2D users = sum(P_d2d * Gain_d2d_to_BS)
        interference_at_bs = np.sum(ddp_tx_powers * g_ddp_tx_bs)
        signal_cue = self.cue_p_max_watts * g_cue_bs
        cue_sinrs = signal_cue / (interference_at_bs + self.noise_watts)

        # B. D2D SINR Calculation
        # Interference from other D2Ds:
        # We need sum(P_j * G_ij) for j != i.
        # We can do (P dot G_full_matrix) - (P * G_diagonal)
        # Note: We transpose G because we want effect of all Tx on one Rx (columns)
        total_d2d_interference = np.dot(ddp_tx_powers, g_ddp_tx_rx_matrix) 
        self_signal_power = ddp_tx_powers * g_signal
        
        # Subtract self-signal from total received power to get interference
        inter_d2d_interference = total_d2d_interference - self_signal_power
        
        # Interference from paired CUE
        # Each D2D 'i' is affected by CUE 'cue_indices[i]'
        # We use advanced indexing to pick the specific CUE gain for each D2D Rx
        inter_cue_interference = self.cue_p_max_watts * g_cue_ddp_rx[self.cue_indices, np.arange(self.config.num_ddps)]

        ddp_sinrs = calculate_sinr(
            tx_power_watts=ddp_tx_powers, # Broadcasting handles this? No, calculate_sinr is scalar logic mostly.
            channel_gain_linear=g_signal, # Use vectorized math directly below for safety
            interference_watts=inter_d2d_interference + inter_cue_interference,
            noise_watts=self.noise_watts
        )
        
        # Re-implement sinr/throughput vectorized to be safe
        ddp_sinrs = (ddp_tx_powers * g_signal) / (inter_d2d_interference + inter_cue_interference + self.noise_watts)
        ddp_throughputs = shannon_throughput(ddp_sinrs, self.config.bandwidth)

        return ddp_throughputs, cue_sinrs

    def _get_observation(self) -> np.ndarray:
        """Vectorized Observation Generation"""
        # 1. D2D Signal Path Loss
        d_signal = np.linalg.norm(self.ddp_tx_pos - self.ddp_rx_pos, axis=1) / 1000.0
        pl_signal = path_loss_noman2024(d_signal, is_d2d=True)
        
        # 2. D2D to BS Path Loss
        d_bs = np.linalg.norm(self.ddp_tx_pos - self.bs_pos, axis=1) / 1000.0
        pl_bs = path_loss_noman2024(d_bs, is_d2d=False)
        
        # 3. Interference Link Path Loss (CUE to D2D Rx)
        # Calculate distance from assigned CUE to D2D Rx
        # cue_pos shape: (num_cues, 2), accessing with indices array
        assigned_cues_pos = self.cue_pos[self.cue_indices] # Shape: (num_ddps, 2)
        d_inter = np.linalg.norm(assigned_cues_pos - self.ddp_rx_pos, axis=1) / 1000.0
        pl_inter = path_loss_noman2024(d_inter, is_d2d=False)

        # Construct Matrix
        obs = np.zeros((self.config.num_ddps, 3), dtype=np.float32)
        obs[:, 0] = 1.0 - self._normalize_db(pl_signal)
        obs[:, 1] = 1.0 - self._normalize_db(pl_bs)
        obs[:, 2] = 1.0 - self._normalize_db(pl_inter)
        
        return obs

    def _normalize_db(self, db_val: np.ndarray, min_db: float = 60.0, max_db: float = 160.0) -> np.ndarray:
        clipped_db = np.clip(db_val, min_db, max_db)
        return (clipped_db - min_db) / (max_db - min_db)

    def _calculate_reward(self, ddp_throughput: float, ddp_power: float, cue_sinrs: np.ndarray) -> float:
        cfg = self.config.reward_config
        
        throughput_mbps = ddp_throughput / 1e6
        ee = throughput_mbps / (ddp_power + 1e-9)
        log_ee_reward = np.log1p(ee)
        log_tp_reward = np.log1p(throughput_mbps)
        
        qos_thr_linear = db_to_linear(cfg.qos_threshold_db)
        # Vectorized sum of violations
        raw_penalty = np.sum(np.maximum(0.0, qos_thr_linear - cue_sinrs))
        penalty = min(raw_penalty * cfg.interference_penalty_scale, cfg.max_penalty)
        
        reward = (cfg.reward_ee_weight * log_ee_reward + 
                  cfg.reward_throughput_weight * log_tp_reward - 
                  penalty)

        return float(np.tanh(reward / cfg.tanh_scale) * cfg.tanh_scale)
