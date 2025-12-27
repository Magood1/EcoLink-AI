#src/envs/d2d_env.py
"""
Final, robust Gymnasium environment for D2D resource allocation.
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
        self.cue_p_max_watts = db_to_linear(self.config.cue_tx_power_max_dbm - 30)
        self.ddp_p_max_watts = db_to_linear(self.config.ddp_tx_power_max_dbm - 30)
        self.ddp_power_options = np.linspace(0, self.ddp_p_max_watts, self.config.power_levels)
        self.action_space = spaces.MultiDiscrete([self.config.power_levels] * self.config.num_ddps)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.num_ddps, 3), dtype=np.float32)
        self.current_step = 0
        self.bs_pos = np.array([self.config.area_size / 2, self.config.area_size / 2])
        self.cue_pos, self.ddp_tx_pos, self.ddp_rx_pos = None, None, None

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.cue_pos = self.np_random.random((self.config.num_cues, 2)) * self.config.area_size
        self.ddp_tx_pos = self.np_random.random((self.config.num_ddps, 2)) * self.config.area_size
        radii = self.np_random.random(self.config.num_ddps) * 50
        angles = self.np_random.random(self.config.num_ddps) * 2 * np.pi
        offsets = np.array([radii * np.cos(angles), radii * np.sin(angles)]).T
        self.ddp_rx_pos = np.clip(self.ddp_tx_pos + offsets, 0, self.config.area_size)
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.current_step += 1
        ddp_tx_powers = self.ddp_power_options[action]
        all_throughputs, cue_sinrs = self._calculate_throughputs(ddp_tx_powers)
        total_ddp_throughput = sum(all_throughputs)
        total_ddp_power_consumed = np.sum(ddp_tx_powers)
        reward = self._calculate_reward(
            total_ddp_throughput, total_ddp_power_consumed, cue_sinrs
        )
        terminated = False
        truncated = self.current_step >= self.config.episode_length
        info = {
            "ddp_throughputs": all_throughputs,
            "ddp_powers": ddp_tx_powers,
            "cue_sinrs": cue_sinrs,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_throughputs(self, ddp_tx_powers: np.ndarray) -> Tuple[List[float], List[float]]:
        interference_at_bs = sum(
            ddp_tx_powers[i] * db_to_linear(-path_loss_noman2024(np.linalg.norm(self.ddp_tx_pos[i] - self.bs_pos) / 1000, is_d2d=False))
            for i in range(self.config.num_ddps)
        )
        cue_sinrs = [
            calculate_sinr(self.cue_p_max_watts, db_to_linear(-path_loss_noman2024(np.linalg.norm(self.cue_pos[i] - self.bs_pos) / 1000, is_d2d=False)), interference_at_bs, self.noise_watts)
            for i in range(self.config.num_cues)
        ]
        ddp_throughputs = []
        for i in range(self.config.num_ddps):
            signal_gain = db_to_linear(-path_loss_noman2024(np.linalg.norm(self.ddp_tx_pos[i] - self.ddp_rx_pos[i]) / 1000, is_d2d=True))
            interference_at_ddp_rx = 0.0
            for j in range(self.config.num_ddps):
                if i == j: continue
                interference_at_ddp_rx += ddp_tx_powers[j] * db_to_linear(-path_loss_noman2024(np.linalg.norm(self.ddp_tx_pos[j] - self.ddp_rx_pos[i]) / 1000, is_d2d=True))
            cue_idx = i % self.config.num_cues
            interference_at_ddp_rx += self.cue_p_max_watts * db_to_linear(-path_loss_noman2024(np.linalg.norm(self.cue_pos[cue_idx] - self.ddp_rx_pos[i]) / 1000, is_d2d=False))
            sinr = calculate_sinr(ddp_tx_powers[i], signal_gain, interference_at_ddp_rx, self.noise_watts)
            ddp_throughputs.append(shannon_throughput(sinr, self.config.bandwidth))
        return ddp_throughputs, cue_sinrs

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.config.num_ddps, 3), dtype=np.float32)
        for i in range(self.config.num_ddps):
            obs[i, 0] = 1.0 - self._normalize_db(path_loss_noman2024(np.linalg.norm(self.ddp_tx_pos[i] - self.ddp_rx_pos[i]) / 1000, is_d2d=True))
            obs[i, 1] = 1.0 - self._normalize_db(path_loss_noman2024(np.linalg.norm(self.ddp_tx_pos[i] - self.bs_pos) / 1000, is_d2d=False))
            cue_idx = i % self.config.num_cues
            obs[i, 2] = 1.0 - self._normalize_db(path_loss_noman2024(np.linalg.norm(self.cue_pos[cue_idx] - self.ddp_rx_pos[i]) / 1000, is_d2d=False))
        return obs

    def _normalize_db(self, db_val: float, min_db: float = 60.0, max_db: float = 160.0) -> float:
        clipped_db = np.clip(db_val, min_db, max_db)
        return (clipped_db - min_db) / (max_db - min_db)

    def _calculate_reward(self, ddp_throughput: float, ddp_power: float, cue_sinrs: List[float]) -> float:
        cfg = self.config.reward_config
        
        # Positive components (what we want to maximize)
        throughput_mbps = ddp_throughput / 1e6
        ee = throughput_mbps / (ddp_power + 1e-9)
        log_ee_reward = np.log1p(ee)
        log_tp_reward = np.log1p(throughput_mbps)
        
        # Negative component (what we want to minimize)
        qos_thr_linear = db_to_linear(cfg.qos_threshold_db)
        raw_penalty = sum(max(0.0, qos_thr_linear - sinr) for sinr in cue_sinrs)
        penalty = min(raw_penalty * cfg.interference_penalty_scale, cfg.max_penalty)
        
        # Weighted combination
        reward = (cfg.reward_ee_weight * log_ee_reward + 
                  cfg.reward_throughput_weight * log_tp_reward - 
                  penalty)

        # Final scaling to a consistent range (e.g., -5 to +5)
        return float(np.tanh(reward / cfg.tanh_scale) * cfg.tanh_scale)