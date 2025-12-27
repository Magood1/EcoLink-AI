# tiny_d2d_rl_saba/src/config_schema.py
"""
Pydantic schemas for validating and parsing experiment configuration files.
Updated to match configs/base_config.yaml and to include newly added reward fields.
"""
from pydantic import BaseModel
from typing import List

class EnvRewardConfig(BaseModel):
    # Reward shaping / scaling parameters (defaults mirror configs/base_config.yaml)
    reward_ee_weight: float = 1.0
    reward_throughput_weight: float = 0.0
    interference_penalty_scale: float = 1.0   # NEW: scales QoS violation penalty
    max_penalty: float = 5.0
    qos_threshold_db: float = 5.0
    tanh_scale: float = 5.0                    # NEW: final tanh scaling to bound reward

class EnvConfig(BaseModel):
    num_cues: int
    num_ddps: int
    area_size: int
    bandwidth: float
    noise_dbm_per_hz: float
    cue_tx_power_max_dbm: float
    ddp_tx_power_max_dbm: float
    episode_length: int
    power_levels: int
    reward_config: EnvRewardConfig

class AgentNetworkConfig(BaseModel):
    hidden_dims: List[int] = [256, 256]

class AgentConfig(BaseModel):
    name: str = "dqn"
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    learning_rate: float = 0.0001     # default from base_config
    buffer_size: int = 100000         # default from base_config
    batch_size: int = 64              # default from base_config
    tau: float = 0.005
    use_double_dqn: bool = True
    use_dueling_dqn: bool = False
    grad_clip_norm: float = 10.0
    network: AgentNetworkConfig

class TrainingConfig(BaseModel):
    seed: int = 42
    device: str = "cuda"               # default in base_config.yaml (override in smoke_test if needed)
    num_episodes: int = 2000
    eval_freq: int = 50
    eval_episodes: int = 20
    results_dir: str = "data"
    model_filename: str = "dqn_best.pth"

class ExperimentConfig(BaseModel):
    env: EnvConfig
    agent: AgentConfig
    training: TrainingConfig
