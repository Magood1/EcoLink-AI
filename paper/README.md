#data/paper/readme.md:

Paper Implementation Notes

This file documents how key concepts from the reference papers were translated into the codebase.

Noman et al., "FeDRL-D2D: Federated Deep Reinforcement Learning..."

Path Loss Models: The path loss models for Macro Base Station (MBS) and D2D links are implemented directly from Table 3.

MBS to UE: 128.1 + 37.6 * log10(d_km)

D2D Tx to Rx: 148.0 + 40.0 * log10(d_km)

These are implemented in src.envs.physics.path_loss_d2d.

Noise Power: The noise power of -174 dBm/Hz is used. The total noise in Watts is calculated as 10**((-174 - 30) / 10) * bandwidth. See src.envs.D2DEnv.

SINR Equations: The SINR formulations for cellular uplink (Eq. 1) and D2D links (Eq. 2) are the basis for the implementation in src.envs.physics.calculate_sinr.

Metrics: Energy Efficiency (EE) as defined in Eq. 6 is a primary reward function option. Cellular Outage Probability (Eq. 7) is implemented as an evaluation metric in src.utils.metrics.

Pan and Yang, "Deep Reinforcement Learning-Based Optimization Method for D2D..."

Objective Function: The concept of maximizing Energy Efficiency (EE), defined as sum(Rate) / sum(Power), is a core principle adopted in our default reward function. See src.envs.D2DEnv._calculate_reward.

Hyperparameters: The Double DQN hyperparameters listed in Section V.A serve as the basis for our default agent configuration in configs/base_config.yaml.

learning_rate: 2.5e-4

discount_factor: 0.99

batch_size: 32 (we use 64 as a more modern default)

target_update_frequency: 100 steps (we use a soft update tau instead, a common modern practice).

Mishra, "Artificial Intelligence Assisted Enhanced Energy Efficient Model..."

Metrics: The emphasis on metrics like throughput, delay, fairness, and energy efficiency guided the selection of metrics implemented in src.utils.metrics. We specifically implemented Jain's Fairness Index based on its common use for evaluating resource distribution fairness.
