# Assumptions and Default Values

This document lists assumptions made when parameters or models were not explicitly defined in the provided research papers.

- **Bandwidth**: The channel bandwidth for all users (CUE and D2D) is assumed to be 5 MHz. This is a common value in LTE simulations and is mentioned in Noman et al., "FeDRL-D2D" (Table 3).

- **Receiver Power Consumption**: The power consumed by the receiver circuitry (P_rx) is not always specified. We assume a fixed P_rx of 100 mW for all devices when calculating energy efficiency, a common simplification. This is configurable in `configs/base_config.yaml`.

- **Mobility Model**: The default model is static. Users are placed randomly at the beginning of each episode and do not move. This is consistent with the initial problem setup in many of the reference papers and provides a controlled environment for thesis-level experiments.

- **Resource Block (RB) Allocation**: We assume each Cellular User (CUE) is allocated an orthogonal channel. Each D2D pair (DDP) underlays one of these CUEs, meaning it shares the same channel. The mapping of DDP to CUE channel is part of the agent's action space or can be fixed depending on the scenario.

- **Action Space Discretization**: The continuous range of power transmission is discretized into 5 levels: [0, 0.25, 0.5, 0.75, 1.0] of P_max. This is a common technique to make the problem tractable for DQN and is specified as a default in the prompt.