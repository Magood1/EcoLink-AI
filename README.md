EcoLink-AI: Autonomous 5G Resource Orchestration Framework
A production-grade simulation framework optimizing Energy Efficiency (EE) and QoS in Device-to-Device (D2D) underlay networks using Deep Reinforcement Learning (DQN).
Executive Summary
As 5G/6G networks face exponential device growth, static resource allocation methods (like Convex Optimization) fail to handle the non-linear dynamics of interference and power consumption. EcoLink-AI addresses this by deploying an autonomous Deep Q-Network (DQN) agent capable of learning optimal power control policies in real-time.
Unlike standard academic scripts, this project is engineered as a modular, containerized system. It demonstrates a 6x improvement in Energy Efficiency compared to greedy algorithms while maintaining Cellular User (CUE) outage rates below 26%, proving that AI can effectively balance the trade-off between spectral throughput and green communication.
Core Engineering Pillars:
Scalable Architecture: Modular design separating Physics, Environment, and Agent logic.
Type Safety: Robust configuration validation using Pydantic.
Reproducibility: Deterministic seeding, Dockerized environment, and comprehensive logging.
Technical Architecture & Key Features
1. The AI Engine (DQN Agent)
Algorithm: Implements Double DQN to mitigate Q-value overestimation, utilizing Huber Loss (SmoothL1) for gradient stability against outliers.
Stability: Features Gradient Clipping and Soft Target Updates (Polyak averaging) to ensure convergence in high-variance wireless environments.
State Space: Processes complex environmental data (SINR, Channel Gains, Interference Vectors) normalized for Neural Network stability.
2. High-Fidelity Physics Simulation (src/envs)
Custom Gymnasium Environment: Fully compliant with OpenAI Gym API.
Realistic Channel Modeling: Implements Noman et al. (2024) Path Loss models for both Cellular and D2D links.
Reward Engineering: sophisticated multi-objective reward function using tanh scaling to balance Energy Efficiency (bits/Joule) vs. QoS Penalty (Outage).
3. Backend Best Practices
Configuration Management: Uses YAML coupled with Pydantic schemas (src/config_schema.py) to enforce strict type checking on experiment parameters before runtime.
Dependency Management: Utilizes Poetry for deterministic dependency resolution.
Containerization: Full Dockerfile support for deploying the training pipeline in isolated environments.
Performance Results
![System Performance Comparison](assets/final_performance_comparison.png)
The system was evaluated against industry-standard baselines: Greedy (Max Power), Random, and Proportional Fair (PF).
Metric  EcoLink-AI (DQN)  Greedy Policy  Improvement
Energy Efficiency  2.17 Gbits/Joule  0.33 Gbits/Joule  ~6.5x Increase
CUE Outage Rate  26%  91.5%  71% Reduction
Mean Reward  573.7  32.9  Stable Convergence
Insight: While the Greedy policy maximizes raw throughput, it causes catastrophic interference (91% outage). The AI Agent learns a "Smart Conservative" strategy, sacrificing marginal throughput to maximize network stability and energy savings.
Quick Start
Option A: Using Docker (Recommended)
Run the entire training and evaluation pipeline in a container to guarantee reproducibility.
code
Bash
# Build the image
docker build -t ecolink-ai .

# Run the Smoke Test (Quick verification)
docker run ecolink-ai python train.py --config configs/smoke_test.yaml
Option B: Local Setup (Poetry/Pip)
Prerequisites: Python 3.10+, Poetry (optional)
code
Bash
# 1. Clone Repository
git clone https://github.com/Magood1/EcoLink-AI.git
cd d2d_rl

# 2. Install Dependencies (via Poetry)
poetry install
# OR (via pip)
pip install -r requirements.txt

# 3. Run Training (Base Configuration)
python train.py --config config/base_config.yaml
# 4. Run Evaluation (Compare with Baselines)
python evaluate.py --config config/base_config.yaml --model-path data/dqn_best.pth --num-seeds 10
System Workflow
The project follows a rigorous data flow pipeline:
code
Mermaid
graph TD
    A[Config (YAML + Pydantic)] -->|Validate| B(Experiment Setup);
    B --> C[D2D Environment (Physics Engine)];
    C -->|State (SINR, Gains)| D[DQN Agent];
    D -->|Action (Power Level)| C;
    C -->|Reward (EE - Penalty)| D;
    D -->|Store Transition| E[Replay Buffer];
    E -->|Batch Sample| F[Optimization Step (PyTorch)];
    F --> G[Checkpoints & TensorBoard Logs];

# Project Structure
d2d_rl/
├── config/              # YAML Configs & Smoke Tests
├── src/
│   ├── agents/          # DQN Logic & Baselines (Proportional Fair)
│   ├── envs/            # Gymnasium Environment & Physics Models
│   ├── utils/           # Metrics (Jain's Fairness, EE) & Logging
│   └── config_schema.py # Pydantic Validation Schemas
├── Dockerfile           # Containerization setup
├── pyproject.toml       # Poetry Dependency Management
├── train.py             # Training Entry Point
├── evaluate.py          # Evaluation & Visualization Pipeline
└── README.md            # Documentation
# Tech Stack
Core: Python 3.10
ML Framework: PyTorch
Environment: Gymnasium (OpenAI), NumPy
Data & Viz: Pandas, Matplotlib, Seaborn
DevOps: Docker, Poetry, Pydantic
# Contributing
This project is open-source. Engineering improvements (e.g., migrating to Multi-Agent PPO, implementing Graph Neural Networks) are welcome via Pull Requests.
Fork the Project
Create your Feature Branch (git checkout -b feature/GNN-Architecture)
Commit your Changes (git commit -m 'Add GNN support')
Push to the Branch (git push origin feature/GNN-Architecture)
Open a Pull Request
# Author
Magood1
Backend & AI Engineer
Focus: Scalable AI Systems, Wireless Network Optimization, and Python Software Architecture.
