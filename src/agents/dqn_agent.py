# agents/dqn_agent.py
"""
DQN Agent with Parameter Sharing, Double DQN, Huber Loss, and Gradient Clipping.
This agent is designed for a multi-agent environment where all agents are homogeneous.
"""
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from src.config_schema import ExperimentConfig

class QNetwork(nn.Module):
    """Shared MLP Q-Network for all agents."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    """A multi-agent DQN that uses a single shared network for all agents."""
    def __init__(self, obs_space, action_space, config: ExperimentConfig):
        self.agent_config = config.agent
        self.training_config = config.training
        self.device = torch.device(self.training_config.device)

        self.obs_dim = obs_space.shape[1]
        self.action_dim = action_space.nvec[0]
        self.num_agents = obs_space.shape[0]

        # Parameter Sharing: One network for all agents
        self.policy_net = QNetwork(
            self.obs_dim, self.action_dim, self.agent_config.network.hidden_dims
        ).to(self.device)
        self.target_net = QNetwork(
            self.obs_dim, self.action_dim, self.agent_config.network.hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.agent_config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss() # Huber Loss is more robust to outliers

        # Experience Replay Buffer
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
        self.replay_buffer = deque(maxlen=self.agent_config.buffer_size)
        
        self._sync_networks()

    def _sync_networks(self):
        """Copies weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        """Selects an action for each agent using an epsilon-greedy policy."""
        actions = []
        for i in range(self.num_agents):
            if random.random() < epsilon:
                actions.append(random.randint(0, self.action_dim - 1))
            else:
                obs_tensor = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(obs_tensor)
                actions.append(q_values.argmax().item())
        return np.array(actions)

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.append(self.Transition(state, action, reward, next_state, done))

    def update(self) -> float | None:
        """Updates the Q-network using a batch of experiences from the replay buffer."""
        if len(self.replay_buffer) < self.agent_config.batch_size:
            return None

        transitions = random.sample(self.replay_buffer, self.agent_config.batch_size)
        batch = self.Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        total_loss = 0.0
        for i in range(self.num_agents):
            # Get Q-values for the actions that were actually taken
            q_values = self.policy_net(states[:, i, :]).gather(1, actions[:, i].unsqueeze(1)).squeeze(1)

            # Calculate target Q-values using the target network
            with torch.no_grad():
                if self.agent_config.use_double_dqn:
                    # Double DQN: select action with policy_net, evaluate with target_net
                    next_actions = self.policy_net(next_states[:, i, :]).argmax(1, keepdim=True)
                    next_q_values = self.target_net(next_states[:, i, :]).gather(1, next_actions).squeeze(1)
                else:
                    # Standard DQN
                    next_q_values = self.target_net(next_states[:, i, :]).max(1)[0]
            
            target_q_values = rewards + self.agent_config.gamma * next_q_values * (1 - dones)
            
            loss = self.loss_fn(q_values, target_q_values)
            total_loss += loss

        # Optimize the model
        self.optimizer.zero_grad()
        avg_loss = total_loss / self.num_agents
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.agent_config.grad_clip_norm)
        self.optimizer.step()

        # Soft update the target network
        tau = self.agent_config.tau
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        
        return avg_loss.item()