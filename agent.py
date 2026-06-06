"""
Double DQN Agent with Prioritized Experience Replay.
"""

import numpy as np
import torch
import torch.optim as optim
from network import QNetwork
from replay import PrioritizedReplayBuffer
from config import Config


class DoubleDQNAgent:
    """
    Encapsulates:
      - online and target Q-networks
      - epsilon-greedy action selection
      - Double DQN learning step
      - target network syncing
    """

    def __init__(self, n_frames, n_actions, config, device="cpu"):
        """
        Args:
            n_frames:  int, number of stacked frames
            n_actions: int, number of possible actions
            config:    Config object with hyperparameters
            device:    str, "cpu" or "cuda"
        """
        self.n_actions = n_actions
        self.config = config
        self.device = device
        self.step_count = 0

        # Networks
        self.online_net = QNetwork(n_frames, n_actions).to(device)
        self.target_net = QNetwork(n_frames, n_actions).to(device)
        self.sync_target()

        # Optimiser
        self.optimiser = optim.Adam(
            self.online_net.parameters(), lr=config.LEARNING_RATE
        )

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.BUFFER_SIZE,
            alpha=config.PER_ALPHA,
            beta_start=config.PER_BETA_START,
            beta_end=config.PER_BETA_END,
            beta_steps=config.PER_BETA_STEPS,
            epsilon=config.PER_EPSILON,
        )

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        return self.target_net.eval()

    def get_epsilon(self):
        """
        Compute current epsilon from the linear decay schedule.
        Returns:
            float
        """
        epsilon = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * max(0, (self.config.EPS_DECAY_STEPS - self.step_count) / self.config.EPS_DECAY_STEPS)
        return max(self.config.EPS_END, epsilon)

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        Args:
            state: np.ndarray of shape (n_frames, 84, 84)
        Returns:
            int, chosen action
        """
        epsilon = self.get_epsilon()
        self.step_count += 1
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Push a transition into the PER buffer."""
        self.buffer.store(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a prioritised batch, compute Double DQN targets,
        update the network, and update PER priorities.

        Double DQN target:
            a* = argmax_a Q_online(s', a)
            y  = r + gamma * Q_target(s', a*) * (1 - done)

        Returns:
            float, mean loss (for logging), or None if not enough samples
        """
        states, actions, rewards, next_states, dones, indices, is_weights = self.buffer.sample(self.config.BATCH_SIZE)
        if states is None:
            return None
            
        # Convert everything to PyTorch tensors first, then move to the device
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        current_q = self.online_net(states).gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_state_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_state_actions).squeeze(1)
            target_q = rewards + (self.config.GAMMA * next_q_values * (1 - terminated))
            
        is_weights = np.array(is_weights, dtype=np.float32)
        weights_tensor = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        
        # Update PER priorities based on temporal-difference error
        td_errors = torch.abs(current_q - target_q).detach()
        td_errors_np = td_errors.cpu().numpy()
        self.buffer.update_priorities(indices, td_errors_np + self.config.PER_EPSILON)

        # Compute element-wise loss directly across the tensors weighted by PER importance sampling
        elementwise_loss = (current_q - target_q).pow(2)
        loss_val = torch.mean(weights_tensor * elementwise_loss)
        
        self.optimiser.zero_grad()
        loss_val.backward()
        self.optimiser.step()
        
        return loss_val.item()

    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        """Load model weights from disk safely across CPU/GPU targets."""
        # map_location ensures hardware cross-compatibility
        device_loc = torch.device(self.device)
        self.online_net.load_state_dict(torch.load(path, map_location=device_loc))
        if hasattr(self, "target_net"):
            self.target_net.load_state_dict(torch.load(path, map_location=device_loc))