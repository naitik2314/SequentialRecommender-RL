# models/dqn_agent.py

import random
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from models.network import QNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)

class DQNAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        device='cpu',
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=10_000,
        target_update_freq=1000,
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # networks
        self.q_net = QNetwork(state_dim, num_actions).to(device)
        self.target_net = QNetwork(state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # epsilon schedule
        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay_steps
        self.steps_done = 0

    def select_action(self, state):
        """
        Epsilon-greedy action selection from state (numpy array).
        """
        self.steps_done += 1
        # linear decay
        self.epsilon = max(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.eps_decay)
        )
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_vals = self.q_net(s)
                return int(q_vals.argmax(dim=1).item())

    def store_transition(self, *args):
        self.buffer.push(*args)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        # sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q(s,a)
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0]
        # TD target
        td_target = rewards + self.gamma * q_next * (1 - dones)
        loss = F.mse_loss(q_vals, td_target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
