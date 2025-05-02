# models/dqn_agent.py
import random, collections, numpy as np, torch
import torch.nn.functional as F
from torch import optim
from models.network import DuelingQNetwork

class ReplayBuffer:
    def __init__(self, capacity, prioritized=False, alpha=0.6):
        self.prioritized = prioritized
        self.alpha = alpha
        self.data = collections.deque(maxlen=capacity)
        self.prios = collections.deque(maxlen=capacity)

    def push(self, *transition, td_error=1.0):
        self.data.append(transition)
        self.prios.append(abs(td_error) + 1e-5)

    def _probabilities(self):
        scaled = np.array(self.prios) ** self.alpha
        return scaled / scaled.sum()

    def sample(self, batch_size):
        if self.prioritized:
            probs = self._probabilities()
            idxs = np.random.choice(len(self.data), batch_size, p=probs)
        else:
            idxs = np.random.choice(len(self.data), batch_size)
        batch = [self.data[i] for i in idxs]
        return map(np.array, zip(*batch))

    def __len__(self): return len(self.data)


class DQNAgent:
    def __init__(
        self, state_dim, fdim, history_len, num_actions,
        device='cpu', prioritized=False,
        buffer_size=100_000, batch_size=64, gamma=0.99,
        lr=5e-4, eps_start=1.0, eps_end=0.05, eps_decay=25_000,
        target_update=2000
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy = DuelingQNetwork(fdim, history_len, num_actions).to(device)
        self.target = DuelingQNetwork(fdim, history_len, num_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, prioritized)
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.epsilon = eps_start
        self.steps = 0

    # ------------ acting ------------
    def select(self, state):
        self.steps += 1
        self.epsilon = max(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * (self.steps / self.eps_decay)
        )
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy(s).argmax(1).item())

    # ------------ learning ------------
    def store(self, *tr): self.buffer.push(*tr)

    def update(self):
        if len(self.buffer) < self.batch_size: return None
        S, A, R, S2, D = self.buffer.sample(self.batch_size)
        S  = torch.FloatTensor(S).to(self.device)
        A  = torch.LongTensor(A).to(self.device)
        R  = torch.FloatTensor(R).to(self.device)
        S2 = torch.FloatTensor(S2).to(self.device)
        D  = torch.FloatTensor(D).to(self.device)

        q_sa = self.policy(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # ----- Double-DQN target -----
        with torch.no_grad():
            a_next = self.policy(S2).argmax(1)
            q_next = self.target(S2).gather(1, a_next.unsqueeze(1)).squeeze(1)
        td_target = R + self.gamma * q_next * (1 - D)

        loss = F.smooth_l1_loss(q_sa, td_target)  # Huber

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()

        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()
