# models/network.py
import torch
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    """
    LSTM encoder over the last K item-feature vectors + scalar extras.
    Dueling heads: V(s) and A(s,a).
    """
    def __init__(self, fdim: int, history_len: int, num_actions: int,
                 hidden_size: int = 128):
        super().__init__()
        self.fdim = fdim
        self.K = history_len
        self.lstm = nn.LSTM(
            input_size=fdim,
            hidden_size=hidden_size,
            batch_first=True
        )
        # scalar extras: fatigue, t
        self.extras_fc = nn.Linear(2, hidden_size)

        # dueling heads
        self.V = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.A = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state_flat):
        """
        state_flat: [B, K*fdim + 2]
        """
        extras = state_flat[:, -2:]                       # [B, 2]
        item_seq = state_flat[:, :-2].view(
            -1, self.K, self.fdim)                        # [B, K, fdim]

        lstm_out, _ = self.lstm(item_seq)                 # [B, K, hidden]
        h_seq = lstm_out[:, -1]                           # last hidden

        h_extras = torch.relu(self.extras_fc(extras))     # [B, hidden]
        h = torch.cat([h_seq, h_extras], dim=1)           # [B, hidden*2]

        V = self.V(h)                                     # [B, 1]
        A = self.A(h)                                     # [B, num_actions]
        Q = V + (A - A.mean(dim=1, keepdim=True))         # dueling combine
        return Q
