# models/network.py

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    A simple feed-forward Q-network. 
    Input: state_dim vector
    Output: num_actions Q-values
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_sizes=(256,128)):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, num_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, state_dim]
        return self.net(x)
