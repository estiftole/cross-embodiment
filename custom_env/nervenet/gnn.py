import torch
import torch.nn as nn

class MessageFunction(nn.Module):
    def __init__(self, hidden_state_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, hidden_state):
        return self.net(hidden_state)

class UpdateFunction(nn.Module):
    def __init__(self, hidden_state_dim, msg_aggr_dim, hidden_dim, out_dim):
        super().__init__()
        in_dim = hidden_state_dim+msg_aggr_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, hidden_state, msg_aggr):
        x = torch.cat([hidden_state, msg_aggr], dim=-1)
        return self.net(x)

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass
