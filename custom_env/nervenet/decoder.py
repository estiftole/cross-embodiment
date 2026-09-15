import torch.nn as nn

class ActionDecoder(nn.Module):
    def __init__(self, hidden_state_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, hidden_state):
        return self.net(hidden_state)
