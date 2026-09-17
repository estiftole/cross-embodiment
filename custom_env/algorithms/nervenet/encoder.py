import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, hidden_state_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_state_dim)
        )

    def forward(self, obs):
        return self.encoder(obs)
