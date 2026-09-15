import torch
import torch.nn as nn

class CoreNetwork(nn.Module):
    def __init__(self, joints_dim, feet_dim, general_obs_dim, hidden_dim, action_latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joints_dim+feet_dim+general_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_latent_dim),
            nn.ELU(),
        )

    def forward(self, joints_latent_vector, feet_latent_vector, general_obs_vector):
        x = torch.cat([joints_latent_vector, feet_latent_vector, general_obs_vector])
        action_latent_vector = self.net(x)
        return action_latent_vector
