import torch
import torch.nn as nn

class CoreNetwork(nn.Module):
    def __init__(self, j_latent_dim, ee_latent_dim, target_obs_dim, base_obs_dim, hidden_dim, action_latent_dim):
        super().__init__()
        in_dim = j_latent_dim+ee_latent_dim+target_obs_dim+base_obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_latent_dim),
            nn.ELU(),
        )

    def forward(self, j_latent, ee_latent, target_obs, base_obs):
        j_global = j_latent.mean(dim=1)
        ee_global = ee_latent.mean(dim=1)

        x = torch.cat([j_global, ee_global, target_obs, base_obs], dim=-1)
        action_latent = self.net(x)
        return action_latent
