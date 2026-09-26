import torch
import torch.nn as nn
from torch.distributions import Normal

class DecoderMLP(nn.Module):
    def __init__(self, j_desc_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(j_desc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, j_desc):
        return self.net(j_desc)

class MuMLP(nn.Module):
    def __init__(self, decoder_out_dim, action_latent_dim, j_prod_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(decoder_out_dim+action_latent_dim+j_prod_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, decoder_out, action_latent, j_prod):
        batch_size, num_joints, _ = j_prod.shape
        action_latent_expanded = action_latent.unsqueeze(1).expand(-1, num_joints, -1)

        x = torch.cat([decoder_out, action_latent_expanded, j_prod], dim=-1)
        mu = self.net(x)
        return mu

class SigmaLayer(nn.Module):
    def __init__(self, decoder_out_dim, out_dim, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(decoder_out_dim, out_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_std = self.linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return torch.exp(log_std)

class ActionDecoder(nn.Module):
    def __init__(self,
        j_desc_dim,
        decoder_hidden_dim,
        decoder_out_dim,

        action_latent_dim,
        j_prod_dim,
        mu_hidden_dim,
        action_dim,
    ):
        super().__init__()

        self.decoder = DecoderMLP(j_desc_dim, decoder_hidden_dim, decoder_out_dim)
        self.mu_mlp = MuMLP(decoder_out_dim, action_latent_dim, j_prod_dim, mu_hidden_dim, action_dim)
        self.sigma_layer = SigmaLayer(decoder_out_dim, action_dim)

    def forward(self, j_desc, action_latent, j_prod, action=None, deterministic=False):
        dec_out = self.decoder(j_desc)
        mu = self.mu_mlp(dec_out, action_latent, j_prod)
        sigma = self.sigma_layer(dec_out)

        if mu.ndim > 2 and mu.shape[-1] == 1:
            mu = mu.squeeze(-1)
        if sigma.ndim > 2 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        dist = Normal(mu, sigma)

        if deterministic:
            act = mu
        elif action is None:
            act = dist.rsample()
        else:
            if action.ndim > mu.ndim:
                action = action.squeeze(-1)
            act = action

        log_prob = dist.log_prob(act).sum(dim=-1)

        return act, log_prob
