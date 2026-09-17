import torch
import torch.nn as nn

class DecoderMLP(nn.Module):
    def __init__(self, joint_desc_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_desc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, joint_desc):
        return self.net(joint_desc)

class MuMLP(nn.Module):
    def __init__(self, decoder_out_dim, action_latent_dim, joint_product_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(decoder_out_dim+action_latent_dim+joint_product_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, decoder_out, action_latent_vector, joint_product):
        x = torch.cat([decoder_out, action_latent_vector, joint_product], dim=-1)
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
        joint_desc_dim,
        decoder_hidden_dim,
        decoder_out_dim,

        action_latent_dim,
        joint_product_dim,
        mu_hidden_dim,
        action_dim,
    ):
        super().__init__()

        self.decoder = DecoderMLP(joint_desc_dim, decoder_hidden_dim, decoder_out_dim)
        self.mu_mlp = MuMLP(decoder_out_dim, action_latent_dim, joint_product_dim, mu_hidden_dim, action_dim)
        self.sigma_layer = SigmaLayer(decoder_out_dim, action_dim)

    def forward(self, joint_desc, action_latent_vector, joint_product, deterministic=False):
        dec_out = self.decoder(joint_desc)
        mu = self.mu_mlp(dec_out, action_latent_vector, joint_product)
        sigma = self.sigma_layer(dec_out)

        if deterministic:
            action = mu
        else:
            epsilon = torch.randn_like(sigma)
            action = mu + epsilon * sigma

        return action
