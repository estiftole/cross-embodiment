import torch
import torch.nn as nn

class DescriptionEncoder(nn.Module):
    def __init__(self, desc_dim, hidden_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(desc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, out_dim),
            nn.ELU()
        )

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

class AttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.mha(x, x, x)
        return attn_out

class CompositeEncoderNetwork(nn.Module):
    def __init__(self, obs_dim: int, desc_dim: int, hidden_dim: int, latent_dim: int, num_heads: int):
        super().__init__()
        self.obs_encoder = ObservationEncoder(obs_dim, latent_dim)
        self.desc_encoder = DescriptionEncoder(desc_dim, hidden_dim, latent_dim)
        self.attn_encoder = AttentionEncoder(latent_dim, num_heads)

    def forward(self, obs: torch.Tensor, desc: torch.Tensor):
        obs_latent = self.obs_encoder(obs)
        desc_latent = self.desc_encoder(desc)

        latent_product = obs_latent * desc_latent

        latent_vector = self.attn_encoder(latent_product)
        return latent_vector
