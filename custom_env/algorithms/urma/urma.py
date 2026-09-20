from encoder import CompositeEncoder, AttentionModule
from core_network import CoreNetwork
from decoder import ActionDecoder

import torch.nn as nn

class URMA(nn.Module):
    def __init__(self,
        j_obs_dim, j_desc_dim,
        ee_obs_dim, ee_desc_dim,

        general_obs_dim,
        target_obs_dim,#=2

        enc_hidden_dim,
        j_latent_dim,
        ee_latent_dim,

        embed_dim,
        attn_heads,#=4,

        hidden_dim,
        action_latent_dim,

        dec_hidden_dim,
        dec_out_dim,

        mu_hidden_dim,
        action_dim,
    ) -> None:
        super().__init__()
        self.j_composite_enc = CompositeEncoder(j_obs_dim, j_desc_dim, enc_hidden_dim, j_latent_dim)
        self.j_attn_module = AttentionModule(embed_dim, attn_heads)
        self.ee_composite_enc = CompositeEncoder(ee_obs_dim, ee_desc_dim, enc_hidden_dim, ee_latent_dim)
        self.ee_attn_module = AttentionModule(embed_dim, attn_heads)
        self.core_net = CoreNetwork(j_latent_dim, ee_latent_dim, general_obs_dim, target_obs_dim, hidden_dim, action_latent_dim)
        self.action_dec = ActionDecoder(j_desc_dim, dec_hidden_dim, dec_out_dim, action_latent_dim, j_latent_dim, mu_hidden_dim, action_dim)

    def forward(self, j_obs, j_desc, ee_obs, ee_desc, target_obs, general_obs):
        j_prod = self.j_composite_enc(j_obs, j_desc)
        j_latent = self.j_attn_module(j_prod)

        ee_prod = self.j_composite_enc(ee_obs, ee_desc)
        ee_latent = self.j_attn_module(ee_prod)

        action_latent = self.core_net(j_latent, ee_latent, general_obs, target_obs)

        action = self.action_dec(j_desc, action_latent, j_prod)

        return action
