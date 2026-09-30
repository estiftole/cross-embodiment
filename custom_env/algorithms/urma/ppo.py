from typing import Dict, Tuple, Any
from gymnasium import spaces

import torch
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from algorithms.urma import URMAActor, URMACritic


class SB3URMAPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Any,

        enc_hidden_dim: int = 64,
        j_latent_dim: int = 64,
        ee_latent_dim: int = 64,
        embed_dim: int = 64,
        attn_heads: int = 4,
        hidden_dim: int = 128,
        action_latent_dim: int = 64,
        dec_hidden_dim: int = 64,
        dec_out_dim: int = 32,
        mu_hidden_dim: int = 32,
        action_dim: int = 1,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        target_obs_dim = observation_space["target_obs"].shape[-1]
        base_obs_dim = observation_space["base_obs"].shape[-1]

        j_obs_dim = observation_space["j_obs"].shape[-1]
        j_desc_dim = observation_space["j_desc"].shape[-1]

        ee_obs_dim = observation_space["ee_obs"].shape[-1]
        ee_desc_dim = observation_space["ee_desc"].shape[-1]

        self.actor_net = URMAActor(
            j_obs_dim=j_obs_dim,
            j_desc_dim=j_desc_dim,
            ee_obs_dim=ee_obs_dim,
            ee_desc_dim=ee_desc_dim,
            base_obs_dim=base_obs_dim,
            target_obs_dim=target_obs_dim,
            enc_hidden_dim=enc_hidden_dim,
            j_latent_dim=j_latent_dim,
            ee_latent_dim=ee_latent_dim,
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            hidden_dim=hidden_dim,
            action_latent_dim=action_latent_dim,
            dec_hidden_dim=dec_hidden_dim,
            dec_out_dim=dec_out_dim,
            mu_hidden_dim=mu_hidden_dim,
            action_dim=action_dim,
        )

        self.critic_net = URMACritic(
            j_obs_dim=j_obs_dim,
            j_desc_dim=j_desc_dim,
            ee_obs_dim=ee_obs_dim,
            ee_desc_dim=ee_desc_dim,
            base_obs_dim=base_obs_dim,
            target_obs_dim=target_obs_dim,
            enc_hidden_dim=enc_hidden_dim,
            j_latent_dim=j_latent_dim,
            ee_latent_dim=ee_latent_dim,
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            hidden_dim=hidden_dim,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, _ = self.actor_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
            action=None
        )

        if actions.ndim > 2:
            actions = actions.squeeze(-1)

        actions = actions * obs["act_mask"]

        if log_prob.ndim > 1:
            log_prob = (log_prob * obs["act_mask"]).sum(dim=-1)

        values = self.critic_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
        )

        if values.ndim > 2:
            values = values.mean(dim=1)
        values = values.reshape(-1, 1)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evaluated_actions, log_prob, entropy = self.actor_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
            action=actions
        )

        if log_prob.ndim > 1:
            log_prob = (log_prob * obs["act_mask"]).sum(dim=-1)

        values = self.critic_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
        )

        if values.ndim > 2:
            values = values.mean(dim=1)
        values = values.reshape(-1, 1)

        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        values = self.critic_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
        )
        if values.ndim > 2:
            values = values.mean(dim=1)
        return values.reshape(-1, 1)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions
