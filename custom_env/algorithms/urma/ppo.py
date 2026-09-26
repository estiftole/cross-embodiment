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
        # URMA Hidden Architecture Dimensions
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

        # 1. Infer raw feature dimensions dynamically from observation spaces
        target_obs_dim = observation_space["target_obs"].shape[-1]
        base_obs_dim = observation_space["base_obs"].shape[-1]

        j_obs_dim = observation_space["j_obs"].shape[-1]
        j_desc_dim = observation_space["j_desc"].shape[-1]

        ee_obs_dim = observation_space["ee_obs"].shape[-1]
        ee_desc_dim = observation_space["ee_desc"].shape[-1]

        # 2. Instantiate URMA Actor
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

        # 3. Instantiate URMA Critic
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

    def forward(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during rollout collection."""
        # 1. Compute Actions & Log Probabilities via Actor
        actions, log_prob = self.actor_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
            action=None  # Triggers sampling inside ActionDecoder
        )

        # Ensure action shape is flat across nodes: (B, max_joints)
        if actions.ndim > 2:
            actions = actions.squeeze(-1)

        # Apply action mask to zero out dummy padded joints
        actions = actions * obs["act_mask"]

        # Ensure log_prob matches padded joint dimensions and mask
        if log_prob.ndim > 1:
            log_prob = (log_prob * obs["act_mask"]).sum(dim=-1)

        # 2. Compute State Values via Critic
        values = self.critic_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
        )

        if values.ndim > 2:
            values = values.mean(dim=1)  # Pool per-node scalar values to graph level
        values = values.reshape(-1, 1)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during PPO optimization step."""
        # 1. Evaluate given actions using URMA Actor
        evaluated_actions, log_prob = self.actor_net(
            target_obs=obs["target_obs"],
            base_obs=obs["base_obs"],
            j_obs=obs["j_obs"],
            j_desc=obs["j_desc"],
            ee_obs=obs["ee_obs"],
            ee_desc=obs["ee_desc"],
            action=actions  # Evaluate provided rollout actions
        )

        # Mask log probabilities for valid active nodes
        if log_prob.ndim > 1:
            log_prob = (log_prob * obs["act_mask"]).sum(dim=-1)

        # Compute entropy surrogate from evaluated actions distribution
        # (Assuming standard Gaussian log_prob gradient proxy)
        entropy = -log_prob

        # 2. Evaluate State Values
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
        """Used for GAE value estimation."""
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
