from typing import Dict, Tuple, Any, Callable
from gymnasium import spaces

import torch
from torch.distributions import Normal
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from algorithms.nervenet import NerveNetActor, NerveNetCritic

class SB3NerveNetPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Any,

        obs_enc_hidden_dim: int = 32,
        hidden_state_dim: int = 64,
        updater_hidden_dim: int = 64,
        msg_hidden_dim: int = 32,
        msg_dim: int = 16,
        iterations: int = 2,
        dec_hidden_dim: int = 32,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        target_obs_dim = observation_space["target_obs"].shape[-1]
        base_obs_dim = observation_space["base_obs"].shape[-1]
        j_obs_dim = observation_space["j_obs"].shape[-1]

        padded_action_dim = observation_space["act_mask"].shape[0]

        self.actor_net = NerveNetActor(
            j_obs_dim=j_obs_dim,
            target_obs_dim=target_obs_dim,
            base_obs_dim=base_obs_dim,
            obs_enc_hidden_dim=obs_enc_hidden_dim,
            hidden_state_dim=hidden_state_dim,
            updater_hidden_dim=updater_hidden_dim,
            msg_hidden_dim=msg_hidden_dim,
            msg_dim=msg_dim,
            iterations=iterations,
            dec_hidden_dim=dec_hidden_dim,
            action_dim=1,
            total_action_dim=padded_action_dim  # Now guaranteed to be 16
        )

        self.critic_net = NerveNetCritic(
            j_obs_dim=j_obs_dim,
            target_obs_dim=target_obs_dim,
            base_obs_dim=base_obs_dim,
            obs_enc_hidden_dim=obs_enc_hidden_dim,
            hidden_state_dim=hidden_state_dim,
            updater_hidden_dim=updater_hidden_dim,
            msg_hidden_dim=msg_hidden_dim,
            msg_dim=msg_dim,
            iterations=iterations,
            dec_hidden_dim=dec_hidden_dim
        )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor_net(
            obs["target_obs"], obs["base_obs"], obs["j_obs"],
            obs["senders"], obs["receivers"], obs["actuatable_nodes"]
        )
        values = self.critic_net(
            obs["target_obs"], obs["base_obs"], obs["j_obs"],
            obs["senders"], obs["receivers"]
        )

        std = torch.exp(self.actor_net.log_std)
        distribution = Normal(mu, std)

        actions = mu if deterministic else distribution.sample()

        actions = actions * obs["act_mask"]

        # FIX 2: Zero out the dummy log probabilities before summing!
        log_prob = distribution.log_prob(actions)
        log_prob = log_prob * obs["act_mask"]
        log_prob = log_prob.sum(dim=-1)

        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor_net(
            obs["target_obs"], obs["base_obs"], obs["j_obs"],
            obs["senders"], obs["receivers"], obs["actuatable_nodes"]
        )
        values = self.critic_net(
            obs["target_obs"], obs["base_obs"], obs["j_obs"],
            obs["senders"], obs["receivers"]
        )

        std = torch.exp(self.actor_net.log_std)
        distribution = Normal(mu, std)

        log_prob = distribution.log_prob(actions) * obs["act_mask"]
        entropy = distribution.entropy() * obs["act_mask"]

        log_prob = log_prob.sum(dim=-1)
        entropy = entropy.sum(dim=-1)

        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic_net(
            obs["target_obs"], obs["base_obs"], obs["j_obs"],
            obs["senders"], obs["receivers"]
        )
