from typing import Tuple, Dict, Any, Callable
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from algorithms.urma import URMAActor, URMACritic


class URMAPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Callable[[float], float],
        urma_actor_kwargs: Dict[str, Any],
        urma_critic_kwargs: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            **kwargs,
        )

        self.actor = URMAActor(**urma_actor_kwargs)
        self.critic = URMACritic(**urma_critic_kwargs)

        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **self.optimizer_kwargs
        )

    def _unpack_obs(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, ...]:
        """Extract and order observation dictionary keys for URMA modules."""
        return (
            obs["target_obs"],
            obs["base_obs"],
            obs["j_obs"],
            obs["j_desc"],
            obs["ee_obs"],
            obs["ee_desc"],
        )

    def forward(
        self, obs: Dict[str, th.Tensor], deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Called during rollout collection (env.step)."""
        unpacked_obs = self._unpack_obs(obs)

        values = self.critic(*unpacked_obs)

        actions, log_prob, _ = self.actor(*unpacked_obs, action=None, deterministic=deterministic)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Dict[str, th.Tensor], actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Called during the PPO gradient update step."""
        unpacked_obs = self._unpack_obs(obs)

        values = self.critic(*unpacked_obs)

        _, log_prob, entropy = self.actor(*unpacked_obs, action=actions)

        return values, log_prob, entropy

    def _predict(
        self, observation: Dict[str, th.Tensor], deterministic: bool = False
    ) -> th.Tensor:
        """Called during model evaluation (model.predict)."""
        unpacked_obs = self._unpack_obs(observation)
        actions, _, _ = self.actor(*unpacked_obs, action=None, deterministic=deterministic)
        return actions
