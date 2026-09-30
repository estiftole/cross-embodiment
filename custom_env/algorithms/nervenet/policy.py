from typing import Tuple
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from algorithms.nervenet import NerveNetActor, NerveNetCritic

# 1. Wrap your custom Actor and Critic into a single module
class CustomGNNNetwork(nn.Module):
    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 128, last_layer_dim_vf: int = 128):
        super().__init__()
        # SB3 requires latent dimensions to set up final action/value distribution heads
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Instantiate your custom NerveNet / URMA models
        # (These should output feature vectors of size last_layer_dim_pi and last_layer_dim_vf)
        self.actor_backbone = NerveNetActor
        self.critic_backbone = NerveNetCritic

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.actor_backbone(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.critic_backbone(features)

class CustomPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomGNNNetwork(self.features_dim)


# model = PPO(CustomPolicy, env, verbose=1)
# model.learn(total_timesteps=100_000)
