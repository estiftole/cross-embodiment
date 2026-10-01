import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algorithms.nervenet import NerveNetActor, NerveNetCritic


def unpad_and_batch_graphs(features: dict):
    """
    Converts a padded minibatch dictionary into a single disjoint graph
    containing ONLY valid nodes and edges.
    """
    node_mask = features["node_mask"].bool()    # [B, max_joints]
    edge_mask = features["edge_mask"].bool()    # [B, max_edges]
    act_mask = features["act_mask"].bool()      # [B, max_joints]

    B, max_joints = node_mask.shape
    device = node_mask.device

    # 1. Extract valid joint observations using boolean indexing directly
    j_obs_disjoint = features["j_obs"][node_mask]  # [Total_Active_Nodes, j_obs_dim]

    # 2. Compute cumulative node offsets per graph using prefix sum
    nodes_per_graph = node_mask.sum(dim=-1)                       # [B] tensor
    cum_nodes = torch.cumsum(nodes_per_graph, dim=0)              # [B] tensor
    node_offsets = cum_nodes - nodes_per_graph                    # [B] tensor

    # 3. Broadcast node offsets across edge and actuator dimensions
    edge_offsets = node_offsets.unsqueeze(1)                      # [B, 1]

    senders_offset = features["senders"].long() + edge_offsets     # [B, max_edges]
    receivers_offset = features["receivers"].long() + edge_offsets # [B, max_edges]
    act_offset = features["actuatable_nodes"].long() + edge_offsets# [B, max_joints]

    # 4. Extract valid edges and actuatable nodes using boolean masks
    senders_disjoint = senders_offset[edge_mask]                  # [Total_Active_Edges]
    receivers_disjoint = receivers_offset[edge_mask]              # [Total_Active_Edges]
    act_nodes_disjoint = act_offset[act_mask]                     # [Total_Active_Actuators]

    # 5. Generate node_batch indices vectorially
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(node_mask)
    node_batch = batch_idx[node_mask]

    return {
        "j_obs": j_obs_disjoint,
        "senders": senders_disjoint,
        "receivers": receivers_disjoint,
        "actuatable_nodes": act_nodes_disjoint,
        "nodes_per_graph": nodes_per_graph,                       # Retained as GPU Tensor!
        "act_nodes_per_graph": act_mask.sum(dim=-1),              # Retained as GPU Tensor!
        "batch_size": B,
        "node_batch": node_batch,
    }

class DynamicGraphDictExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: dict) -> dict:
        return observations

class NetworkWrapper(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module, action_dim: int, max_joints=16):
        self.max_joints=max_joints
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.latent_dim_pi = action_dim
        self.latent_dim_vf = 1

    def forward(self, features: dict) -> torch.Tensor:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: dict) -> torch.Tensor:
        batch_size = features["target_obs"].shape[0]
        device = features["target_obs"].device

        graph_data = unpad_and_batch_graphs(features)

        real_actions = self.actor(
            target_obs=features["target_obs"],
            base_obs=features["base_obs"],
            j_obs=graph_data["j_obs"],
            senders=graph_data["senders"],
            receivers=graph_data["receivers"],
            actuatable_nodes=graph_data["actuatable_nodes"],
            node_batch=graph_data["node_batch"],
        )
        padded_actions = torch.zeros((batch_size, self.max_joints), device=device)

        curr_idx = 0
        for b in range(batch_size):
            n_act = graph_data["act_nodes_per_graph"][b]
            sample_act = real_actions[curr_idx : curr_idx + n_act]
            padded_actions[b, :n_act] = sample_act.squeeze()
            curr_idx += n_act

        return padded_actions

    def forward_critic(self, features: dict) -> torch.Tensor:
        graph_data = unpad_and_batch_graphs(features)

        values = self.critic(
            target_obs=features["target_obs"],
            base_obs=features["base_obs"],
            j_obs=graph_data["j_obs"],
            senders=graph_data["senders"],
            receivers=graph_data["receivers"],
            node_batch=graph_data["node_batch"],
        )
        return values

class NerveNetPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule,
        **kwargs
    ):

        self.target_obs_dim = observation_space["target_obs"].shape[-1]
        self.base_obs_dim = observation_space["base_obs"].shape[-1]
        self.j_obs_dim = observation_space["j_obs"].shape[-1]

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=DynamicGraphDictExtractor,
            **kwargs
        )

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        actor = NerveNetActor(self.j_obs_dim, self.target_obs_dim, self.base_obs_dim)
        critic = NerveNetCritic(self.j_obs_dim, self.target_obs_dim, self.base_obs_dim)

        self.mlp_extractor = NetworkWrapper(
            actor=actor,
            critic=critic,
            action_dim=self.action_space.shape[0],
        )
