import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from algorithms.nervenet import NerveNetActor, NerveNetCritic


def unpad_and_batch_graphs(features: dict, max_joints: int = 16):
    node_mask = features["node_mask"].bool()
    edge_mask = features["edge_mask"].bool()
    act_mask = features["act_mask"].bool()

    B, max_nodes = node_mask.shape
    device = node_mask.device

    j_obs_disjoint = features["j_obs"][node_mask]

    # --- Node ID Extraction / Auto-Generation ---
    if "node_ids" in features:
        node_ids_disjoint = features["node_ids"][node_mask]
    else:
        # Auto-generate one-hot positional index embeddings per node in the graph
        local_node_idx = torch.arange(max_nodes, device=device).unsqueeze(0).expand(B, max_nodes)
        active_indices = local_node_idx[node_mask]
        node_ids_disjoint = F.one_hot(active_indices, num_classes=max_nodes).float()

    nodes_per_graph = node_mask.sum(dim=-1)
    cum_nodes = torch.cumsum(nodes_per_graph, dim=0)
    node_offsets = cum_nodes - nodes_per_graph

    edge_offsets = node_offsets.unsqueeze(1)

    senders_offset = features["senders"].long() + edge_offsets
    receivers_offset = features["receivers"].long() + edge_offsets
    act_offset = features["actuatable_nodes"].long() + edge_offsets

    senders_disjoint = senders_offset[edge_mask]
    receivers_disjoint = receivers_offset[edge_mask]
    act_nodes_disjoint = act_offset[act_mask]

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(node_mask)
    node_batch = batch_idx[node_mask]

    return {
        "j_obs": j_obs_disjoint,
        "node_ids": node_ids_disjoint,
        "senders": senders_disjoint,
        "receivers": receivers_disjoint,
        "actuatable_nodes": act_nodes_disjoint,
        "nodes_per_graph": nodes_per_graph,
        "act_nodes_per_graph": act_mask.sum(dim=-1),
        "batch_size": B,
        "node_batch": node_batch,
    }


class ObservationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: dict) -> dict:
        return observations


class NetworkWrapper(nn.Module):
    def __init__(self, actor: nn.Module, critic: nn.Module, action_dim: int, max_joints: int = 16):
        super().__init__()
        self.max_joints = max_joints
        self.actor = actor
        self.critic = critic

        self.latent_dim_pi = action_dim
        self.latent_dim_vf = 1

    def forward(self, features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: dict) -> torch.Tensor:
        batch_size = features["target_obs"].shape[0]
        device = features["target_obs"].device

        graph_data = unpad_and_batch_graphs(features, max_joints=self.max_joints)

        # Updated to pass graph_data["node_ids"]
        real_actions = self.actor(
            target_obs=features["target_obs"],
            base_obs=features["base_obs"],
            j_obs=graph_data["j_obs"],
            node_ids=graph_data["node_ids"],
            senders=graph_data["senders"],
            receivers=graph_data["receivers"],
            actuatable_nodes=graph_data["actuatable_nodes"],
            node_batch=graph_data["node_batch"],
        )

        padded_actions = torch.zeros((batch_size, self.max_joints), device=device)
        act_mask = features["act_mask"].bool()
        padded_actions[act_mask] = real_actions.flatten()

        return padded_actions

    def forward_critic(self, features: dict) -> torch.Tensor:
        graph_data = unpad_and_batch_graphs(features, max_joints=self.max_joints)

        # Updated to pass graph_data["node_ids"]
        values = self.critic(
            target_obs=features["target_obs"],
            base_obs=features["base_obs"],
            j_obs=graph_data["j_obs"],
            node_ids=graph_data["node_ids"],
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
        max_joints: int = 16,
        **kwargs
    ):
        self.max_joints = max_joints
        self.target_obs_dim = observation_space["target_obs"].shape[-1]
        self.base_obs_dim = observation_space["base_obs"].shape[-1]
        self.j_obs_dim = observation_space["j_obs"].shape[-1]

        # Determine node_id_dim from observation space or fallback to max_joints
        if "node_ids" in observation_space.spaces:
            self.node_id_dim = observation_space["node_ids"].shape[-1]
        else:
            self.node_id_dim = self.max_joints

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ObservationExtractor,
            **kwargs
        )

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _build_mlp_extractor(self) -> None:
        actor = NerveNetActor(
            j_obs_dim=self.j_obs_dim,
            target_obs_dim=self.target_obs_dim,
            base_obs_dim=self.base_obs_dim,
            node_id_dim=self.node_id_dim
        )
        critic = NerveNetCritic(
            j_obs_dim=self.j_obs_dim,
            target_obs_dim=self.target_obs_dim,
            base_obs_dim=self.base_obs_dim,
            node_id_dim=self.node_id_dim
        )

        self.mlp_extractor = NetworkWrapper(
            actor=actor,
            critic=critic,
            action_dim=self.action_space.shape[0],
            max_joints=self.max_joints
        )
