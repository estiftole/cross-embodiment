from .encoder import ObservationEncoder
from .gnn import GraphNN
from .decoder import ActionDecoder

import torch
import torch.nn as nn
# from torch.distributions import Normal

def prepare_inputs(obs, graph_meta, device):

    base_obs = torch.as_tensor(obs["base_obs"], dtype=torch.float32).unsqueeze(0).to(device)
    target_obs = torch.as_tensor(obs["target_obs"], dtype=torch.float32).unsqueeze(0).to(device)

    j_obs = torch.as_tensor(obs["j_obs"], dtype=torch.float32).to(device)
    if j_obs.ndim == 2:
        j_obs = j_obs.unsqueeze(0)

    senders = torch.as_tensor(graph_meta["senders"], dtype=torch.long).to(device)
    receivers = torch.as_tensor(graph_meta["receivers"], dtype=torch.long).to(device)
    actuatable_nodes = torch.as_tensor(graph_meta["actuatable_nodes"], dtype=torch.long).to(device)

    return {
        "target_obs": target_obs,
        "base_obs": base_obs,
        "j_obs": j_obs,
        "senders": senders,
        "receivers": receivers,
        "actuatable_nodes": actuatable_nodes
    }

class NerveNetActor(nn.Module):
    def __init__(self,
        j_obs_dim: int, target_obs_dim: int, base_obs_dim: int,

        obs_enc_hidden_dim=32,
        hidden_state_dim=64,

        updater_hidden_dim=64, msg_hidden_dim=32,
        msg_dim=16,
        iterations=12,

        dec_hidden_dim=32, action_dim=1
    ) -> None:
        super().__init__()
        combined_obs_dim = j_obs_dim + base_obs_dim
        self.target_enc = ObservationEncoder(target_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.j_enc = ObservationEncoder(combined_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)
        self.action_dec = ActionDecoder(hidden_state_dim, dec_hidden_dim, action_dim)

        # self.log_std = nn.Parameter(torch.zeros(total_action_dim))

    def forward(self, target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes, node_batch):
        target_hidden = self.target_enc(target_obs)
        base_obs_per_node = base_obs[node_batch]

        combined_obs = torch.cat([base_obs_per_node, j_obs], dim=-1)
        phys_hidden = self.j_enc(combined_obs)
        gnn_out = self.gnn(phys_hidden, target_hidden, senders, receivers, node_batch)

        motor_joint_states = gnn_out[actuatable_nodes]
        mu = self.action_dec(motor_joint_states)

        return mu.squeeze(-1)


class NerveNetCritic(nn.Module):
    def __init__(self,
        j_obs_dim: int, target_obs_dim: int, base_obs_dim: int,

        obs_enc_hidden_dim=32,
        hidden_state_dim=64,

        updater_hidden_dim=64, msg_hidden_dim=32,
        msg_dim=16,
        iterations=12,

        dec_hidden_dim=32
    ) -> None:
        super().__init__()
        combined_obs_dim = j_obs_dim + base_obs_dim
        self.target_enc = ObservationEncoder(target_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.j_enc = ObservationEncoder(combined_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)

        self.value_dec = nn.Sequential(
            nn.Linear(hidden_state_dim, dec_hidden_dim),
            nn.ReLU(),
            nn.Linear(dec_hidden_dim, 1)
        )

    def forward(self, target_obs, base_obs, j_obs, senders, receivers, node_batch):
        batch_size = target_obs.shape[0]

        target_hidden = self.target_enc(target_obs)
        base_obs_per_node = base_obs[node_batch]

        combined_obs = torch.cat([base_obs_per_node, j_obs], dim=-1)
        phys_hidden = self.j_enc(combined_obs)
        gnn_out = self.gnn(phys_hidden, target_hidden, senders, receivers, node_batch)

        graph_summary = torch.zeros(batch_size, gnn_out.size(-1), device=gnn_out.device)
        graph_summary.index_add_(0, node_batch, gnn_out)

        node_counts = torch.bincount(node_batch, minlength=batch_size).unsqueeze(1).clamp(min=1)
        graph_summary = graph_summary / node_counts

        value = self.value_dec(graph_summary)

        return value
