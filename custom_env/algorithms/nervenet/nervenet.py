from .encoder import ObservationEncoder
from .gnn import GraphNN
from .decoder import ActionDecoder

import torch
import torch.nn as nn

def prepare_inputs(obs, graph_meta, device="cpu"):

    base_obs = torch.as_tensor(obs["base_obs"], dtype=torch.float32, device=device).unsqueeze(0)
    target_obs = torch.as_tensor(obs["target_obs"], dtype=torch.float32, device=device).unsqueeze(0)

    j_obs = torch.as_tensor(obs["joint_obs"], dtype=torch.float32, device=device)
    if j_obs.ndim == 2:
        j_obs = j_obs.unsqueeze(0)

    senders = torch.as_tensor(graph_meta["senders"], dtype=torch.long, device=device)
    receivers = torch.as_tensor(graph_meta["receivers"], dtype=torch.long, device=device)
    actuatable_nodes = torch.as_tensor(graph_meta["actuatable_nodes"], dtype=torch.long, device=device)

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
        j_obs_dim, target_obs_dim, base_obs_dim,

        obs_enc_hidden_dim,
        hidden_state_dim,

        updater_hidden_dim, msg_hidden_dim,
        msg_dim,
        iterations,

        action_dec_hidden_dim,
        action_dim
    ) -> None:
        super().__init__()
        self.target_enc = ObservationEncoder(target_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.j_enc = ObservationEncoder(j_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.base_enc = ObservationEncoder(base_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)
        self.action_dec = ActionDecoder(hidden_state_dim, action_dec_hidden_dim, action_dim)

    def forward(self, target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes):
        target_hidden = self.target_enc(target_obs)
        j_hidden = self.j_enc(j_obs)
        base_hidden = self.base_enc(base_obs).unsqueeze(1)

        phys_hidden = torch.cat([base_hidden, j_hidden], dim=1)
        # motor_joint_states = self.gnn(phys_hidden, target_hidden, senders, receivers)
        motor_joint_states = self.gnn(phys_hidden, target_hidden, senders, receivers)[:, actuatable_nodes, :]
        actions = self.action_dec(motor_joint_states)

        return actions


class NerveNetCritic(nn.Module):
    def __init__(self,
        j_obs_dim, target_obs_dim, base_obs_dim,

        obs_enc_hidden_dim,
        hidden_state_dim,

        updater_hidden_dim, msg_hidden_dim,
        msg_dim,
        iterations,

        value_dec_hidden_dim
    ) -> None:
        super().__init__()
        self.target_enc = ObservationEncoder(target_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.j_enc = ObservationEncoder(j_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.base_enc = ObservationEncoder(base_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)

        self.value_dec = nn.Sequential(
            nn.Linear(hidden_state_dim, value_dec_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_dec_hidden_dim, 1)
        )

    def forward(self, target_obs, base_obs, j_obs, senders, receivers):
        target_hidden = self.target_enc(target_obs)
        j_hidden = self.j_enc(j_obs)
        base_hidden = self.base_enc(base_obs).unsqueeze(1)

        phys_hidden = torch.cat([base_hidden, j_hidden], dim=1)
        updated_phys = self.gnn(phys_hidden, target_hidden, senders, receivers)

        graph_summary = updated_phys.mean(dim=1)

        value = self.value_dec(graph_summary)

        return value
