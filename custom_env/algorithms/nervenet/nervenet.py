from .encoder import ObservationEncoder
from .gnn import GraphNN
from .decoder import ActionDecoder

import torch
import torch.nn as nn

def prepare_inputs(obs_dict, graph_meta, device="cpu"):

    base_obs = obs_dict["base_obs"]
    target_obs = obs_dict["target_obs"]

    j_obs = torch.tensor(obs_dict["joint_obs"], dtype=torch.float32, device=device).unsqueeze(0)

    senders = graph_meta["senders"].to(device)
    receivers = graph_meta["receivers"].to(device)
    actuatable_nodes = graph_meta["actuatable_nodes"].to(device)

    return {
        "target_obs": target_obs,
        "base_obs": base_obs,
        "j_obs": j_obs,
        "senders": senders,
        "receivers": receivers,
        "actuatable_nodes": actuatable_nodes
    }

class NerveNet(nn.Module):
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
        self.root_enc = ObservationEncoder(base_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)
        self.action_dec = ActionDecoder(hidden_state_dim, action_dec_hidden_dim, action_dim)

    def forward(self, target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes):
        target_hidden = self.target_enc(target_obs)
        j_hidden = self.target_enc(j_obs)
        root_hidden = self.target_enc(base_obs)

        phys_hidden = torch.cat([root_hidden, j_hidden], dim=1)
        motor_joint_states = self.gnn(phys_hidden, target_hidden, senders, receivers)
        motor_joint_states = self.gnn(phys_hidden, target_hidden, senders, receivers)[:, actuatable_nodes, :]
        actions = self.action_dec(motor_joint_states)

        return actions
