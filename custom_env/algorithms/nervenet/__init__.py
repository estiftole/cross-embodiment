from .encoder import ObservationEncoder
from .gnn import GraphNN
from .decoder import ActionDecoder

import torch
import torch.nn as nn
from torch.distributions import Normal

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

import torch
import torch.nn as nn
from torch.distributions import Normal

class NerveNetActor(nn.Module):
    def __init__(self,
        j_obs_dim, target_obs_dim, base_obs_dim,
        obs_enc_hidden_dim, hidden_state_dim,
        updater_hidden_dim, msg_hidden_dim,
        msg_dim, iterations,
        dec_hidden_dim, action_dim, total_action_dim
    ) -> None:
        super().__init__()
        combined_obs_dim = j_obs_dim + base_obs_dim
        self.target_enc = ObservationEncoder(target_obs_dim, obs_enc_hidden_dim, hidden_state_dim)
        self.j_enc = ObservationEncoder(combined_obs_dim, obs_enc_hidden_dim, hidden_state_dim)

        self.gnn = GraphNN(hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations)
        self.action_dec = ActionDecoder(hidden_state_dim, dec_hidden_dim, action_dim)

        self.log_std = nn.Parameter(torch.zeros(total_action_dim))

    def forward(self, target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes):
        target_hidden = self.target_enc(target_obs)
        base_obs_expanded = base_obs.unsqueeze(1).expand(-1, j_obs.size(1), -1)

        combined_obs = torch.cat([base_obs_expanded, j_obs], dim=-1)
        phys_hidden = self.j_enc(combined_obs)
        gnn_out = self.gnn(phys_hidden, target_hidden, senders, receivers)

        batch_size = gnn_out.shape[0]
        batch_idx = torch.arange(batch_size, device=gnn_out.device).unsqueeze(1)
        motor_joint_states = gnn_out[batch_idx, actuatable_nodes, :]

        mu = self.action_dec(motor_joint_states)

        return mu.squeeze(-1)

    def get_action_and_log_prob(self, target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes, action=None):
        mu = self.forward(target_obs, base_obs, j_obs, senders, receivers, actuatable_nodes)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        while log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy


class NerveNetCritic(nn.Module):
    def __init__(self,
        j_obs_dim, target_obs_dim, base_obs_dim,

        obs_enc_hidden_dim,
        hidden_state_dim,

        updater_hidden_dim, msg_hidden_dim,
        msg_dim,
        iterations,

        dec_hidden_dim
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

    def forward(self, target_obs, base_obs, j_obs, senders, receivers):
        target_hidden = self.target_enc(target_obs)
        base_obs_expanded = base_obs.unsqueeze(1).expand(-1, j_obs.size(1), -1)

        combined_obs = torch.cat([base_obs_expanded, j_obs], dim=-1)
        phys_hidden = self.j_enc(combined_obs)
        gnn_out = self.gnn(phys_hidden, target_hidden, senders, receivers)

        graph_summary = gnn_out.mean(dim=1)

        value = self.value_dec(graph_summary)

        return value
