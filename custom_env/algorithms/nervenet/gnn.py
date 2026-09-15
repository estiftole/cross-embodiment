import torch
import torch.nn as nn

class MessageFunction(nn.Module):
    def __init__(self, hidden_state_dim, hidden_dim, msg_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, msg_dim)
        )

    def forward(self, hidden_state):
        return self.net(hidden_state)

class UpdateFunction(nn.Module):
    def __init__(self, hidden_state_dim, hidden_dim, msg_aggr_dim):
        super().__init__()
        in_dim = hidden_state_dim+msg_aggr_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_state_dim)
        )

    def forward(self, hidden_state, msg_aggr):
        x = torch.cat([hidden_state, msg_aggr], dim=-1)
        return self.net(x)

class GNN(nn.Module):
    def __init__(self, iterations, hidden_state_dim, msg_dim, updater_hidden_dim, edge_hidden_dim):
        super().__init__()
        self.msg_dim = msg_dim
        self.iterations = iterations

        self.broadcast_edge = MessageFunction(hidden_state_dim, edge_hidden_dim, msg_dim)
        self.phys_edge = MessageFunction(hidden_state_dim, edge_hidden_dim, msg_dim)
        self.node_updater = UpdateFunction(hidden_state_dim, updater_hidden_dim, msg_dim)

    def forward(self, phys_hidden, goal_hidden, senders, receivers):
        batch_size, num_nodes, _ = phys_hidden.shape
        num_edges = senders.shape[0]
        rx_expanded = receivers.view(1, num_edges, 1).expand(batch_size, num_edges, self.msg_dim)

        for _ in range(self.iterations):
            broadcast_msg = self.broadcast_edge(goal_hidden)
            msg_accumulator = broadcast_msg.unsqueeze(1).repeat(1, num_nodes, 1)

            sender_states = phys_hidden[:, senders, :]
            phys_msgs = self.phys_edge(sender_states)

            msg_accumulator.scatter_add_(dim=1, index=rx_expanded, src=phys_msgs)
            phys_hidden = self.node_updater(phys_hidden, msg_accumulator)

        return phys_hidden
