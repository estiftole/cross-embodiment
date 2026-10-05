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

class GraphNN(nn.Module):
    def __init__(self, hidden_state_dim, updater_hidden_dim, msg_hidden_dim, msg_dim, iterations):
        super().__init__()
        self.iterations = iterations

        self.broadcast_edge = MessageFunction(hidden_state_dim, msg_hidden_dim, msg_dim)
        self.phys_edge = MessageFunction(hidden_state_dim, msg_hidden_dim, msg_dim)
        self.node_updater = UpdateFunction(hidden_state_dim, updater_hidden_dim, msg_dim)

    def forward(self, phys_hidden, goal_hidden, senders, receivers, node_batch):
        goal_msg = self.broadcast_edge(goal_hidden)[node_batch]

        for _ in range(self.iterations):
            sender_states = phys_hidden[senders]
            phys_msgs = self.phys_edge(sender_states)

            msg_accumulator = torch.zeros_like(goal_msg)
            msg_accumulator.index_add_(0, receivers, phys_msgs)

            total_msg = msg_accumulator + goal_msg
            phys_hidden = self.node_updater(phys_hidden, total_msg)

        return phys_hidden
