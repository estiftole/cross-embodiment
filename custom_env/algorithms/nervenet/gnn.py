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

class GraphNN(nn.Module):
    def __init__(self, hidden_state_dim, msg_hidden_dim, msg_dim, iterations):
        super().__init__()
        self.msg_dim = msg_dim
        self.iterations = iterations

        self.broadcast_edge = MessageFunction(hidden_state_dim, msg_hidden_dim, msg_dim)
        self.phys_edge = MessageFunction(hidden_state_dim, msg_hidden_dim, msg_dim)
        # GRU update keeps the hidden state bounded across iterations.
        self.node_updater = nn.GRUCell(msg_dim, hidden_state_dim)

    def forward(self, phys_hidden, goal_hidden, senders, receivers, node_batch):
        goal_msg = self.broadcast_edge(goal_hidden)[node_batch]

        for _ in range(self.iterations):
            sender_states = phys_hidden[senders]
            phys_msgs = self.phys_edge(sender_states)

            msg_accumulator = goal_msg.clone()
            msg_accumulator.index_add_(0, receivers, phys_msgs)

            phys_hidden = self.node_updater(msg_accumulator, phys_hidden)

        return phys_hidden
