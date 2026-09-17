from encoder import CompositeEncoder, AttentionModule
from core_network import CoreNetwork
from decoder import ActionDecoder

import torch
import torch.nn as nn

class URMA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.joint_composite_enc = CompositeEncoder
        self.joint_attn_module = AttentionModule
        self.feet_composite_enc = CompositeEncoder
        self.feet_attn_module = AttentionModule
        self.core_net = CoreNetwork
        self.action_dec = ActionDecoder

    def forward(self, x):
        pass
