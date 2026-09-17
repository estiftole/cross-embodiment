from encoder import ObservationEncoder
from gnn import GraphNN
from decoder import ActionDecoder

import torch
import torch.nn as nn

class NerveNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        pass
