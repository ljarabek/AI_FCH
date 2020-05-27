import torch
import matplotlib
from typing import TypeVar

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch.nn as nn

class IntervalLoss(nn.Module):
    def __init__(self):
        super(IntervalLoss).__init__()

    #def forward(self, interval_out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        
        #return
