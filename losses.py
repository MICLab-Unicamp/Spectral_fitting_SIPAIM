# Symmetric mean absolute percentage error
import numpy as np
import torch
import torch.nn as nn
from soft_dtw_cuda import SoftDTW
import torch.nn.functional as F

class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss, self).__init__()

    def forward(self, x, y):
        loss = torch.abs(x - y).mean(dim=1).mean(axis=0)
        return loss