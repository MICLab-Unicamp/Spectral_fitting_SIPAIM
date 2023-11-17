"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RangeMAELoss(nn.Module):

    def __init__(self):
        super(RangeMAELoss, self).__init__()

    def forward(self, x, y):
        loss = torch.abs(x - y).mean(dim=1).mean(axis=0)
        return loss