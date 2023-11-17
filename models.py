"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
from utils import set_device
from einops import rearrange

DEVICE = set_device()


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size[0], img_size[1]))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out
class TimmSimpleCNNgelu(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, 24)
                #nn.Linear(512, 1024), nn.ReLU(inplace=True),
                #nn.Linear(1024, 2048)
            )

            self.linear_1 = nn.Linear(n_out, 512)
            self.gelu = nn.GELU()
            self.linear_2 = nn.Linear(512, 256)
            self.linear_3 = nn.Linear(n_out, 128)
            self.linear_4 = nn.Linear(128, 24)
            self.relu = nn.ReLU()

        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.GELU(),
                nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 128), nn.GELU(),
                nn.Linear(128, 24)
            )

    def forward(self, signal_input):
        #descomentar essa linha para rodar cnn
        #output = self.encoder(signal_input)[-1]
        #descomentar essa linha para rodar vit
        x = self.encoder(signal_input)

        out1 = self.linear_1(x)
        out1_gelu = self.gelu(out1)

        out2 = self.linear_2(out1_gelu)
        out1_downsampled = F.avg_pool1d(out1_gelu, kernel_size=2, stride=2)
        out2_residue = out2 + out1_downsampled

        out3 = self.linear_3(x)

        out4 = self.linear_4(out3)

        out4_residue = out4

        return out4_residue