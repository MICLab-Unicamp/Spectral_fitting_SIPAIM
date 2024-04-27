"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
            Marcio Almeida (m240781@dac.unicamp.br)
"""

from losses import MAELoss
from utils import set_device
from models import TimmModelSpectrogram
from save_models import SaveBestModel
from datasets import DatasetBasisSetOpenTransformer3ChNormalize
import torch


save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "TimmModelSpectrogram": TimmModelSpectrogram

    },
    "dataset": {
        "DatasetBasisSetOpenTransformer3ChNormalize": DatasetBasisSetOpenTransformer3ChNormalize
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "MAELoss": MAELoss,

    },
}
