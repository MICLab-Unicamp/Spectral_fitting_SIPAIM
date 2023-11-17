"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

from losses import RangeMAELoss
from utils import set_device
from models import TimmSimpleCNNgelu
from save_models import SaveBestModel
from datasets import DatasetBasisSetOpenTransformer3ChNormalize
import torch


save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "TimmSimpleCNNgelu": TimmSimpleCNNgelu

    },
    "dataset": {
        "DatasetBasisSetOpenTransformer3ChNormalize": DatasetBasisSetOpenTransformer3ChNormalize
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "RangeMAELoss": RangeMAELoss(),

    },
}
