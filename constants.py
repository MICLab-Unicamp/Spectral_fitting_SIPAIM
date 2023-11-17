from losses import RangeMAELoss, MultiSpectrumMAELoss
from losses import RangeMAELoss, GabaRangeMAELoss, TripletLoss, CosineTripletLoss
from utils import set_device
from models import SpectrogramModel, ModelConv2deep, TimmSimpleCNN, UNETBaseline, TimmCNN, TimmMultiResolution, TimmSimpleCNNgelu
from models import TimmSimpleCNNTrack3, TimmTripletModel, TimmSimpleTrack3Transfer
from save_models import SaveBestModel
from datasets import DatasetSpectrogram, DatasetSpectrogramThreeChannels, BasicDatasetBaseline, DatasetBasisSetOpen, DatasetBasisSetOpenTransformer, DatasetBasisSetOpenTransformer3Ch
from datasets import DatasetSpectrogramThreeChannelsTrack2, DatasetSpectrogramThreeChannelsAugmentation, DatasetSpectrogramThreeChannelsTrack3, \
    DatasetManyAugmentations, DatasetTriplets, DatasetSpectrogramThreeChannelsTrack1, DatasetSpectrogramTrack2Augment, DatasetSpectrogramThreeChannelsShiftAugmentation
from datasets import DatasetBasisSetOpen3Ch, DatasetBasisSetOpenTransformer3ChNormalize, DatasetBasisSetOpenTransformer3ChDynamic
from datasets import DatasetTriplets
import torch
from losses import soft_dtw, MAPELoss, sMAPELoss
from Augmentations import BasicTransformationsShift

save_best_model = SaveBestModel()

DEVICE = set_device()

FACTORY_DICT = {
    "model": {
        "SpectrogramModel": SpectrogramModel,
        "ModelConv2deep": ModelConv2deep,
        "TimmSimpleCNN": TimmSimpleCNN,
        "UNETBaseline": UNETBaseline,
        "TimmCNN": TimmCNN,
        "TimmSimpleCNNTrack3": TimmSimpleCNNTrack3,
        "TimmMultiResolution": TimmMultiResolution,
        "TimmTripletModel": TimmTripletModel,
        "TimmSimpleTrack3Transfer": TimmSimpleTrack3Transfer,
        "TimmSimpleCNNgelu": TimmSimpleCNNgelu

    },
    "dataset": {
        "DatasetSpectrogram": DatasetSpectrogram,
        "DatasetBasisSetOpen3Ch": DatasetBasisSetOpen3Ch,
        "DatasetSpectrogramThreeChannels": DatasetSpectrogramThreeChannels,
        "DatasetBasisSetOpenTransformer": DatasetBasisSetOpenTransformer,
        "DatasetBasisSetOpenTransformer3Ch": DatasetBasisSetOpenTransformer3Ch,
        "DatasetBasisSetOpen": DatasetBasisSetOpen,
        "DatasetBasisSetOpenTransformer3ChDynamic": DatasetBasisSetOpenTransformer3ChDynamic,
        "BasicDatasetBaseline": BasicDatasetBaseline,
        "DatasetSpectrogramThreeChannelsTrack2": DatasetSpectrogramThreeChannelsTrack2,
        "DatasetSpectrogramThreeChannelsAugmentation": DatasetSpectrogramThreeChannelsAugmentation,
        "DatasetSpectrogramThreeChannelsTrack3": DatasetSpectrogramThreeChannelsTrack3,
        "DatasetManyAugmentations": DatasetManyAugmentations,
        "DatasetSpectrogramThreeChannelsShiftAugmentation": DatasetSpectrogramThreeChannelsShiftAugmentation,
        "DatasetTriplets": DatasetTriplets,
        "DatasetSpectrogramThreeChannelsTrack1": DatasetSpectrogramThreeChannelsTrack1,
        "DatasetSpectrogramTrack2Augment": DatasetSpectrogramTrack2Augment,
        "DatasetBasisSetOpenTransformer3ChNormalize": DatasetBasisSetOpenTransformer3ChNormalize
    },
    "transformation": {
        "BasicTransformationsShift": BasicTransformationsShift
    },
    "optimizer": {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    },
    "loss": {
        "MAELoss": torch.nn.L1Loss(),
        "MAPELoss": MAPELoss(),
        "sMAPELoss": sMAPELoss(),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
        "MSELoss": torch.nn.MSELoss(),
        "RangeMAELoss": RangeMAELoss(),
        "soft_dtw": soft_dtw,
        "MultiSpectrumMAELoss": MultiSpectrumMAELoss(),
        "GabaRangeMAELoss": GabaRangeMAELoss(),
        "TripletLoss": TripletLoss(1),
        "CosineTripletLoss": CosineTripletLoss()
    },
}
