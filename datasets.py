"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import os

import numpy as np
import torch
from torch_snippets import Dataset

from data_corruption import TransientMaker
from pre_processing import PreProcessingPipelineBaseline
from utils import ReadDatasets, transform_spectrogram_2D, transform_spectrogram_2D_complete, padd_zeros_spectrogram, \
    get_fid_params, NormalizeData
from basis_and_generator import ReadDataSpectrum
from generate_data_on_the_fly import GenerateDataApply


class DatasetBasisSetOpenTransformer3ChNormalize(Dataset):
    def __init__(self, **kargs: dict) -> None:
        self.path_data = kargs['path_data']
        self.list_path_data = sorted(os.listdir(self.path_data))
        self.norm = kargs['norm']
        self.input_path = [pred for pred in self.list_path_data if ".txt" in pred]
        self.output_path = [pred for pred in self.list_path_data if ".json" in pred]

    def __len__(self) -> int:
        return len(self.list_path_data) // 2

    def __getitem__(self, idx: int) -> (torch.Tensor, np.ndarray):

        self.acess_tmp = f"{self.path_data}/{self.input_path[idx]}"

        sample_for_pred = f"{self.path_data}/{self.input_path[idx]}"
        sample_for_ground_truth = f"{self.path_data}/{self.output_path[idx]}"

        fid_input = ReadDataSpectrum.read_generated(sample_for_pred)
        fid_params = ReadDataSpectrum.ground_truth_json(sample_for_ground_truth)

        fid_spectrogram = transform_spectrogram_2D_complete(fid_input, 0.00025, hope_size=10)
        rows = [127 - i for i in range(0, 32)]
        fid_spectrogram = np.delete(fid_spectrogram, rows, axis=0)
        fid_spectrogram = np.pad(fid_spectrogram, ((0, 0), (0, 18)), mode='constant', constant_values=0)

        fid_params_array = get_fid_params(fid_params)

        self.snr = fid_params_array[-1]
        fid_params_array = fid_params_array[:-1]

        fid_params_array = torch.from_numpy(fid_params_array)

        normalizer = NormalizeData()

        fid_spectrogram_real = np.real(fid_spectrogram)
        fid_spectrogram_real = normalizer.normalize(fid_spectrogram_real, method=self.norm)
        fid_spectrogram_real = torch.from_numpy(fid_spectrogram_real)

        fid_spectrogram_imag = np.imag(fid_spectrogram)
        fid_spectrogram_imag = normalizer.normalize(fid_spectrogram_imag, method=self.norm)
        fid_spectrogram_imag = torch.from_numpy(fid_spectrogram_imag)

        fid_spectrogram_abs = np.abs(fid_spectrogram)
        fid_spectrogram_abs = normalizer.normalize(fid_spectrogram_abs, method=self.norm)
        fid_spectrogram_abs = torch.from_numpy(fid_spectrogram_abs)

        fid_spectrogram_real = fid_spectrogram_real.unsqueeze(0)
        fid_spectrogram_imag = fid_spectrogram_imag.unsqueeze(0)
        fid_spectrogram_abs = fid_spectrogram_abs.unsqueeze(0)

        three_channels_spectrogram = torch.concat([fid_spectrogram_real, fid_spectrogram_imag, fid_spectrogram_abs], dim=0)

        return three_channels_spectrogram.type(torch.FloatTensor), \
            fid_params_array.type(torch.FloatTensor), torch.empty(2)