"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import numpy as np
from data_augmented import SignalAugment
from utils import transform_spectrogram_2D, NormalizeData
import torch


class PreProcessingPipelineBaseline:
    @staticmethod
    def generate_noise_spectrum(signal_fid: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft(signal_fid, axis=1), axes=1)

    @staticmethod
    def get_real_signal(signal: np.ndarray) -> np.ndarray:
        return np.real(signal)

    @staticmethod
    def normalize_signal_challenge(signal_noise, signal_label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        signal_noise_max = signal_noise.max(axis=(1, 2), keepdims=True)
        signal_noise_mean = signal_noise[:, :].min(axis=(1, 2), keepdims=True)

        signal_label_max = signal_label.max(axis=(1), keepdims=True)
        signal_label_mean = signal_label[:, :].min(axis=(1), keepdims=True)

        x = (signal_noise - signal_noise_mean) / (signal_noise_max - signal_noise_mean)

        y = (signal_label - signal_label_mean) / (signal_label_max - signal_label_mean)

        x = np.expand_dims(x, axis=3)

        return x, y

    @staticmethod
    def train_test_split_data(x: np.ndarray,
                              y: np.ndarray,
                              ppm: np.ndarray,
                              percentage_split=0.8) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train = x[:int(x.shape[0] * percentage_split)]
        x_test = x[int(x.shape[0] * percentage_split):]

        y_train = y[:int(y.shape[0] * percentage_split)]
        y_test = y[int(y.shape[0] * percentage_split):]

        ppm_train = ppm[:int(ppm.shape[0] * percentage_split)]
        ppm_test = ppm[int(ppm.shape[0] * percentage_split):]

        return x_train, y_train, x_test, y_test, ppm_train, ppm_test

    @staticmethod
    def subtract_spectrum_s(signal_1: np.ndarray, t: np.ndarray) -> np.ndarray:
        if len(signal_1.shape) == 3:
            signal_1 = np.fft.fftshift(np.fft.ifft(signal_1, axis=1), axes=1)
            return signal_1[:, :, 1] - signal_1[:, :, 0]

        if len(signal_1.shape) == 4:

            fid_on = signal_1.mean(axis=3)[:, :, 1]
            fid_off = signal_1.mean(axis=3)[:, :, 0]

            if signal_1.shape[1] == 2048:
                signal_spectrum_1 = transform_spectrogram_2D(fid_on, t,
                                                                window_size=256,
                                                                hope_size=10,
                                                                nfft=446)

                signal_spectrum_2 = transform_spectrogram_2D(fid_off, t,
                                                                window_size=256,
                                                                hope_size=10,
                                                                nfft=446)
            else:
                signal_spectrum_1 = transform_spectrogram_2D(fid_on, t,
                                                                window_size=256,
                                                                hope_size=20,
                                                                nfft=446)
                signal_spectrum_2 = transform_spectrogram_2D(fid_off, t,
                                                                window_size=256,
                                                                hope_size=20,
                                                                nfft=446)

            result_signal = signal_spectrum_1 - signal_spectrum_2
            result_signal = np.abs(result_signal) / (np.max(np.abs(result_signal)))

            return result_signal

    @staticmethod
    def subtract_spectrum_s_shift_aug(signal_1: np.ndarray, t: np.ndarray, shift: int) -> np.ndarray:
        # SPEC 1D TRUTH FOR TRACKS 2 AND 3
        if len(signal_1.shape) == 2:
            signal_1_shifted = np.roll(signal_1, shift=shift, axis=1)
            return signal_1_shifted

        # FID TRUTH FOR TRACK 1 -> COMPUTING RESPECTIVE SPEC 1D
        if len(signal_1.shape) == 3:
            signal_1 = np.fft.fftshift(np.fft.ifft(signal_1, axis=1), axes=1)
            result_signal = signal_1[:, :, 1] - signal_1[:, :, 0]
            result_signal = result_signal / (np.max(np.abs(result_signal), axis=1, keepdims=True))
            result_signal_shifted = np.roll(result_signal, shift=shift, axis=1)
            return result_signal_shifted

        if len(signal_1.shape) == 4:

            fid_on = signal_1.mean(axis=3)[:, :, 1]
            fid_off = signal_1.mean(axis=3)[:, :, 0]

            if shift != 0:
                signal_spec1d_1 = np.fft.fftshift(np.fft.ifft(fid_on))[0, :]
                signal_spec1d_1_shifted = np.roll(signal_spec1d_1, shift=shift)
                fid_shifted_1 = np.fft.fft(np.fft.ifftshift(signal_spec1d_1_shifted))

                signal_spec1d_2 = np.fft.fftshift(np.fft.ifft(fid_off)[0, :])
                signal_spec1d_2_shifted = np.roll(signal_spec1d_2, shift=shift)
                fid_shifted_2 = np.fft.fft(np.fft.ifftshift(signal_spec1d_2_shifted))
            else:
                fid_shifted_1 = fid_on[0, :]
                fid_shifted_2 = fid_off[0, :]

            if signal_1.shape[1] == 2048:

                signal_spectrum_1 = transform_spectrogram_2D(fid_shifted_1, t,
                                                             window_size=256,
                                                             hope_size=10,
                                                             nfft=446)

                signal_spectrum_2 = transform_spectrogram_2D(fid_shifted_2, t,
                                                             window_size=256,
                                                             hope_size=10,
                                                             nfft=446)
            else:
                signal_spectrum_1 = transform_spectrogram_2D(fid_shifted_1, t,
                                                             window_size=256,
                                                             hope_size=20,
                                                             nfft=446)
                signal_spectrum_2 = transform_spectrogram_2D(fid_shifted_2, t,
                                                             window_size=256,
                                                             hope_size=20,
                                                             nfft=446)

            result_signal = signal_spectrum_1 - signal_spectrum_2
            result_signal = np.abs(result_signal) / (np.max(np.abs(result_signal)))

            return result_signal

    @staticmethod
    def subtract_spectrum_s_fft(signal_1, transform, sample_rate=2048):
        # Calculate the mean of the last dimension along axis=3
        signal_1_mean = signal_1.mean(dim=3)

        # Slice the resulting tensor to select the desired channels and rows
        signal_2_mean = signal_1_mean[:, :, 0]
        signal_1_mean = signal_1_mean[:, :, 1]

        # Calculate the subtracted residual
        subtract_residual = signal_1_mean - signal_2_mean

        # Permute the tensor to swap width and height axes
        # subtract_residual = subtract_residual.permute(0, 1, 3, 2)

        # Apply 2D inverse FFT and shift the zero-frequency component to the center
        # subtract_residual = torch.fft.fftshift(torch.fft.ifft2(subtract_residual,
        #                                                        dim=1),
        #                                                        dim=1)

        # subtract_residual = torch.fft.fft(subtract_residual, dim=1)
        subtract_residual = np.fft.fftshift(np.fft.fft(subtract_residual, axis=1))

        subtract_residual = torch.from_numpy(subtract_residual)
        # Permute the tensor to swap back the width and height axes
        # subtract_residual = subtract_residual.permute(0, 1, 3, 2)

        # Apply the given transform to the subtracted residual
        shift_fft_subtract_residual = transform(subtract_residual.unsqueeze(0),
                                                sample_rate=sample_rate)

        # Apply 1D inverse FFT to the transformed residual to obtain two FIDs
        # two_FID = torch.fft.ifft(shift_fft_subtract_residual, dim=2)
        two_FID = np.fft.ifft(shift_fft_subtract_residual, axis=2)
        return two_FID

    # def augment_spectrum(self):
