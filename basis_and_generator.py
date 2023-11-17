"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import numpy as np
import random
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
from utils import get_fid_params


class ReadDataSpectrum:
    @staticmethod
    def read_generated(path_name: str):
        return np.loadtxt(path_name, dtype=np.complex128)

    @staticmethod
    def ground_truth_json(path_name: str):
        with open(path_name, 'r') as param_json:
            params = json.load(param_json)
        return params


class BasisRead:
    def __init__(self, basis_path, sample_time=0.00025):
        self.basis_path = basis_path
        self.basis = self.text_to_array_basis()
        self.fids = self.basis
        self.noise = np.zeros(len(self.fids[0]), dtype="complex_")
        self.period = sample_time

    def text_to_array_basis(self) -> np.ndarray:
        #metab_list = [f for f in sorted(listdir(self.basis_path)) if isfile(join(self.basis_path, f))]
        metab_list = ['Ace.txt', 'Ala.txt', 'Asc.txt', 'Asp.txt', 'Cr.txt', 'GABA.txt', 'Glc.txt', 'Gln.txt', 'Glu.txt', 'Gly.txt', 'GPC.txt', 'GSH.txt', 'Ins.txt', 'Lac.txt', 'Mac.txt', 'NAA.txt', 'NAAG.txt', 'PCho.txt', 'PCr.txt', 'PE.txt', 'sIns.txt', 'Tau.txt']
        basis_list = []
        for metab in metab_list:
            metab_path = self.basis_path + "/" + metab

            with open(metab_path, 'r') as file:
                lines = file.readlines()

            met_value = []
            for line in lines:
                values = line.strip().split()
                value = complex(float(values[0]), float(values[1]))
                met_value.append(value)
            basis_list.append(met_value)
        return np.asarray(basis_list)

    def scale(self, scale_factors):
        self.fids = np.asarray([basis * scale_factor for basis, scale_factor in zip(self.basis, scale_factors)])

    def damp_and_shift(self, factors):
        start = 0
        stop = len(self.fids[0]) * self.period
        time_range = np.linspace(start, stop, num=len(self.fids[0]))
        terms = []

        for t in time_range:
            terms.append(np.exp((factors[0] + 2 * np.pi * factors[1] * 1j) * t) * np.exp(factors[2] * 1j))
        terms = np.asarray(terms)

        for i in range(len(self.fids)):
            self.fids[i] = self.fids[i] * terms

    def noise_by_SNR(self, snr):
        self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        for fid in self.fids:
            self.r_fid += fid
        std_deviation = np.amax(np.abs(np.fft.fftshift(np.fft.fft(self.r_fid)))) / snr
        noise_temp = np.random.normal(0, np.sqrt(2.0) * std_deviation / 2.0, size=(len(self.r_fid), 2)).view(complex)
        self.noise = []
        for point in noise_temp:
            self.noise.append(point[0])
        self.noise = np.asarray(self.noise)
        self.noise = np.fft.ifftshift(np.fft.ifft(self.noise))

    def gen_resulting_fid(self):
        self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        for fid in self.fids:
            self.r_fid += fid
        self.r_fid += self.noise

        return self.r_fid

    def reset_fid(self):
        self.fids = self.basis


def get_random_value(limits):
    r_value = random.uniform(limits[0], limits[1])
    r_value = round(r_value, 2)
    return r_value


def generate_data(basis_path, A_ranges, fr):
    basis = BasisRead(basis_path, period)
    scales = []

    param = {
        "A_Ace": 1.0,
        "A_Ala": 1.0,
        "A_Asc": 1.0,
        "A_Asp": 1.0,
        "A_Cr": 1.0,
        "A_GABA": 1.0,
        "A_Glc": 1.0,
        "A_Gln": 1.0,
        "A_Glu": 1.0,
        "A_Gly": 1.0,
        "A_GPC": 1.0,
        "A_GSH": 1.0,
        "A_Ins": 1.0,
        "A_Lac": 1.0,
        "A_MM": 1.0,
        "A_NAA": 1.0,
        "A_NAAG": 1.0,
        "A_PCho": 1.0,
        "A_PCr": 1.0,
        "A_PE": 1.0,
        "A_sIns": 1.0,
        "A_Tau": 1.0,
        "damping": 0.0,
        "freq_s": 0.0,
        "phase_s": 0.0,
    }

    for i in range(0, 22):
        scale = get_random_value(A_ranges[i])
        scales.append(scale)

    damping = get_random_value(alpha)
    freq_shift = get_random_value(freq)
    phase_shift = get_random_value(theta)
    snr = get_random_value((20, 170))

    basis.scale(scales)
    basis.damp_and_shift([damping, freq_shift, phase_shift])
    basis.noise_by_SNR(snr)

    j = 0
    for key in param.keys():
        if j <= 21:
            param[key] = scales[j]
        elif j == 22:
            param[key] = damping
        elif j == 23:
            param[key] = freq_shift
        elif j == 24:
            param[key] = phase_shift
        j += 1
    result = basis.gen_resulting_fid()

    return result, param


# reconstruir o sinal
def signal_reconstruction(basis_path, params):
    basis = BasisRead(basis_path)
    basis.scale(([0] + params)[:22])
    basis.damp_and_shift(([0] + params)[22:])
    reconstructed_fid = basis.gen_resulting_fid()
    return reconstructed_fid


def on_the_fly_signal_reconstruction(basis, params):
    basis.scale(([0] + params)[:22])
    basis.damp_and_shift(params[22:])
    reconstructed_fid = basis.gen_resulting_fid()
    return reconstructed_fid


def plot_fit_results(data, result, frequency):
    residual = data - result

    fig, ax = plt.subplots()

    ax.plot(frequency, data, label="Data", color="black")
    ax.plot(frequency, result, label="Fit", color="red")
    ax.plot(frequency, residual + 20000, label="Residual", color="black")

    plt.legend()
    plt.show()


def transform_frequency(pred, ground):
    frequency_axis = np.linspace(2000, -2000, 2048)

    ppm_axis = 4.65 + frequency_axis / 123.22

    pred_spec = np.fft.fftshift(np.fft.fft(pred))
    truth_spec = np.fft.fftshift(np.fft.fft(ground))
    return pred_spec, truth_spec, ppm_axis


def plot_pred_vs_ground(pred_spec, truth_spec, frequency):
    fig, ax = plt.subplots()

    ax.plot(frequency, truth_spec, label="Ground truth", color="black")
    ax.plot(frequency, pred_spec, label="Predicted signal", color="red")
    # ax.plot(frequency, residual + 20000, label="Residual", color="black")
    plt.xlim((5, 0))
    plt.grid()
    plt.legend()
    plt.show()


def plot_pred_vs_input(pred_spec, input_spec, frequency):
    fig, ax = plt.subplots()

    ax.plot(frequency, input_spec, label="Input signal", color="black")
    ax.plot(frequency, pred_spec, label="Predicted signal", color="red")
    # ax.plot(frequency, residual + 20000, label="Residual", color="black")
    plt.xlim((5, 0))
    plt.grid()
    plt.legend()
    plt.show()


def fft_and_plot(fid):
    spec = np.fft.fftshift(np.fft.fft(fid))
    frequency_axis = np.linspace(2000, -2000, 2048)
    ppm_axis = 4.65 + frequency_axis / 123.22
    plt.plot(ppm_axis, np.real(spec), linewidth=1.0)
    # plt.xlim(4,3)
    plt.xlim(5, 0)
    plt.xlabel("Chemical shift (ppm)")
    plt.ylabel("Amplitude")


def load_txt_spectrum(txt_path):
    fid = ReadDataSpectrum.read_generated(txt_path)
    spec = np.fft.fftshift(np.fft.fft(fid))
    return spec
