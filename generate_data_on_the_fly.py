"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

from constants_generate_data import *
import random

class GeneratorFromBasis:
    def __init__(self, basis_path, sample_time):
        self.basis = self.text_to_array_basis(basis_path)
        self.fids = self.basis
        self.noise = np.zeros(len(self.fids[0]), dtype="complex_")
        self.fids_sum = np.zeros(len(self.fids[0]), dtype="complex_")
        self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        self.period = sample_time
        self.sig_to_noise_r = 0

    # apply_scale method:
    # Just multiply each point of fid by the corresponding amplitude
    def apply_scale(self, amplitudes):
        self.fids = np.asarray([basis * amplitude for basis, amplitude in zip(self.basis, amplitudes)])

    # apply_deltas method:
    # Multiply the signal by
    def apply_deltas(self, deltas):
        # Get the total points of the resulting FID and the final time:
        total_points = len(self.fids[0])
        stop = total_points * self.period

        # Create an array with the time values:
        time_range = np.linspace(0, stop, num=total_points)
        terms = []

        for t in time_range:
            # print(deltas)
            terms.append(np.exp((deltas[0] + 2 * np.pi * deltas[1] * 1j) * t) * np.exp(deltas[2] * 1j))
        terms = np.asarray(terms)
        # For each instant of time, calculates the exponentials shown in the equation
        # terms = [np.exp((deltas[0] + 2*np.pi*deltas[1]*1j)*t) * np.exp(deltas[2]*1j) for t in time_range]
        # terms = np.asarray(terms)
        # terms = np.exp((deltas[0] + 2*np.pi*deltas[1]*1j) * time_range) * np.exp(deltas[2]*1j)

        for i in range(len(self.fids)):
            self.fids[i] = self.fids[i] * terms

        # Apply the term to the fids
        # self.fids = [fid * terms for fid in self.fids]

    def sum_fids(self):
        self.fids_sum = np.zeros(len(self.fids[0]), dtype="complex")
        for fid in self.fids:
            self.fids_sum += fid

    def noise_by_SNR(self, snr):
        # self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        # for fid in self.fids:
        #    self.r_fid += fid

        max_peak = np.amax(np.abs(np.fft.fftshift(np.fft.fft(self.fids_sum))))
        std_deviation = max_peak / snr
        noise_temp = np.random.normal(0, np.sqrt(2.0) * std_deviation / 2.0, size=(len(self.r_fid), 2)).view(complex)
        self.noise = []

        for point in noise_temp:
            self.noise.append(point[0])

        self.noise = np.asarray(self.noise)
        self.sig_to_noise_r = max_peak / np.std(self.noise)
        self.noise = np.fft.ifftshift(np.fft.ifft(self.noise))

    def gen_resulting_fid(self):
        # self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        # for fid in self.fids:
        #    self.r_fid += fid
        self.r_fid = self.fids_sum + self.noise

        return self.r_fid

    def reset_fid(self):
        self.fids = self.basis

    @staticmethod
    def text_to_array_basis(path):
        # metab_list = [f for f in listdir(path) if isfile(join(path, f))]

        # print(metab_list)
        # print(metab_list)
        basis_list = []
        for metab in metab_list:
            metab_path = path + "/" + metab

            with open(metab_path, 'r') as file:
                lines = file.readlines()

            met_value = []
            for line in lines:
                values = line.strip().split()
                value = complex(float(values[0]), float(values[1]))
                met_value.append(value)
            basis_list.append(met_value)

        return np.asarray(basis_list)


class GenerateDataApply:
    def __init__(self, low=52,
                 high=82,
                 path_dir="/home/mateus/sipaim_code_math/Marcio_IC/tests/basisset",
                 t_samples = 0.00025):
        self.low = low
        self.high = high
        self.path_dir = path_dir
        self.t_samples = t_samples

    @staticmethod
    def get_random_value(limits):
        r_value = random.uniform(limits[0], limits[1])
        r_value = round(r_value, 4)
        return r_value

    def generate_data(self):

        basis = GeneratorFromBasis(self.path_dir, self.t_samples)
        scales = []

        for i in range(0, 22):
            scale = self.get_random_value(A_ranges[i])
            scales.append(scale)

        damping = self.get_random_value(alpha)
        freq_shift = self.get_random_value(freq)
        phase_shift = self.get_random_value(theta)
        SNR = self.get_random_value((self.high, self.low))

        if SNR < 10:
            print()

        basis.apply_scale(scales)
        basis.apply_deltas([damping, freq_shift, phase_shift])
        # basis.apply_deltas([damping, 3.0, 20.0])
        basis.sum_fids()
        basis.noise_by_SNR(SNR)

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
            elif j == 25:
                param[key] = basis.sig_to_noise_r
            j += 1
        result = basis.gen_resulting_fid()

        return result, param