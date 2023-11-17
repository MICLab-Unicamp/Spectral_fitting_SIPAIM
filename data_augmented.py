import numpy as np
import math

import numpy as np
#from data_augmented import MRSAugmentation


class MRSAugmentation:
    @staticmethod
    def apply_phase_shift(mr_data, zero_order_shift, first_order_shift):
        num_points = len(mr_data)
        time_points = np.arange(num_points)
        phase_shifts = np.exp(1j * (zero_order_shift + first_order_shift * time_points / num_points))
        return mrs_data * phase_shifts

    @staticmethod
    def phase_distortion_augmentation(mr_data):
        zero_order_shift = np.random.uniform(-np.pi, np.pi)
        first_order_shift = np.random.uniform(-20, 20)
        augmented_spectrum = MRSAugmentation.apply_phase_shift(mr_data, zero_order_shift, first_order_shift)

        return augmented_spectrum

    @staticmethod
    def exponential_line_broadening(fid, t, lb=20):
        fs = t[1] - t[0]
        len_fid = fid.shape[0]
        t = np.linspace(0, len_fid/fs, len_fid, endpoint=False)
        # Calculate the exponential decay factor
        decay_factor = np.exp(-lb * t)

        # Apply the decay factor to the FID
        broadened_fid = fid * decay_factor

        return broadened_fid

    #
    # @staticmethod
    # def
    import numpy as np

    def baseline_distortion(signal, num_u_shapes=2, scale=0.01):
        # Create a normalized x array for evaluating the baseline shape
        x = np.linspace(0, 1, signal.size)

        # Calculate the U shape function
        u_shape = (signal * num_u_shapes - np.floor(signal * num_u_shapes) - 0.5) ** 2

        # Scale the U shape amplitude
        baseline = -scale * u_shape + scale / 4

        # Add the baseline to the original signal
        distorted_signal = signal + baseline

        return distorted_signal


import random


class SignalAugment:
    @staticmethod
    def _augment_method(signal):
        methods = [MRSAugmentation.phase_distortion_augmentation]

        # Randomly shuffle the list of methods
        random.shuffle(methods)

        # Randomly select the number of methods to apply
        num_methods = random.randint(1, len(methods))

        augmented_signal = signal.copy()

        augmented_signal = methods[num_methods](augmented_signal)
        return augmented_signal

    @staticmethod
    def random_augmentation(signal_1, signal_2):
        # List all available augmentation methods

        # methods = [MRSAugmentation.baseline_distortion,

        signal_augment_1 = SignalAugment._augment_method(signal_1)
        signal_augment_2 = SignalAugment._augment_method(signal_2)

        return signal_augment_1, signal_augment_2


if __name__ == "__main__":
    mrs_data = np.random.rand(1024) + 1j * np.random.rand(1024)
    num_augmentations = 1
    augmented_data = MRSAugmentation.phase_distortion_augmentation(num_augmentations, mrs_data)

    for i, data in enumerate(augmented_data):
        print(f"Augmented spectrum {i + 1}: {data[:5]}")
