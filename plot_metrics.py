"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils import NormalizeData

class PlotMetrics:
    @staticmethod
    def shape_score_comparison(predict, ground_truth, ppm,
                               fig_name="artifacts/shape_score_comparison.png"):

        predict = np.real(predict)
        ground_truth = np.real(ground_truth)

        ## calculate shape score
        # selecting indexes of region of interest
        gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
        glx_max_ind, glx_min_ind = np.amax(np.where(ppm >= 3.6)), np.amin(np.where(ppm <= 3.9))

        # cropping reconstruction to gaba region and performing min-max normalization
        gaba_predict = predict[gaba_min_ind:gaba_max_ind]
        gaba_predict = (gaba_predict - gaba_predict.min()) / (gaba_predict.max() - gaba_predict.min())

        # cropping ground-truth spectrum to gaba region and performing min-max normalization
        gaba_ground_truth = ground_truth[gaba_min_ind:gaba_max_ind]
        gaba_ground_truth = (gaba_ground_truth - gaba_ground_truth.min()) / (
                gaba_ground_truth.max() - gaba_ground_truth.min())

        # cropping ppm array to gaba region
        gaba_crop_ppm = ppm[gaba_min_ind:gaba_max_ind]

        # cropping reconstruction to GLX region and performing min-max normalization
        glx_predict = predict[glx_min_ind:glx_max_ind]
        glx_predict = (glx_predict - glx_predict.min()) / (glx_predict.max() - glx_predict.min())

        # cropping ground-truth spectrum to GLX region and performing min-max normalization
        glx_ground_truth = ground_truth[glx_min_ind:glx_max_ind]
        glx_ground_truth = (glx_ground_truth - glx_ground_truth.min()) / (
                glx_ground_truth.max() - glx_ground_truth.min())
        # cropping ppm array to GLX region
        glx_crop_ppm = ppm[glx_min_ind:glx_max_ind]

        # calculating correlations
        gaba_corr = np.corrcoef(gaba_predict, gaba_ground_truth)[0, 1]
        glx_corr = np.corrcoef(glx_predict, glx_ground_truth)[0, 1]

        # plotting each region and their correlation results
        fig, ax = plt.subplots(1, 2, figsize=(17, 5))

        ax[0].plot(gaba_crop_ppm, gaba_ground_truth, label='ground-truth', c='b')
        ax[0].plot(gaba_crop_ppm, gaba_predict, label='reconstruction', c='r')
        ax[0].set_xlabel("ppm")
        ax[0].invert_xaxis()
        ax[0].set_title(f"GABA Peak - Correlation: {gaba_corr:.3f}")
        ax[0].legend()

        ax[1].plot(glx_crop_ppm, glx_ground_truth, label='ground-truth', c='b')
        ax[1].plot(glx_crop_ppm, glx_predict, label='reconstruction', c='r')
        ax[1].set_xlabel("ppm")
        ax[1].invert_xaxis()
        ax[1].set_title(f"GLX Peak - Correlation: {glx_corr:.3f}")
        ax[1].legend()
        # saving figure
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def fwhm_comparison(predict, ground_truth, ppm,
                        fig_name="artifacts/plot_fwhm_comparison.png"):

        predict = np.real(predict)
        ground_truth = np.real(ground_truth)

        # selecting indexes of gaba peak region
        gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))

        # cropping spectrum for only gaba region and performin min-max normalization
        ppm_crop = ppm[gaba_min_ind:gaba_max_ind]
        # For the ground truth signal
        spec_ground_truth = ground_truth[gaba_min_ind:gaba_max_ind]
        spec_ground_truth = (spec_ground_truth - spec_ground_truth.min()) / (
                spec_ground_truth.max() - spec_ground_truth.min())
        # For the predict signal
        spec_predict = predict[gaba_min_ind:gaba_max_ind]
        spec_predict = (spec_predict - spec_predict.min()) / (spec_predict.max() - spec_predict.min())
        # selecting max point index for the ground truth
        max_peak_ground_truth = spec_ground_truth.max()
        ind_max_peak_ground_truth = np.argmax(spec_ground_truth)
        # selecting max point index for the predict
        max_peak_predict = spec_predict.max()
        ind_max_peak_predict = np.argmax(spec_predict)
        # selecting highest ppm value with point above half the peak value for the ground truth
        left_side_ground_truth = spec_ground_truth[:ind_max_peak_ground_truth]
        left_ind_ground_truth = np.amin(np.where(left_side_ground_truth > max_peak_ground_truth / 2)) + gaba_min_ind
        left_ppm_ground_truth = ppm[left_ind_ground_truth]
        # selecting highest ppm value with point above half the peak value for the predict
        left_side_predict = spec_predict[:ind_max_peak_predict]
        left_ind_predict = np.amin(np.where(left_side_predict > max_peak_predict / 2)) + gaba_min_ind
        left_ppm_predict = ppm[left_ind_predict]
        # selecting lowest ppm value with point above half the peak value for the ground truth
        right_side_ground_truth = spec_ground_truth[ind_max_peak_ground_truth:]
        right_ind_ground_truth = np.amax(
            np.where(right_side_ground_truth > max_peak_ground_truth / 2)) + gaba_min_ind + ind_max_peak_ground_truth
        right_ppm_ground_truth = ppm[right_ind_ground_truth]
        # selecting lowest ppm value with point above half the peak value for the predict
        right_side_predict = spec_predict[ind_max_peak_predict:]
        right_ind_predict = np.amax(
            np.where(right_side_predict > max_peak_predict / 2)) + gaba_min_ind + ind_max_peak_predict
        right_ppm_predict = ppm[right_ind_predict]

        # ploting
        fig, ax = plt.subplots(2, 1, figsize=(12, 9))
        # predict
        ax[0].plot(ppm_crop, spec_predict, label="predict")
        ax[0].hlines(max_peak_predict, right_ppm_predict, left_ppm_predict, colors="r", linestyles="--",
                     label="max peak")
        ax[0].hlines(max_peak_predict / 2, right_ppm_predict, left_ppm_predict, colors="orange", linestyles="--",
                     label="half-maximum")
        ax[0].vlines([right_ppm_predict, left_ppm_predict], 0, max_peak_predict * 1.5, colors="g", linestyles="--",
                     label="half-maximum farthest points")
        ax[0].set_title(f"Predict FWHM: {left_ppm_predict - right_ppm_predict}")
        # ax[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax[0].legend(loc='upper right')
        # groud truth
        ax[1].plot(ppm_crop, spec_ground_truth, label="ground-truth")
        ax[1].hlines(max_peak_ground_truth, right_ppm_ground_truth, left_ppm_ground_truth, colors="r", linestyles="--",
                     label="max peak")
        ax[1].hlines(max_peak_ground_truth / 2, right_ppm_ground_truth, left_ppm_ground_truth, colors="orange",
                     linestyles="--", label="half-maximum")
        ax[1].vlines([right_ppm_ground_truth, left_ppm_ground_truth], 0, max_peak_ground_truth * 1.5, colors="g",
                     linestyles="--", label="half-maximum farthest points")
        ax[1].set_title(f"Ground Truth FWHM: {left_ppm_ground_truth - right_ppm_ground_truth}")
        ax[1].legend(loc='upper right')
        # saving figure
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def mse(predict, ground_truth, ppm,
            fig_name="artifacts/plot_mse.png"):

        predict = np.real(predict)
        ground_truth = np.real(ground_truth)

        # Defining Region of interest and indexes
        min_ppm = 2.5
        max_ppm = 4
        max_ind = np.amax(np.where(ppm >= min_ppm))
        min_ind = np.amin(np.where(ppm <= max_ppm))

        # cropping spectrum to region of interest
        spec_predict = predict[min_ind:max_ind]
        spec_predict = (spec_predict - spec_predict.min()) / (spec_predict.max() - spec_predict.min())
        spec_ground_truth = ground_truth[min_ind:max_ind]
        spec_ground_truth = (spec_ground_truth - spec_ground_truth.min()) / (
                spec_ground_truth.max() - spec_ground_truth.min())
        ppm_crop = ppm[min_ind:max_ind]

        # calculate MSE
        #mse = np.square(spec_ground_truth - spec_predict).mean()

        #MIN-MAX NORM
        #predict = (predict - predict.min()) / (predict.max() - predict.min())
        #ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
        #Z-SCORE NORM
        #predict = (predict - np.mean(predict)) / np.std(predict)
        #ground_truth = (ground_truth - np.mean(ground_truth)) / np.std(ground_truth )
        #Maximum normalization
        predict = predict / np.max(np.abs(predict))
        ground_truth = ground_truth / np.max(np.abs(ground_truth))

        #Defining the max and min global values to set the range in the y axis
        max_global = np.max(predict)
        min_global = np.min(predict)
        if max_global < np.max(ground_truth):
            max_global = np.max(ground_truth)
        if min_global > np.min(ground_truth):
            min_global = np.min(ground_truth)

        #PLOTTING
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        ax[0].plot(ppm, predict, label="reconstruction", color="red")
        ax[0].plot(ppm, ground_truth, label="ground-truth", color="blue")
        #ax[0].plot(ppm, ground_truth - 0.4, label="ground-truth", color="blue")
        ax[0].invert_xaxis()
        ax[0].set_xlabel("ppm")
        ax[0].set_yticks([])
        ax[0].set_ylim(min_global, max_global)
        ax[0].set_title("Spectra Comparison")
        ax[0].fill_between([2.5, 4], min_global, max_global, color="yellow", alpha=0.5)
        ax[0].legend(loc='upper right')

        ax[1].plot(ppm_crop, spec_predict, label="reconstruction", color="red")
        ax[1].plot(ppm_crop, spec_ground_truth, label="ground-truth", color="blue")
        ax[1].fill_between(ppm_crop, spec_predict, spec_ground_truth, color="yellow", alpha=0.5)
        ax[1].invert_xaxis()
        ax[1].set_xlabel("ppm")
        ax[1].set_title("Zoom - Region of MSE Calculation (2.5ppm-4ppm)")

        plt.legend()
        # saving figure
        plt.savefig(fig_name)
        plt.close(fig)
    @staticmethod
    def spectrogram_comparison(predict, ground_truth, t, window_size=256,
                               hope_size=64, window="hann", nfft=None,
                               fig_name="artifacts/spectrogram_comparison.png"):

        def normalize_spectogram(X):
            return (X - X.min()) / (X.max() - X.min())

        def transform_spectrogram_2D(spec, t, window_size=256, hope_size=64, window='hann', nfft=None):

            # calculating the overlap between the windows
            noverlap = window_size - hope_size

            # https://gauss256.github.io/blog/cola.html
            # checking for the NOLA criterion
            if not signal.check_NOLA(window, window_size, noverlap):
                raise ValueError("signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
                                 "STFT not invertible")

            # computing the STFT

            FID = np.fft.ifft(spec.real)

            f, t, Zxx = signal.stft(FID.real, fs=(1 / (t[1] - t[0])), nperseg=window_size, noverlap=noverlap,
                                    return_onesided=True, nfft=nfft)

            # Calculate the magnitude of the STFT
            # The magnitude squared of the STFT yields the spectrogram representation of the power spectral density of the function
            X = np.abs(Zxx) ** 2

            X_dcb = 20 * np.log(X / np.amax(X))

            X_dcb = normalize_spectogram(X_dcb)

            #X_dcb = np.fft.fftshift(X_dcb, axes=0)

            return X_dcb[:, :X_dcb.shape[1] // 2]

        # nfft = 3 * window_size

        X_dcb_predict = transform_spectrogram_2D(predict, t, window_size=window_size, hope_size=hope_size,
                                                 window=window, nfft=nfft)
        X_dcb_ground_truth = transform_spectrogram_2D(ground_truth, t, window_size=window_size, hope_size=hope_size,
                                                      window=window, nfft=nfft)
        fig, axs = plt.subplots(1, 2, figsize=(18, 7))

        title = ["Predict Spectrogram", "Ground Truth Spectrogram"]
        spectrogram = [X_dcb_predict, X_dcb_ground_truth]

        for col in range(2):
            ax = axs[col]
            pcm = ax.pcolormesh(spectrogram[col], cmap="magma", shading="gouraud")
            ax.set_title(title[col])
            ax.set_yticks([])
            ax.set_xticks([])
            plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
        # saving figure
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def snr(predict, ground_truth, ppm,
            fig_name="artifacts/plot_snr.png"):
        # selecting indexes of regions of interest
        gaba_max_ind, gaba_min_ind = np.amax(np.where(ppm >= 2.8)), np.amin(np.where(ppm <= 3.2))
        dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 9.8)), np.amin(np.where(ppm <= 10.8))

        gaba_crop_predict = np.real(predict[gaba_min_ind:gaba_max_ind])
        gaba_crop_predict = (gaba_crop_predict - gaba_crop_predict.min()) / (
            gaba_crop_predict.max() - gaba_crop_predict.min())

        gaba_crop_grount_truth = np.real(ground_truth[gaba_min_ind:gaba_max_ind])
        gaba_crop_grount_truth = (gaba_crop_grount_truth - gaba_crop_grount_truth.min()) / (
            gaba_crop_grount_truth.max() - gaba_crop_grount_truth.min()
        )
        # calculating fitted standard deviation of noise region
        dt_predict = np.polyfit(ppm[dt_min_ind:dt_max_ind], predict[dt_min_ind:dt_max_ind], 2)
        dt_truth = np.polyfit(ppm[dt_min_ind:dt_max_ind], ground_truth[dt_min_ind:dt_max_ind], 2)
        # estimated baseline
        baseline_predict = np.polyval(dt_predict, ppm[dt_min_ind:dt_max_ind])
        baseline_truth = np.polyval(dt_truth, ppm[dt_min_ind:dt_max_ind])
        # estimated noise
        est_noise_predict = np.real(predict[dt_min_ind:dt_max_ind] - np.polyval(dt_predict, ppm[dt_min_ind:dt_max_ind]))
        est_noise_truth = np.real(
            ground_truth[dt_min_ind:dt_max_ind] - np.polyval(dt_truth, ppm[dt_min_ind:dt_max_ind]))


        # plotting
        # fig,ax = plt.subplots(2,2,figsize=(20,10))
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))

        # ax[0][0].plot(ppm[dt_min_ind:dt_max_ind], np.real(predict[dt_min_ind:dt_max_ind]), label="reconstruction",color="red")
        # ax[0][0].plot(ppm[dt_min_ind:dt_max_ind], np.real(ground_truth[dt_min_ind:dt_max_ind]), label="ground-truth",color="blue")
        # ax[0][0].invert_xaxis()
        # ax[0][0].set_xlabel("ppm")
        # ax[0][0].set_title("Noise Region")
        # ax[0][0].legend(loc='upper right')

        ax[1].plot(ppm[gaba_min_ind:gaba_max_ind], gaba_crop_predict, label="reconstruction",
                   color="red")
        ax[1].plot(ppm[gaba_min_ind:gaba_max_ind], gaba_crop_grount_truth,
                   label="ground-truth", color="blue")
        ax[1].invert_xaxis()
        ax[1].set_xlabel("ppm")
        ax[1].set_title("Valid Region")
        ax[1].legend(loc='upper right')

        # ax[1][0].plot(ppm[dt_min_ind:dt_max_ind], np.real(baseline_predict), label = "reconstruction", color = "red")
        # ax[1][0].plot(ppm[dt_min_ind:dt_max_ind], np.real(baseline_truth), label = "ground-truth",color = "blue")
        # ax[1][0].invert_xaxis()
        # ax[1][0].set_xlabel("ppm")
        # ax[1][0].set_title("Estimated Baseline")
        # ax[1][0].legend(loc='upper right')

        ax[0].plot(ppm[dt_min_ind:dt_max_ind], est_noise_predict, label="reconstruction", color="red")
        ax[0].plot(ppm[dt_min_ind:dt_max_ind], est_noise_truth, label="ground-truth", color="blue")
        ax[0].invert_xaxis()
        ax[0].set_xlabel("ppm")
        ax[0].set_title("Estimated Noise")
        ax[0].legend(loc='upper right')

        plt.legend()
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def plot_pred_vs_input(pred_spec, input_spec, frequency, fig_name="artifacts/plot_pred_vs_input.png"):
        fig, ax = plt.subplots()

        ax.plot(frequency, input_spec, label="Input signal", color="black")
        ax.plot(frequency, pred_spec, label="Predicted signal", color="red")
        # ax.plot(frequency, residual + 20000, label="Residual", color="black")
        plt.xlim((5, 0))
        plt.grid()
        plt.legend()
        # saving figure
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def plot_pred_vs_ground(pred_spec, truth_spec, frequency, fig_name,
                            label_1, label_2):

        normalizer = NormalizeData()
        pred_spec = normalizer.min_max_normalize(pred_spec)
        truth_spec = normalizer.min_max_normalize(truth_spec)

        fig, ax = plt.subplots()

        ax.plot(frequency, np.real(truth_spec), label=label_2, color="black")
        ax.plot(frequency, np.real(pred_spec), label=label_1, color="red")
        # ax.plot(frequency, residual + 20000, label="Residual", color="black")
        plt.xlim((5, 0))
        plt.grid()
        plt.legend()
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def plot_fitted_evaluation(data_signal, fitted_signal, fit_residual_signal, ground_truth_minus_fit, ppm_axis,
                               fig_name, spacing=1e4):
        #normalizer = NormalizeData()

        # data_signal = normalizer.min_max_normalize(np.real(data_signal))
        # fitted_signal = normalizer.min_max_normalize(np.real(fitted_signal))
        # fit_residual_signal = normalizer.min_max_normalize(np.real(fit_residual_signal))
        # ground_truth_minus_fit = normalizer.min_max_normalize(np.real(ground_truth_minus_fit))

        data_signal = np.real(data_signal)
        fitted_signal = np.real(fitted_signal)
        fit_residual_signal = np.real(fit_residual_signal)
        ground_truth_minus_fit = np.real(ground_truth_minus_fit)

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(7, 7))

        # Plot the signals with vertical spacing

        ax.plot(ppm_axis, data_signal, label="Input", color="black")
        ax.plot(ppm_axis, fitted_signal + 1.2*spacing, label="Fit", color="red")
        ax.plot(ppm_axis, fit_residual_signal + 3.5 * spacing, label="Fit Residual", color="orange")
        ax.plot(ppm_axis, ground_truth_minus_fit + 3.8 * spacing, label="Ground Truth minus Fit", color="green")

        # Set labels, title, and legend
        ax.set_xlabel("Chemical Shift (ppm)")
        # Remove the y-axis
        ax.yaxis.set_visible(False)
        # ax.set_ylabel("Signal Value")
        ax.set_title("Fitting Evaluation")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.xlim((5, 0))
        # Display the plot
        plt.savefig(fig_name)
        plt.close(fig)
    @staticmethod
    def plot_fitted_evaluation_beta(data_signal, fitted_signal, fit_residual_signal, ground_truth_minus_fit, ppm_axis,
                               fig_name, spacing=1e4):

        data_signal = np.real(data_signal)
        fitted_signal = np.real(fitted_signal)
        fit_residual_signal = np.real(fit_residual_signal)
        ground_truth_minus_fit = np.real(ground_truth_minus_fit)

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the signals with vertical spacing
        ax.plot(ppm_axis, data_signal, label="Input", color="black")
        ax.plot(ppm_axis, fitted_signal + 1.2 * spacing, label="Fit", color="red")
        ax.plot(ppm_axis, fit_residual_signal + 3.5 * spacing, label="Fit Residual", color="orange")
        ax.plot(ppm_axis, ground_truth_minus_fit + 3.8 * spacing, label="Ground Truth minus Fit", color="green")

        # Set labels, title, and legend
        ax.set_xlabel("Chemical Shift (ppm)")
        # Remove the y-axis
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Remove the bottom spine to have only the bottom axis line
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['bottom'].set_linewidth(0.5)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.xlim((5, 0))

        # Save the plot without border and background
        plt.savefig(fig_name, transparent=True, bbox_inches='tight')

        plt.close(fig)