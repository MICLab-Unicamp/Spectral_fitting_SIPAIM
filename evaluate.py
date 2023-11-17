"""
Maintainer: Mateus Oliveira (mateus.oliveira@icomp.ufam.edu.br)
        Gabriel Dias (g172441@dac.unicamp.br)
        Marcio Almeida (m240781@dac.unicamp.br)
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import torch
from tqdm import tqdm
from metrics import GenerateMetricsPandas
from statistics import mean
from utils import read_yaml, calculate_fqn
from plot_metrics import PlotMetrics
from basis_and_generator import signal_reconstruction
from constants import *
from basis_and_generator import transform_frequency, load_txt_spectrum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predicted neural network MRS")

    parser.add_argument(
        "config_file", type=str, help="config neural network yaml",
    )
    parser.add_argument(
        "weights", type=str, help="WEIGHTs neural network"
    )

    parser.add_argument(
        "path_sample", type=str, help="add path sample .h5"
    )

    parser.add_argument(
        "model_name", type=str, help="add model name (for csv filename)"
    )

    dict_metrics = {"mse": [],
                    "mae": [],
                    "mape": [],
                    "r2": [],
                    "fqn": [],
                    "coefs_mse": np.empty((24,)),
                    "coefs_mae": np.empty((24,)),
                    "coefs_mape": np.empty((24,)),
                    "coefs_r2": np.empty((24,)),
                    "coefs_pred": [],
                    "coefs_ground": []
                    }

    processing_time = []

    args = parser.parse_args()

    sheet_name = args.model_name

    generate_metrics_pandas = GenerateMetricsPandas()

    configs = read_yaml(args.config_file)

    model = TimmSimpleCNNgelu(**configs["model"]["TimmSimpleCNNgelu"])
    load_dict = torch.load(args.weights)

    model.load_state_dict(load_dict["model_state_dict"])

    # DIR VALIDATION
    dataset = DatasetBasisSetOpenTransformer3ChNormalize(**{"path_data": args.path_sample, "norm": "z_norm"})

    list_samples_valid = dataset.input_path
    epsilon = 1e-12  # A small value to prevent division by zero
    output_labels = ['A_Ala', 'A_Asc', 'A_Asp', 'A_Cr', 'A_GABA', 'A_Glc', 'A_Gln', 'A_Glu',
                     'A_Gly', 'A_GPC', 'A_GSH', 'A_Ins', 'A_Lac', 'A_MM', 'A_NAA', 'A_NAAG',
                     'A_PCho', 'A_PCr', 'A_PE', 'A_sIns', 'A_Tau', 'damping', 'freq_s', 'phase_s']
    for i, dir_sample in tqdm(enumerate(list_samples_valid)):

        spectrogram, spectrum_truth, _ = dataset[i]
        input_spec = load_txt_spectrum(f'tests/test/{dataset.input_path[i]}')

        spectrogram = torch.unsqueeze(spectrogram, dim=0)

        pred_labels = model(spectrogram)

        spectrogram = spectrogram.detach().cpu()
        # coefficients pred
        pred_labels = pred_labels.detach().cpu().numpy().squeeze()
        # normalizing only the amplitude coefficients
        pred_labels_amplitude = pred_labels[:21].copy()
        pred_labels_amplitude_norm = (pred_labels_amplitude - pred_labels_amplitude.min()) / (
                pred_labels_amplitude.max() - pred_labels_amplitude.min())
        pred_labels_norm = np.append(pred_labels_amplitude_norm, pred_labels[21:])

        # coefficients truth
        spectrum_truth = spectrum_truth.detach().cpu().numpy().squeeze()
        # normalizing only the amplitude coefficients
        spectrum_truth_amplitude = spectrum_truth[:21].copy()
        spectrum_truth_amplitude_norm = (spectrum_truth_amplitude - spectrum_truth_amplitude.min()) / (
                spectrum_truth_amplitude.max() - spectrum_truth_amplitude.min())
        spectrum_truth_norm = np.append(spectrum_truth_amplitude_norm, spectrum_truth[21:])

        basis_path = "tests/basisset"

        pred_fid = signal_reconstruction(basis_path, list(pred_labels))
        truth_fid = signal_reconstruction(basis_path, list(spectrum_truth))

        # acquiring the signals in the frequency domain
        pred, ground, ppm_axis = transform_frequency(pred_fid, truth_fid)
        fit_residual = pred - input_spec
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
        ground_norm = (ground - ground.min()) / (ground.max() - ground.min())
        # computing the error for the fitted curve
        mse = np.square(np.real(pred_norm) - np.real(ground_norm)).mean()
        mae = np.abs(np.real(pred_norm) - np.real(ground_norm)).mean()
        mape = np.abs((np.real(ground) - np.real(pred)) / (np.real(ground) + epsilon)).mean()
        fqn = calculate_fqn(input_spec, fit_residual, ppm_axis)
        r2 = r2_score(np.real(ground), np.real(pred))

        c_mse = []
        c_mae = []
        c_mape = []

        for j in range(0, len(pred_labels_norm)):
            # computing the loss for each coefficient
            coef_mse = (pred_labels_norm[j] - spectrum_truth_norm[j]) ** 2
            coef_mae = abs(pred_labels_norm[j] - spectrum_truth_norm[j])
            coef_mape = abs((pred_labels[j] - spectrum_truth[j]) / (spectrum_truth[j] + epsilon))

            c_mse.append(coef_mse)
            c_mae.append(coef_mae)
            c_mape.append(coef_mape)

        c_mse = np.array(c_mse)
        c_mae = np.array(c_mae)
        c_mape = np.array(c_mape)

        # FITTING ERROR
        # appending a single number from the fitting error
        dict_metrics["mse"].append(mse)
        print(f'MSE Fitting error: {mse}')
        dict_metrics["mae"].append(mae)
        print(f'MAE Fitting error: {mae}')
        dict_metrics["mape"].append(mape)
        print(f'MAPE Fitting error: {mape}')
        dict_metrics["fqn"].append(fqn)
        print(f'FQN Fitting error: {fqn}')
        dict_metrics["coefs_pred"].append(pred_labels)
        dict_metrics["coefs_ground"].append(spectrum_truth)
        dict_metrics["r2"].append(r2)
        print(f'R2 Fitting error: {r2}')

        # COEFF ERROR
        # appending an array of shape 24 from the computation of the coefficients error
        dict_metrics["coefs_mse"] = np.vstack((dict_metrics["coefs_mse"], c_mse))
        dict_metrics["coefs_mae"] = np.vstack((dict_metrics["coefs_mae"], c_mae))
        dict_metrics["coefs_mape"] = np.vstack((dict_metrics["coefs_mape"], c_mape))
        # discarding the first row used as aux for the initialization
        if i == 0:
            dict_metrics["coefs_mse"] = dict_metrics["coefs_mse"][1:, :]
            dict_metrics["coefs_mae"] = dict_metrics["coefs_mae"][1:, :]
            dict_metrics["coefs_mape"] = dict_metrics["coefs_mape"][1:, :]

        path_save = "Edited_MRS_Challenge/artifacts/evaluate"
        PlotMetrics.plot_pred_vs_ground(input_spec, ground, ppm_axis,
                                        f"{path_save}/input_ground/file_{i}.png",
                                        label_1="input_spec",
                                        label_2="ground_truth")

        PlotMetrics.plot_pred_vs_ground(pred, ground, ppm_axis, f"{path_save}/pred_ground/file_{i}.png",
                                        label_1="pred",
                                        label_2="ground_truth")
        PlotMetrics.plot_fitted_evaluation_beta(input_spec, pred, fit_residual, (ground - pred), ppm_axis,
                                                f"{path_save}/fit_eval/file_{i}.png")

    print(2 * "\n")
    print(f"The average time computation per sample is: {np.array(processing_time).mean() * 1000} ms")
    # computing the mean of the fitting metrics for the evaluation batch
    mean_mse = mean(dict_metrics["mse"])
    mean_mae = mean(dict_metrics["mae"])
    mean_mape = mean(dict_metrics["mape"])
    mean_fqn = mean(dict_metrics["fqn"])
    mean_r2 = mean(dict_metrics["r2"])
    # computing the mean of the coefficient metrics for the evaluation batch
    mean_mse_coeff = np.mean(dict_metrics["coefs_mse"], axis=0)
    mean_mae_coeff = np.mean(dict_metrics["coefs_mae"], axis=0)
    mean_mape_coeff = np.mean(dict_metrics["coefs_mape"], axis=0)
    # computing the r2 for the coefs
    r2_scores = r2_score(np.array(dict_metrics["coefs_ground"]), np.array(dict_metrics["coefs_pred"]),
                         multioutput='raw_values')

    df_fitting_metrics = pd.DataFrame(data=[[mean_mse, mean_mae, mean_mape, mean_fqn, mean_r2]],
                                      columns=["MSE", "MAE", "MAPE", "FQN", "R2"])
    df_fitting_metrics["MAPE"] = df_fitting_metrics["MAPE"] * 100

    df_mse_coeff = pd.DataFrame(data=mean_mse_coeff[np.newaxis, :],
                                columns=output_labels,
                                index=[["MSE"]])
    df_mae_coeff = pd.DataFrame(data=mean_mae_coeff[np.newaxis, :],
                                columns=output_labels,
                                index=[["MAE"]])
    df_mape_coeff = pd.DataFrame(data=100 * mean_mape_coeff[np.newaxis, :],
                                 columns=output_labels,
                                 index=[["MAPE"]])
    df_r2_coeff = pd.DataFrame(data=r2_scores[np.newaxis, :],
                               columns=output_labels,
                               index=[["R2"]])

    df_coef_metrics = pd.concat([df_mse_coeff, df_mae_coeff, df_mape_coeff, df_r2_coeff])

    df_fitting_metrics.to_csv(f"{sheet_name}_fitting_metrics.csv")
    df_coef_metrics.to_csv(f"{sheet_name}_coeff_metrics.csv")

