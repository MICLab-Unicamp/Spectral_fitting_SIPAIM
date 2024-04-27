"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
            Marcio Almeida (m240781@dac.unicamp.br)
"""

import time
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from statistics import mean
from utils import read_yaml, calculate_fqn
from plot_metrics import PlotMetrics
from basis_and_generator import signal_reconstruction
from constants import *
from basis_and_generator import transform_frequency, ReadDataSpectrum
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", type=str, help="",
    )
    parser.add_argument(
        "weight", type=str, help="WEIGHTs neural network"
    )

    parser.add_argument(
        "basis_set_path", type=str, help="basis set path"
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

    epsilon = 1e-12
    output_labels = ['A_Ala', 'A_Asc', 'A_Asp', 'A_Cr', 'A_GABA', 'A_Glc', 'A_Gln', 'A_Glu',
                     'A_Gly', 'A_GPC', 'A_GSH', 'A_Ins', 'A_Lac', 'A_MM', 'A_NAA', 'A_NAAG',
                     'A_PCho', 'A_PCr', 'A_PE', 'A_sIns', 'A_Tau', 'damping', 'freq_s', 'phase_s']
    processing_time = []

    args = parser.parse_args()
    basis_set_path = args.basis_set_path
    configs = read_yaml(args.config_file)
    load_dict = torch.load(args.weight)

    save_dir_path = "evaluate_results"
    model_configs = configs["model"]

    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    model.load_state_dict(load_dict["model_state_dict"])

    test_dataset_configs = configs["test_dataset"]
    test_dataset = FACTORY_DICT["dataset"][list(test_dataset_configs)[0]](
        **test_dataset_configs[list(test_dataset_configs.keys())[0]])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, dataset in enumerate(tqdm(test_loader)):

        spectrogram_3ch, spectrum_truth = dataset

        input_spec = ReadDataSpectrum.load_txt_spectrum(f"{test_dataset.path_data}/{test_dataset.input_path[i]}")

        filename = test_dataset.input_path[i].split(".")[0]

        t1 = time.time()

        pred_labels = model(spectrogram_3ch)

        t2 = time.time()
        processing_time.append(t2 - t1)

        spectrogram_3ch = spectrogram_3ch.detach().cpu()

        pred_labels = pred_labels.detach().cpu().numpy().squeeze()
        pred_labels_amplitude = pred_labels[:21].copy()
        pred_labels_amplitude_norm = (pred_labels_amplitude - pred_labels_amplitude.min()) / (
                pred_labels_amplitude.max() - pred_labels_amplitude.min())
        pred_labels_norm = np.append(pred_labels_amplitude_norm, pred_labels[21:])

        spectrum_truth = spectrum_truth.detach().cpu().numpy().squeeze()
        spectrum_truth_amplitude = spectrum_truth[:21].copy()
        spectrum_truth_amplitude_norm = (spectrum_truth_amplitude - spectrum_truth_amplitude.min()) / (
                spectrum_truth_amplitude.max() - spectrum_truth_amplitude.min())
        spectrum_truth_norm = np.append(spectrum_truth_amplitude_norm, spectrum_truth[21:])

        pred_fid = signal_reconstruction(basis_set_path, list(pred_labels))
        truth_fid = signal_reconstruction(basis_set_path, list(spectrum_truth))

        pred, ground, ppm_axis = transform_frequency(pred_fid, truth_fid)
        fit_residual = pred - input_spec
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
        ground_norm = (ground - ground.min()) / (ground.max() - ground.min())

        mse = np.square(np.real(pred_norm) - np.real(ground_norm)).mean()
        mae = np.abs(np.real(pred_norm) - np.real(ground_norm)).mean()
        mape = np.abs((np.real(ground) - np.real(pred)) / (np.real(ground) + epsilon)).mean()
        fqn = calculate_fqn(input_spec, fit_residual, ppm_axis)
        r2 = r2_score(np.real(ground), np.real(pred))

        c_mse = []
        c_mae = []
        c_mape = []

        for j in range(0, len(pred_labels_norm)):
            coef_mse = (pred_labels_norm[j] - spectrum_truth_norm[j]) ** 2
            coef_mae = abs(pred_labels_norm[j] - spectrum_truth_norm[j])
            coef_mape = abs((pred_labels[j] - spectrum_truth[j]) / (spectrum_truth[j] + epsilon))

            c_mse.append(coef_mse)
            c_mae.append(coef_mae)
            c_mape.append(coef_mape)

        c_mse = np.array(c_mse)
        c_mae = np.array(c_mae)
        c_mape = np.array(c_mape)

        print(f" \n {filename} results:")
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

        dict_metrics["coefs_mse"] = np.vstack((dict_metrics["coefs_mse"], c_mse))
        dict_metrics["coefs_mae"] = np.vstack((dict_metrics["coefs_mae"], c_mae))
        dict_metrics["coefs_mape"] = np.vstack((dict_metrics["coefs_mape"], c_mape))

        if i == 0:
            dict_metrics["coefs_mse"] = dict_metrics["coefs_mse"][1:, :]
            dict_metrics["coefs_mae"] = dict_metrics["coefs_mae"][1:, :]
            dict_metrics["coefs_mape"] = dict_metrics["coefs_mape"][1:, :]

        os.makedirs(save_dir_path, exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "input_ground"), exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "pred_ground"), exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "fit_eval"), exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "metrics"), exist_ok=True)

        PlotMetrics.spectrum_comparison(input_spec, ground, ppm_axis,
                                        label_1="input_spec",
                                        label_2="ground_truth",
                                        fig_name=f"{save_dir_path}/input_ground/{filename}.png")

        PlotMetrics.spectrum_comparison(pred, ground, ppm_axis,
                                        label_1="pred",
                                        label_2="ground_truth",
                                        fig_name=f"{save_dir_path}/pred_ground/{filename}.png")

        PlotMetrics.plot_fitted_evaluation(input_spec, pred, fit_residual, (ground - pred), ppm_axis,
                                           f"{save_dir_path}/fit_eval/{filename}.png")

    print(2 * "\n")
    print(f"The average time computation per sample is: {np.array(processing_time).mean() * 1000} ms")

    mean_mse = mean(dict_metrics["mse"])
    mean_mae = mean(dict_metrics["mae"])
    mean_mape = mean(dict_metrics["mape"])
    mean_fqn = mean(dict_metrics["fqn"])
    mean_r2 = mean(dict_metrics["r2"])

    mean_mse_coeff = np.mean(dict_metrics["coefs_mse"], axis=0)
    mean_mae_coeff = np.mean(dict_metrics["coefs_mae"], axis=0)
    mean_mape_coeff = np.mean(dict_metrics["coefs_mape"], axis=0)

    r2_scores = r2_score(np.array(dict_metrics["coefs_ground"]), np.array(dict_metrics["coefs_pred"]),
                         multioutput='raw_values')

    df_mean_fitting_metrics = pd.DataFrame(data=[[mean_mse, mean_mae, mean_mape, mean_fqn, mean_r2]],
                                           columns=["MSE", "MAE", "MAPE", "FQN", "R2"])
    df_mean_fitting_metrics["MAPE"] = df_mean_fitting_metrics["MAPE"] * 100

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

    df_mean_fitting_metrics.to_csv(f"{save_dir_path}/metrics/df_mean_fitting_metrics.csv")
    df_coef_metrics.to_csv(f"{save_dir_path}/metrics/coefficient_metrics.csv")
